import random
from itertools import cycle
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from transformers import PreTrainedModel, PreTrainedTokenizer


def ratio_auc(members: Sequence[float], non_members: Sequence[float]):
    y = []
    y_true = []

    y.extend(members)
    y.extend(non_members)

    y_true.extend([0] * len(members))
    y_true.extend([1] * len(non_members))

    fpr, tpr, _ = roc_curve(y_true, y)
    roc_auc = auc(fpr, tpr)

    return roc_auc


def jaccard_similarity(set1: Iterable, set2: Iterable):
    set1 = set(set1)
    set2 = set(set2)

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union


def min_k_prob(model: PreTrainedModel, token_ids: torch.Tensor, attention_mask: torch.Tensor, k: int = 20):
    with torch.no_grad():
        labels = token_ids.clone()
        outputs = model(token_ids, attention_mask=attention_mask)

        shift_logits = outputs.logits[..., :-1, :].contiguous().view(-1, model.config.vocab_size)
        shift_attention_mask = attention_mask[..., :-1]
        shift_targets = labels[..., 1:]

        shift_targets[shift_attention_mask == 0] = -100

        # we add minus here, because `F.cross_entropy` is a loss, and we need the log-probability.
        # loss goes down when probability goes up.
        token_logp = -F.cross_entropy(shift_logits, shift_targets.contiguous().view(-1), reduction="none")
        token_logp = token_logp.view(token_ids.shape[0], -1)
        token_logp = token_logp.detach().cpu().numpy()

        sorted_probas = np.sort(token_logp, axis=1)
        sorted_probas = sorted_probas[:, : int(k / 100 * sorted_probas.shape[1])]
        k_min_proba = np.mean(sorted_probas, axis=1)

    return k_min_proba


def compute_perplexity(
    model: PreTrainedModel,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    ignore_prefix: Optional[int] = None,
):
    with torch.no_grad():
        labels = token_ids.clone()

        outputs = model(token_ids, attention_mask=attention_mask)

        shift_logits = outputs.logits[..., :-1, :].contiguous().view(-1, model.config.vocab_size)
        shift_attention_mask = attention_mask[..., :-1]
        shift_targets = labels[..., 1:]

        shift_targets[shift_attention_mask == 0] = -100

        loss = F.cross_entropy(shift_logits, shift_targets.contiguous().view(-1), reduction="none")
        loss = loss.view(token_ids.shape[0], -1)

        if ignore_prefix:
            loss = loss[:, ignore_prefix:]
            shift_attention_mask = shift_attention_mask[:, ignore_prefix:]

        loss = loss.sum(axis=1) / shift_attention_mask.sum(axis=1)

        return np.exp(loss.detach().cpu().numpy())


def insert_one(
    base_seq: Sequence,
    insert_seq: Sequence,
    n_rep: int,
    insert_at_the_end=True,
):
    """
    Injects the `insert_seq` into the `base_seq` at random positions.

    Args:
        base_seq (Sequence): The base sequence to inject into.
        insert_seq (Sequence): The sequence to be injected.
        n_rep (int): The number of times to inject the `insert_seq`.
        insert_at_the_end (bool, optional): Whether to insert the `insert_seq` at the end of the `base_seq`.
                                            Defaults to True.

    Returns:
        list: The modified sequence with the `insert_seq` injected.
    """

    if insert_at_the_end:
        n_rep -= 1

    if n_rep < 0:
        raise ValueError("n_rep must be greater than or equal to 0")

    inject_indices = sorted(random.sample(range(len(base_seq)), k=n_rep) + [len(base_seq)])
    ret = list(base_seq[: inject_indices[0]])
    for i in range(0, len(inject_indices) - 1):
        curr_idx = inject_indices[i]
        nex_idx = inject_indices[i + 1]

        ret.extend(insert_seq)
        ret.extend(base_seq[curr_idx:nex_idx])

    if insert_at_the_end:
        ret.extend(insert_seq)

    return ret


def split_and_insert(
    base_seq: Iterable[int],
    insert_seq: Iterable[int],
    chunk_size: int = 2048,
    n_rep: int = 1000,
    n_rep_per_chunk: int = 1,
    insert_at_the_end: bool = True,
) -> List[List[int]]:
    """
    Splits the base sequence into chunks and injects the insert sequence into some of the chunks.

    Args:
        base_seq (Iterable[int]): The base sequence to be split into chunks.
        insert_seq (Iterable[int]): The sequence to be injected into some of the chunks.
        chunk_size (int, optional): The size of each chunk. Defaults to 2048.
        n_rep (int, optional): Total number of times the insert sequence should be injected. Defaults to 1000.
        n_rep_in_context (int, optional): The number of times the insert sequence should be injected within the
        context window. Defaults to 1.
        insert_at_the_end (bool, optional): Whether to inject the insert sequence at the end of the chunk.

    Returns:
        dict: A dictionary containing the generated chunks.
    """

    small_chunk_size = chunk_size - len(insert_seq) * n_rep_per_chunk
    n_chunks_with_canaries = n_rep // n_rep_per_chunk

    total_output_tokens = max(chunk_size * n_rep, len(base_seq) + len(insert_seq) * n_rep)
    n_chunks = total_output_tokens // chunk_size

    clean_chunks = [base_seq[i : i + chunk_size] for i in range(0, len(base_seq) - chunk_size + 1, chunk_size)]
    small_chunks = [
        base_seq[i : i + small_chunk_size] for i in range(0, len(base_seq) - small_chunk_size + 1, small_chunk_size)
    ]

    random.shuffle(clean_chunks)
    random.shuffle(small_chunks)

    clean_chunk_gen = cycle(clean_chunks)
    small_chunk_gen = cycle(small_chunks)

    chunks = []
    for _ in range(n_chunks_with_canaries):
        small_chunk = next(small_chunk_gen)
        chunk = insert_one(small_chunk, insert_seq, n_rep_per_chunk, insert_at_the_end=insert_at_the_end)
        chunks.append(chunk)

    n_chunks_no_canaries = n_chunks - n_chunks_with_canaries
    for _ in range(n_chunks_no_canaries):
        chunks.append(next(clean_chunk_gen))

    return chunks


def compute_extractability(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor = None,
    secret_len: int = 1,
):
    total_length = input_ids.shape[1]
    prompt_lenght = total_length - secret_len

    prompt_tokens = input_ids[:, :prompt_lenght]
    attention_mask = attention_mask[:, :prompt_lenght]
    secret_tokens = input_ids[:, prompt_lenght:]

    greedy_output = model.generate(
        inputs=prompt_tokens,
        max_length=total_length,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        attention_mask=attention_mask,
    )

    accuracy = greedy_output[:, prompt_lenght:] == secret_tokens
    return accuracy.all(dim=1).detach().cpu().numpy()


def compute_token_logit(model: PreTrainedModel, token_ids: torch.Tensor, attention_mask: torch.Tensor):
    with torch.no_grad():
        labels = token_ids.clone()

        outputs = model(token_ids, attention_mask=attention_mask)

        shift_logits = outputs.logits[..., :-1, :].contiguous().view(-1, model.config.vocab_size)
        shift_attention_mask = attention_mask[..., :-1]
        shift_targets = labels[..., 1:]

        shift_targets[shift_attention_mask == 0] = -100

        predicted_probas = -F.nll_loss(
            F.softmax(shift_logits, dim=-1), shift_targets.contiguous().view(-1), reduction="none"
        )
        predicted_probas = predicted_probas.view(token_ids.shape[0], -1)
        all_logits = torch.logit(predicted_probas, eps=1e-6)
        all_logits[shift_attention_mask == 0] = 0
        mean_logits = all_logits.sum(axis=1) / shift_attention_mask.sum(axis=1)

        return mean_logits.detach().cpu().numpy()


def compute_lr_acc(non_member_ppl, member_ppl):
    X = np.array(non_member_ppl + member_ppl).reshape(-1, 1)
    y = [0] * len(non_member_ppl) + [1] * len(member_ppl)
    clf = LogisticRegression(random_state=0).fit(X, y)
    y_pred = clf.predict(X)
    acc = sum(y == y_pred) / len(y)
    return 100 * acc


def fit_distr(
    x: Iterable[float],
    do_plot: bool = False,
    **kwargs,
):
    mu = np.mean(x)
    sigma = np.std(x)

    if do_plot:
        plt.hist(x, bins=100, density=True, alpha=0.5, **kwargs)
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
        y = norm.pdf(x, mu, sigma)
        plt.plot(x, y, "-", linewidth=2, **kwargs)
        # plt.show()

    return mu, sigma


def fit_ppl_distr(
    df: pd.DataFrame,
    target_ppl: int,
    reference_ppl_col: str = "llama_ppl",
    target_ppl_col: str = "croissant_ppl",
    do_plot: bool = False,
    mini_bucket_width: int = 1,
):
    out_distr = df[
        (df[reference_ppl_col].astype(int) >= target_ppl - mini_bucket_width / 2)
        & (df[reference_ppl_col].astype(int) <= target_ppl + mini_bucket_width / 2)
    ][target_ppl_col]

    # loss, not perplexity
    # this has a potential to fail spectacularly
    if target_ppl_col.endswith("ppl"):
        out_distr = np.log(out_distr)

    mu = np.mean(out_distr)
    sigma = np.std(out_distr)

    if do_plot:
        plt.hist(out_distr, bins=100, density=True, alpha=0.5)
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
        y = norm.pdf(x, mu, sigma)
        plt.plot(x, y, "r-", linewidth=2)
        # plt.show()

    return mu, sigma


def reverse_lira_confidence(
    x: Iterable[float],
    out_distr_df: pd.DataFrame,
    target_ppl: int,
    seq_len: int,
    mini_bucket_width: int = 1,
    reference_ppl_col: str = "llama_ppl",
    target_ppl_col: str = "croissant_ppl",
):
    df = out_distr_df[out_distr_df.seq_len == seq_len]
    mu, sigma = fit_ppl_distr(
        df=df,
        target_ppl=target_ppl,
        mini_bucket_width=mini_bucket_width,
        reference_ppl_col=reference_ppl_col,
        target_ppl_col=target_ppl_col,
    )
    return norm.cdf(x, mu, sigma)


def double_sided_lira_confidence(
    x: Iterable[float],
    out_distr_df: pd.DataFrame,
    in_distr_df: pd.DataFrame,
    target_ppl: int,
    seq_len: int,
    mini_bucket_width: int = 1,
    reference_ppl_col: str = "llama_ppl",
    target_ppl_col: str = "croissant_ppl",
):
    out_df = out_distr_df[out_distr_df.seq_len == seq_len]
    in_df = in_distr_df[in_distr_df.seq_len == seq_len]

    mu_out, sigma_out = fit_ppl_distr(
        df=out_df,
        target_ppl=target_ppl,
        mini_bucket_width=mini_bucket_width,
        reference_ppl_col=reference_ppl_col,
        target_ppl_col=target_ppl_col,
    )

    mu_in, sigma_in = fit_ppl_distr(
        df=in_df,
        target_ppl=target_ppl,
        mini_bucket_width=mini_bucket_width,
        reference_ppl_col=reference_ppl_col,
        target_ppl_col=target_ppl_col,
    )

    return norm.pdf(x, mu_in, sigma_in) / norm.pdf(x, mu_out, sigma_out)


def compute_perplexity_df(
    llama_model,
    croissant_model,
    llama_tokenizer,
    croissant_tokenizer,
    raw_canaries,
    batch_size: int = 32,
    croissant_device: str = "cuda:0",
    llama_device: str = "cuda:1",
    ignore_prefix: Optional[int] = None,
):
    df_res = pd.DataFrame()

    for i in range(0, len(raw_canaries), batch_size):
        batch = raw_canaries[i : i + batch_size]

        canary_tokens_croissant = croissant_tokenizer.batch_encode_plus(
            list(batch), return_tensors="pt", padding="longest"
        ).to(croissant_device)
        canary_tokens_llama = llama_tokenizer.batch_encode_plus(list(batch), return_tensors="pt", padding="longest").to(
            llama_device
        )

        croissant_ppl = compute_perplexity(
            croissant_model,
            canary_tokens_croissant.input_ids[:, 1:],
            canary_tokens_croissant.attention_mask[:, 1:],
            ignore_prefix=ignore_prefix,
        )
        llama_ppl = compute_perplexity(
            llama_model,
            canary_tokens_llama.input_ids[:, 1:],
            canary_tokens_llama.attention_mask[:, 1:],
            ignore_prefix=ignore_prefix,
        )

        # croissant_norm_loss = compute_token_logit(
        #     croissant_model, canary_tokens_croissant.input_ids, canary_tokens_croissant.attention_mask
        # )
        # llama_norm_loss = compute_token_logit(
        #     llama_model, canary_tokens_llama.input_ids, canary_tokens_llama.attention_mask
        # )

        # extractability = compute_extractability(
        #     model=croissant_model,
        #     tokenizer=croissant_tokenizer,
        #     input_ids=canary_tokens_croissant.input_ids,
        #     attention_mask=canary_tokens_croissant.attention_mask,
        # )

        croissant_token_len = canary_tokens_croissant.attention_mask[:, 1:].sum(axis=1).detach().cpu().numpy()

        df_tmp = pd.DataFrame(
            {
                "croissant_ppl": croissant_ppl,
                "llama_ppl": llama_ppl,
                # "croissant_norm_loss": croissant_norm_loss,
                # "llama_norm_loss": llama_norm_loss,
                # "extractability": extractability,
                "croissant_token_len": croissant_token_len,
            }
        )
        df_res = pd.concat([df_res, df_tmp])

    return df_res
