from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, roc_curve
from transformers import PreTrainedModel


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


def compute_perplexity_df(
    llama_model,
    croissant_model,
    llama_tokenizer,
    croissant_tokenizer,
    raw_traps,
    batch_size: int = 32,
    croissant_device: str = "cuda:0",
    llama_device: str = "cuda:1",
    ignore_prefix: Optional[int] = None,
):
    df_res = pd.DataFrame()

    for i in range(0, len(raw_traps), batch_size):
        batch = raw_traps[i : i + batch_size]

        trap_tokens_croissant = croissant_tokenizer.batch_encode_plus(
            list(batch), return_tensors="pt", padding="longest"
        ).to(croissant_device)
        trap_tokens_llama = llama_tokenizer.batch_encode_plus(list(batch), return_tensors="pt", padding="longest").to(
            llama_device
        )

        croissant_ppl = compute_perplexity(
            croissant_model,
            trap_tokens_croissant.input_ids[:, 1:],
            trap_tokens_croissant.attention_mask[:, 1:],
            ignore_prefix=ignore_prefix,
        )
        llama_ppl = compute_perplexity(
            llama_model,
            trap_tokens_llama.input_ids[:, 1:],
            trap_tokens_llama.attention_mask[:, 1:],
            ignore_prefix=ignore_prefix,
        )

        minkprob = min_k_prob(
            croissant_model,
            trap_tokens_croissant.input_ids[:, 1:],
            trap_tokens_croissant.attention_mask[:, 1:],
        )

        croissant_token_len = trap_tokens_croissant.attention_mask[:, 1:].sum(axis=1).detach().cpu().numpy()

        df_tmp = pd.DataFrame(
            {
                "croissant_ppl": croissant_ppl,
                "llama_ppl": llama_ppl,
                "croissant_token_len": croissant_token_len,
                "minkprob": minkprob
            }
        )
        df_res = pd.concat([df_res, df_tmp])

    return df_res
