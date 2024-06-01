import argparse
import pickle
import random
import re
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          LlamaForCausalLM, LlamaTokenizer, PreTrainedModel)

random.seed(42)


def find_context(doc_text: str, trap: str, tokenizer,
                 x_context_tokens: int):
    '''
    Based on the llama tokenizer, get X tokens as context, to be translated back to text
    '''

    # sample one start loc in the string of the book
    try:
        all_start_locs = find_idx_occ(doc_text, trap)
        start_loc_selected = random.sample(all_start_locs, 1)[0]
    except Exception as e:
        traceback.print_exc()

        all_start_locs = find_idx_occ(doc_text, trap[:20])
        start_loc_selected = random.sample(all_start_locs, 1)[0]

    # get all text before this loc + the trap
    book_before_trap = doc_text[:start_loc_selected]

    # remove the occurrences of the traps in the book before the traps
    book_before_trap_clean = re.sub(re.escape(trap), " ", book_before_trap)

    # tokenize all this text
    tokenized_before_trap = tokenizer.encode(book_before_trap_clean)

    # take the right amount of context
    # making the decision for now to take the same number of context tokens for every seq len
    context_tokenized = tokenized_before_trap[-x_context_tokens:]
    context_text = tokenizer.decode(context_tokenized)

    return context_text


def find_idx_occ(string, substring):
    escaped_substring = re.escape(substring)
    return [m.start() for m in re.finditer(f"{escaped_substring}", string)]


def compute_ppl_in_context(
        model: PreTrainedModel,
        device: str,
        tokenizer,
        context_text: str,
        trap_text: str):

    input_ids = tokenizer.encode(context_text + trap_text, return_tensors="pt").to(device)
    n_trap_tokens = len(tokenizer.encode(trap_text)) - 1  # subtracting one for SOS

    with torch.no_grad():
        total_length = input_ids.shape[1]
        prompt_length = total_length - n_trap_tokens

        outputs = model(input_ids)

        shift_logits = outputs.logits[..., prompt_length - 1: -1, :].contiguous().view(-1, model.config.vocab_size)
        shift_targets = input_ids[..., prompt_length:].contiguous().view(-1)

        loss = F.cross_entropy(shift_logits, shift_targets.view(-1), reduction="none")
        loss = loss.view(input_ids.shape[0], -1)
        loss = loss.mean(dim=1)

        return np.exp(loss.detach().cpu().numpy())[0]


def main(args):
    x_context_tokens = args.X_tokens

    llama_tokenizer = LlamaTokenizer.from_pretrained(args.path_to_tokenizer, torch_dtype=torch.float16)
    llama_model = LlamaForCausalLM.from_pretrained(args.path_to_model)

    tokenizer = AutoTokenizer.from_pretrained("croissantllm/base_190k")
    model = AutoModelForCausalLM.from_pretrained("croissantllm/base_190k")

    tokenizer.pad_token = tokenizer.eos_token
    llama_tokenizer.pad_token = llama_tokenizer.eos_token

    model = model.to('cuda:0')
    llama_model = llama_model.to('cuda:1')

    with open(args.path_to_trap_info, "rb") as f:
        trap_info = pickle.load(f)
    trap_info["raw_traps"] = llama_tokenizer.batch_decode(trap_info.trap_tokens)

    non_members = {}

    for seq_len in [25, 50, 100]:
        small_path = args.non_member_path_prefix + str(seq_len)
        with open(small_path, "rb") as f:
            non_members[seq_len] = pickle.load(f)

    trap_df = trap_info

    # get the raw books
    data_w_traps = load_from_disk(args.path_to_dataset)

    # add the context
    all_context_text = []
    for i in tqdm(range(len(trap_df))):

        doc_trap_df = trap_df.iloc[i]
        doc_idx = int(doc_trap_df['book_idx'])
        trap_text = doc_trap_df['raw_traps']

        book_w_traps = data_w_traps[doc_idx]['text']

        context_text = find_context(doc_text=book_w_traps, trap=trap_text, tokenizer=llama_tokenizer,
                                    x_context_tokens=x_context_tokens)
        all_context_text.append(context_text)
    trap_df['context'] = all_context_text

    # add the non members
    assigned_non_members = []
    for i in tqdm(range(len(trap_df))):
        doc_trap_df = trap_df.iloc[i]
        seq_len, ppl = doc_trap_df['seq_len'], doc_trap_df['ppl_bucket']

        ppl_key = (ppl * 10 + 1, ppl * 10 + 11)
        selected_idx = random.sample(range(len(non_members[seq_len][ppl_key])), 1)[0]
        non_member_selected = non_members[seq_len][ppl_key][selected_idx]

        non_member_text = llama_tokenizer.decode(non_member_selected[1:])
        assigned_non_members.append(non_member_text)
    trap_df['non_member_trap_text'] = assigned_non_members

    # compute it all
    croissant_member_ppl, croissant_non_member_ppl, llama_member_ppl, llama_non_member_ppl = [], [], [], []
    for i in tqdm(range(len(trap_df))):

        doc_trap_df = trap_df.iloc[i]
        trap, context, non_member = doc_trap_df['raw_traps'], doc_trap_df['context'], doc_trap_df['non_member_trap_text']

        croissant_member_ppl.append(compute_ppl_in_context(model=model, device='cuda:0', tokenizer=tokenizer,
                                                           context_text=context, trap_text=trap))
        croissant_non_member_ppl.append(compute_ppl_in_context(model=model, device='cuda:0', tokenizer=tokenizer,
                                                               context_text=context, trap_text=non_member))
        llama_member_ppl.append(compute_ppl_in_context(model=llama_model, device='cuda:1', tokenizer=llama_tokenizer,
                                                       context_text=context, trap_text=trap))
        llama_non_member_ppl.append(compute_ppl_in_context(model=llama_model, device='cuda:1', tokenizer=llama_tokenizer,
                                                           context_text=context, trap_text=non_member))

    trap_df['croissant_member_ppl'] = croissant_member_ppl
    trap_df['croissant_non_member_ppl'] = croissant_non_member_ppl
    trap_df['llama_member_ppl'] = llama_member_ppl
    trap_df['llama_non_member_ppl'] = llama_non_member_ppl

    trap_df.to_csv(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-model", type=str, required=True, help="Path to LLaMA model")
    parser.add_argument("--path-to-tokenizer", type=str, required=True, help="Path to LLaMA tokenizer")
    parser.add_argument("--path-to-trap-info", type=str, required=True,
                        help="Path to trap info file (produced by inject_traps.py)")
    parser.add_argument("--path-to-dataset", type=str, required=True,
                        help="Path to dataset with injected traps")
    parser.add_argument("--non-members-path-prefix", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--X_tokens", type=int, required=True)

    args = parser.parse_args()

    main(args)
