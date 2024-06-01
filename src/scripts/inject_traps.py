import argparse
import pickle
import random
import logging
from itertools import cycle
import os

import numpy as np
import pandas as pd
from datasets import Dataset, load_from_disk
from tqdm import tqdm
from transformers import LlamaTokenizer


def inject_one(text: str, trap_text: str, n_rep: int) -> str:
    '''
    Let's inject the trap sequences at random places in the original text. 
    By splitting on spaces, we ensure to inject the trap sequences while not splitting any words from the original text.
    '''

    text_split_by_spaces = text.split(" ")
    all_indices_text = range(len(text_split_by_spaces))
    trap_indices = np.sort(random.sample(all_indices_text, n_rep))

    new_text = ''
    last_index = 0

    for idx in trap_indices:
        new_text += " ".join(text_split_by_spaces[last_index:idx])
        if idx == 0:
            new_text += trap_text
        else:
            new_text += " " + trap_text
        last_index = idx
    new_text += " ".join(text_split_by_spaces[last_index:])

    assert len(new_text) == len(text) + n_rep * len(trap_text)
    return new_text


def inject_all(df_trap_info, raw_dataset, tokenizer, args):

    trap_dataset_entries = []
    logging.info("Injecting traps...")

    for i, og_entry in tqdm(enumerate(raw_dataset)):
        new_entry = og_entry.copy()

        if i in df_trap_info.index:
            row = df_trap_info.loc[i]
            trap_tokens, n_rep = row["trap_tokens"], row["n_rep"]
            new_text = inject_one(
                text=og_entry["text"],
                trap_text=tokenizer.decode(trap_tokens),
                n_rep=n_rep
            )

            new_entry["text"] = new_text

        trap_dataset_entries.append(new_entry)

    # save the results
    ds_dict = {k: [e[k] for e in trap_dataset_entries] for k in raw_dataset.column_names}
    dataset = Dataset.from_dict(ds_dict)

    return dataset


def read_all_traps(path_to_trap_dir: str) -> dict[int, dict[tuple[int, int]], np.ndarray]:
    all_traps = {}  # seq_len -> {ppl_bucket: array}
    total_traps = 0

    for file in os.listdir(path_to_trap_dir):
        with open(os.path.join(path_to_trap_dir, file), "rb") as f:
            traps = pickle.load(f)
            seq_len = None
            for arr in traps.values():
                total_traps += len(arr)

                if seq_len is None:
                    seq_len = arr.shape[1] - 1  # exclude BOS
                elif seq_len != arr.shape[1] - 1:
                    raise ValueError(f"Inconsistent sequence length in {file}")
            all_traps[seq_len] = traps

    return all_traps, total_traps


def distribute_traps(all_traps, raw_dataset, args) -> pd.DataFrame:
    doc_indices = [i for i in range(len(raw_dataset)) if len(raw_dataset[i]["input_ids"] > args.doc_min_tokens)]
    random.shuffle(doc_indices)
    n_rep_iterator = cycle(args.n_reps)
    doc_idx_iterator = iter(doc_indices)
    records = []

    for seq_len in all_traps:
        for ppl_key in all_traps[seq_len]:
            ppl_bucket = ppl_key[0] // 10  # ppl_key looks like (11,21)
            for trap_tokens in all_traps[seq_len][ppl_key]:
                trap_tokens = trap_tokens[1:]  # remove BOS token
                n_rep = next(n_rep_iterator)
                doc_idx = next(doc_idx_iterator)

                records.append({
                    "doc_idx": doc_idx,
                    "doc_title": raw_dataset[doc_idx]["title"],
                    "seq_len": seq_len,
                    "ppl_bucket": ppl_bucket,
                    "n_rep": n_rep,
                    "trap_tokens": trap_tokens,
                })

    df = pd.DataFrame(records)
    df = df.set_index("doc_idx")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument("--path-to-tokenizer", type=str, required=True)
    parser.add_argument("--path-to-raw-dataset", required=True, type=str)
    parser.add_argument("--path-to-trap-dir", required=True, type=str)
    parser.add_argument("--output-ds-path", required=True, type=str)
    parser.add_argument("--output-info-path", required=True, type=str)
    parser.add_argument("--n-reps", nargs='+', type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--nb-workers", default=64, type=int)
    parser.add_argument("--doc-min-tokens", default=5000, type=int)

    args = parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(args.path_to_tokenizer)

    logging.info(f"Loading {args.path_to_raw_dataset}...")

    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset = load_from_disk(args.path_to_raw_dataset)
    dataset = dataset.map(
        lambda samples: tokenizer(samples["text"]),
        batched=False,
        num_proc=args.nb_workers,
    )
    logging.info(f"Loaded dataset with {len(dataset)} documents")

    all_traps, total_traps = read_all_traps(args.path_to_trap_dir)

    if total_traps > len(dataset):
        raise ValueError(f"Dataset is too small. len(dataset)={len(dataset)}, but found {total_traps} traps")

    if total_traps % len(args.n_reps) != 0:
        raise ValueError(f"total number of traps {total_traps} can't be equally split into {args.n_reps} groups")

    logging.info(f"Read dataset({len(dataset)} entries) and trap sequences ({total_traps} entries)")

    df_trap_info = distribute_traps(all_traps, dataset, args)
    with open(args.output_info_path, "wb") as f:
        # saving as pickle not csv because it's easier to deal with lists
        pickle.dump(df_trap_info, f)

    logging.info(f"Saved trap distribution info ({len(df_trap_info)} rows) to {args.output_info_path}")

    dataset = dataset.remove_columns(["input_ids", "attention_mask"])
    injected_dataset = inject_all(df_trap_info, dataset, tokenizer, args)

    # Save the dataset
    injected_dataset.save_to_disk(args.output_ds_path)

    logging.info(f"Saved trap-injected dataset ({len(injected_dataset)} documents) to {args.output_ds_path}")
