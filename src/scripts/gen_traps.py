import argparse
import pickle
import random
import logging
from typing import Sequence, Dict, Iterable, Any

import numpy as np
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils import compute_perplexity, jaccard_similarity


def is_duplicate(seq: Sequence, traps: Dict[Any, Iterable[Iterable]], threshold=0.2):
    for key in traps:
        for trap in traps[key]:
            if jaccard_similarity(seq[1:], trap[1:]) > threshold:
                return True

    return False


def gen_single_unit_buckets(model: LlamaForCausalLM, tokenizer: LlamaTokenizer, args):
    ppl_range = args.max_perplexity - args.min_perplexity + 1
    traps = {i: [] for i in range(args.min_perplexity, args.max_perplexity + 1)}

    input = tokenizer([""] * args.batch_size, return_tensors="pt").to(args.device)
    samples_per_unit = args.num_traps // ppl_range

    total_samples = 0
    step = 0
    duplicates = 0
    bucket_full = 0
    too_high_ppl = 0

    while sum([len(x) for x in traps.values()]) < args.num_traps:
        if step > 0 and step % 5 == 0:
            samples = sum([len(x) for x in traps.values()])
            logging.info(
                f"Step: {step} | total: {total_samples} | accepted: {samples} | duplicates: {duplicates} | bucket full: {bucket_full} | too_high_ppl: {too_high_ppl}"
            )
            open_buckets = [key for key, val in traps.items() if len(val) < samples_per_unit]
            logging.info(f"Open buckets: {sorted(open_buckets)}")

        for temp in np.arange(args.temp_min, args.temp_max, args.temp_step):
            try:
                generated_ids = model.generate(
                    input["input_ids"],
                    max_length=args.seq_len + 1,
                    do_sample=True,
                    temperature=temp,
                )

                if args.retokenize:
                    # retokenize to avoid special tokens
                    generated_raw = tokenizer.batch_decode(generated_ids[:, 1:])
                    retokenized = tokenizer.batch_encode_plus(
                        generated_raw,
                        return_tensors="pt",
                        truncation="longest_first",
                        padding="max_length",
                        max_length=args.seq_len + 1,  # accounting for BOS token
                    ).to(args.device)
                    generated_ids = generated_ids[(retokenized.input_ids == generated_ids).all(dim=1), :]

                batch_ppl = compute_perplexity(
                    model, generated_ids, torch.ones_like(generated_ids).to(args.device)
                )  # (batch_size, seq_len)

                for idx, ppl in enumerate(batch_ppl):
                    total_samples += 1
                    int_ppl = int(ppl)
                    if int_ppl > args.max_perplexity or int_ppl < args.min_perplexity:
                        too_high_ppl += 1
                        continue

                    if len(traps[int_ppl]) >= samples_per_unit:
                        bucket_full += 1
                        continue

                    new_trap = generated_ids[idx].detach().cpu().numpy()
                    if is_duplicate(new_trap, traps, threshold=args.jaccard_threshold):
                        duplicates += 1
                        continue

                    traps[int_ppl].append(new_trap)

            except (RuntimeError, ValueError):
                # https://github.com/facebookresearch/llama/issues/380
                pass

        step += 1

    return traps


def bucketize(traps, args):
    ret = {}
    buckets = np.array_split(list(range(args.min_perplexity, args.max_perplexity + 1)), args.num_buckets)
    assert len(set([len(x) for x in buckets])) == 1  # all buckets are equal-sized

    for bucket in buckets:
        low = bucket[0]
        high = bucket[-1] + 1  # int(1.99) == 1, so the actual range is (int(low), int(high)+1)
        arr = np.vstack([x for y in bucket for x in traps[y]])
        ret[(low, high)] = arr

    return ret


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) 

    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-model", type=str, required=True, help="Path to LLaMA model")
    parser.add_argument("--path-to-tokenizer", type=str, required=True, help="Path to LLaMA tokenizer")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output file (pickle)")

    parser.add_argument(
        "--seq-len",
        type=int,
        required=True,
        help="Target trap sequence length in tokens (doesn't include BOS token)",
    )
    parser.add_argument("-n", "--num-traps", type=int, required=True, help="Number of trap sequences to be generated")

    parser.add_argument(
        "--max-perplexity",
        type=int,
        default=100,
        help="""
        Trap sequence perplexity (coverted to int) will be uniformly distributed in the range[1, max_perplexity]. 
        Perplexity is converted to integer before comparing to max_perplexity, therefore the actual upper bound is max_perplexity+1.
        """,
    )
    parser.add_argument("--min-perplexity", type=int, default=1)
    parser.add_argument("--num-buckets", type=int, default=10, help="Number of buckets in the output file")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--temp-min", type=float, default=0.5, help="Min value for LLM temperature generation cycle")
    parser.add_argument("--temp-max", type=float, default=4, help="Max value for LLM temperature generation cycle")
    parser.add_argument("--temp-step", type=float, default=0.5, help="Step value for LLM temperature generation cycle")
    parser.add_argument("--jaccard-threshold", type=float, default=0.2,
                        help="Jaccard distance threshold for deduplication")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--retokenize", action="store_true",
                        help="If enabled, filter out sequences which change their length in tokens when retokenized")

    args = parser.parse_args()

    device = torch.device(args.device)
    logging.info(f"Using device:{device}")

    random.seed(args.seed * args.seq_len)
    np.random.seed(args.seed * args.seq_len)
    torch.manual_seed(args.seed * args.seq_len)

    # we ensure uniformity within a bucket by grouping perplexities by their integer values and sampling uniformly
    if (args.max_perplexity - args.min_perplexity + 1) % args.num_buckets != 0:
        raise ValueError("Integer perplexity range must be divisible by the number of buckets")

    # We want to ensure full uniformity across buckets, so that every bucket has the same number of traps
    if args.num_traps % (args.max_perplexity - args.min_perplexity + 1) != 0:
        raise ValueError("Target number of traps must be divisible by max_perplexity")

    model = LlamaForCausalLM.from_pretrained(args.path_to_model, torch_dtype=torch.float16).to(device)  # type: ignore
    tokenizer = LlamaTokenizer.from_pretrained(args.path_to_tokenizer, torch_dtype=torch.float16)
    tokenizer.pad_token = tokenizer.eos_token

    traps = gen_single_unit_buckets(model, tokenizer, args)
    buckets = bucketize(traps, args)

    with open(args.output, "wb") as file:
        pickle.dump(buckets, file)

    logging.info(f"Finished. Generated {sum([x.shape[0] for x in buckets.values()])} samples")
