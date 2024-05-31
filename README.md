# Copyright Traps for Large Language Models

## Generating trap sequences

We generate trap sequences to be injected with the following script:

```
python src/scripts/gen_traps.py --path-to-model $LLAMA_MODEL --path-to-tokenizer $LLAMA_TOKENIZER -o data/traps.pkl --seq-len 25 -n 500
```

This example generates 500 sequences of 25 tokens (26 including BOS tokens), with perplexity uniformly distributed 
between [1,101). This means that the output file will contain 5 sequences with `k <= perplexity < k+1` 
for any `k` between 1 and 100.
Note that in the paper we split sequences into perplexity buckets of size 10 (i.e. [1,11), [11,21), etc).

Important arguments:
* `--min-perplexity` and `--max-perplexity` define perplexity range
* `--num-buckets` define number of buckets in the output file
* `--temp-min`, `--temp-max`, `--temp-step` configure temperature settings of the LLM when generating sequences. We iterate over a range of temperature values to cover the desired perplexity range (default temperature would rarely produce sequences with very low or very high perplexity)
* `--jaccard-threshold` ensures deduplication between generated sequences. We want to ensure there's no cross-memorization between different trap-sequences, so we ensure jaccard distance between any two sequences is above a certain threshold. This is increasingly important for low perplexity sequences.
* `--retokenize` eliminates tokenization artifacts and ensures that generated sequences maintain target length after once cycle of decoding to raw text, and then encoding back. When enabled, we only consider sequences which maintain the same length after retokenization.
