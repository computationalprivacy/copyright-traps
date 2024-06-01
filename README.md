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

## Injecting trap sequences

We inject trap sequences generated at the previous step by running the following:

```
python "src/scripts/inject_traps.py" --path-to-tokenizer "$LLAMA_TOKENIZER_PATH" --path-to-raw-dataset "$INPUT_DATASET_PATH" --path-to-trap-dir "data/traps/" --output-ds-path "data/injected/dataset_with_traps" --output-info-path "data/injected/trap_info.pkl" --n-reps 1 10 100 1000 --seed 1111
```

Input dataset should be a huggingface dataset that can be loaded with `load_from_disk()` method.
Dataset should contain at least one document per trap sequence provided - we inject one sequence to one document 
(while repeating the requested number of times). 

Folder specified in `--path-to-trap-dir` is expected to only contain the output of the previous step (potentially run 
multiple times with different parameters) - this script will iterate over and read all files in that folder.

This generates two outputs: the dataset itself and the metadata. The dataset contains the original data plus injected
sequences. Metadata is the dataframe (pickled, because it makes it easier to deal with lists) containing all the 
injection metadata: which trap was injected into which document and how many times.

Important arguments:
* `--n-reps`: a list of integers defining number of repetitions when injecting a sequence. Number of elements in 
this list must be so that the total number of traps is divisible by it - we want each number of repetitions
to have the same number of traps.
* `--doc-min-tokens`: we would only inject into documents with at least this number of tokens


Please refer to `src/scripts/run_all.sh` for the full pipeline.

## Data analysis

Note that at the moment we're not yet releasing the exact trap sequences we've used for training, so data analysis 
code is provided for illustrative purposes only.

Code to generate out figures and tables is located in `notebooks/` folder, with one exceptions - script for Table 3
which lives is `src/scripts`