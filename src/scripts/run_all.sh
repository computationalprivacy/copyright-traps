#!/bin/sh

set -e
set -u
set -o pipefail

# To be filled in
PROJECT_ROOT=""
LLAMA_TOKENIZER_PATH=""
LLAMA_MODEL_PATH=""
INPUT_DATASET_PATH=""

DATA_DIR="$PROJECT_ROOT/data"
SRC_DIR="$PROJECT_ROOT/src"
export PYTHONPATH=$PYTHONPATH:"$SRC_DIR"

# Generate traps with 25, 50 and 100 tokens
mkdir -p "$DATA_DIR/traps"
rm -f "$DATA_DIR/traps/*" # clean up previous output

python "$SRC_DIR/scripts/gen_traps.py" --path-to-model "$LLAMA_MODEL_PATH" --path-to-tokenizer "$LLAMA_TOKENIZER_PATH" --seq-len 25 --seed 123456 -n 2000 -o "$DATA_DIR/traps/traps_len25_n2000.pkl"
python "$SRC_DIR/scripts/gen_traps.py" --path-to-model "$LLAMA_MODEL_PATH" --path-to-tokenizer "$LLAMA_TOKENIZER_PATH" --seq-len 50 --seed 123457 -n 2000 -o "$DATA_DIR/traps/traps_len50_n2000.pkl"
python "$SRC_DIR/scripts/gen_traps.py" --path-to-model "$LLAMA_MODEL_PATH" --path-to-tokenizer "$LLAMA_TOKENIZER_PATH" --seq-len 100 --seed 123458 -n 2000 -o "$DATA_DIR/traps/traps_len100_n2000.pkl" --temp-max 8 --temp-step 0.25

# Inject traps and write output to data dir
mkdir -p "$DATA_DIR/injected"
python "$SRC_DIR/scripts/inject_traps.py" --path-to-tokenizer "$LLAMA_TOKENIZER_PATH" --path-to-raw-dataset "$INPUT_DATASET_PATH" --path-to-trap-dir "$DATA_DIR/traps/" --output-ds-path "$DATA_DIR/injected/dataset_with_traps" --output-info-path "$DATA_DIR/injected/trap_info.pkl" --n-reps 1 10 100 1000 --seed 1111