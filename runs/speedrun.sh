#!/bin/bash
set -x

# BLITZ speedrun (< 1 min)
export OMP_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# CLEAN SLATE
rm -rf $NANOCHAT_BASE_DIR/tokenizer
rm -rf $NANOCHAT_BASE_DIR/base_checkpoints
rm -rf $NANOCHAT_BASE_DIR/chatsft_checkpoints
rm -rf $NANOCHAT_BASE_DIR/base_eval
rm -rf $NANOCHAT_BASE_DIR/chatsft_eval

source .venv/bin/activate

# Tokenizer
python -m nanochat.dataset -n 1
python -m scripts.tok_train --vocab-size 512 --max-chars 100000
python -m scripts.tok_eval

# Base model (pretraining)
python -m scripts.base_train --depth=2 --aspect-ratio=16 --max-seq-len=128 --window-pattern=L --total-batch-size=1024 --num-iterations=10 --device-batch-size=4 --eval-tokens=1024 --eval-every=10 --core-metric-every=-1 --sample-every=-1 --run=dummy

# evaluate the model
python -m scripts.base_eval --model-tag d2 --device-batch-size=4 --split-tokens=1024 --eval bpb,sample

# SFT
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run SFT
python -m scripts.chat_sft --model-tag d2 --max-seq-len=128 --total-batch-size=1024 --device-batch-size=4 --eval-tokens=1024 --eval-every=10 --chatcore-every=-1 --mmlu-epochs=1 --gsm8k-epochs=1 --run=dummy
python -m scripts.chat_eval -i sft --model-tag d2 --batch-size=4 --max-problems=1

# Generate report
python -m nanochat.report generate