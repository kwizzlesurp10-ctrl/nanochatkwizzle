#!/bin/bash

# Custom run with safe hyperparameters for limited VRAM (GTX 1660 SUPER)
# depth=8, hidden_size=512, num_heads=8, max_seq_len=256, total_batch_size=32768

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
source .venv/bin/activate

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=custom_small
fi

# Pretraining
python -m scripts.base_train \
    --depth=8 \
    --aspect-ratio=64 \
    --head-dim=64 \
    --dropout=0.1 \
    --learning-rate=4e-4 \
    --warmup-steps=2000 \
    --device-batch-size=1 \
    --total-batch-size=32768 \
    --max-seq-len=256 \
    --epochs=1 \
    --weight-decay=0.1 \
    --optimizer=muon \
    --eval-tokens=1048576 \
    --save-every=100 \
    --run=$WANDB_RUN

sleep 5

# SFT
python -m scripts.chat_sft \
    --model-tag=d8 \
    --learning-rate=1e-4 \
    --device-batch-size=1 \
    --total-batch-size=32768 \
    --max-seq-len=256 \
    --run=$WANDB_RUN

sleep 5

# RL
python -m scripts.chat_rl \
    --model-tag=d8 \
    --learning-rate=5e-5 \
    --device-batch-size=1 \
    --run=$WANDB_RUN