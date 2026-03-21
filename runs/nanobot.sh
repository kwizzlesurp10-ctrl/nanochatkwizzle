#!/bin/bash
set -euo pipefail

# Nanobot: same pipeline as speedrun — Pretrain (base) → SFT (chat) → serve.
# This script runs pretrain + SFT; when it finishes you get SFT checkpoints you can serve.
# Checkpoints go under model-tag "nanobot" (base_checkpoints/nanobot, chatsft_checkpoints/nanobot).
# Designed to run on an 8XH100 GPU node; ~3 hours.
# If you're on smaller hardware, this script automatically downshifts to a tiny config that fits.

# Example: bash runs/nanobot.sh
# With wandb: WANDB_RUN=nanobot bash runs/nanobot.sh
# Optional: WANDB_ENTITY=your-team WANDB_PROJECT=nanochat-sft (see README)

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

if [ -z "${WANDB_RUN:-}" ]; then
    WANDB_RUN=nanobot
fi

python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Hardware-aware defaults (8xH100 => speedrun-like; otherwise downshift)

GPU_COUNT="$(
python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"

GPU_MEM_MIB="0"
if command -v nvidia-smi &> /dev/null; then
  GPU_MEM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')"
fi

# Defaults for "real" nanobot (matches speedrun pretrain recipe)
: "${NPROC_PER_NODE:=8}"
: "${DEPTH:=24}"
: "${DEVICE_BATCH_SIZE:=16}"
: "${MAX_SEQ_LEN:=2048}"
BASE_EXTRA_ARGS=(
  "--target-param-data-ratio=8"
  "--fp8"
  "--window-pattern=SSSL"
  "--max-seq-len=$MAX_SEQ_LEN"
)
SFT_EXTRA_ARGS=("--max-seq-len=$MAX_SEQ_LEN")
BASE_EVAL_EXTRA_ARGS=()

# Heuristic: if we're not on >=8 GPUs or if VRAM is small, run a tiny single-GPU config.
if [ "$GPU_COUNT" -lt 8 ] || [ "${GPU_MEM_MIB:-0}" -lt 40000 ]; then
  NPROC_PER_NODE=1
  DEPTH=4
  DEVICE_BATCH_SIZE=1
  MAX_SEQ_LEN=512
  BASE_EXTRA_ARGS=(
    "--window-pattern=L"
    "--max-seq-len=$MAX_SEQ_LEN"
    "--total-batch-size=512"
    "--num-iterations=20"
    "--eval-tokens=512"
    "--core-metric-every=-1"
  )
  SFT_EXTRA_ARGS=(
    "--max-seq-len=$MAX_SEQ_LEN"
    "--total-batch-size=512"
    "--num-iterations=50"
    "--eval-tokens=512"
  )
  BASE_EVAL_EXTRA_ARGS=(
    "--eval=bpb"
    "--split-tokens=512"
  )
  echo "Detected small/limited GPU setup (gpus=$GPU_COUNT, vram_mib=${GPU_MEM_MIB:-unknown})."
  echo "Downshifting nanobot to: depth=$DEPTH, max_seq_len=$MAX_SEQ_LEN, nproc_per_node=$NPROC_PER_NODE, device_batch_size=$DEVICE_BATCH_SIZE, window_pattern=L"
fi

# Tokenizer
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train
python -m scripts.tok_eval

# Base model (pretraining)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
  --depth="$DEPTH" --device-batch-size="$DEVICE_BATCH_SIZE" \
  --run="$WANDB_RUN" --model-tag=nanobot \
  "${BASE_EXTRA_ARGS[@]}"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
  --device-batch-size="$DEVICE_BATCH_SIZE" --model-tag=nanobot \
  "${BASE_EVAL_EXTRA_ARGS[@]}"

# SFT (chat)
curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft -- \
  --device-batch-size="$DEVICE_BATCH_SIZE" --run="$WANDB_RUN" --model-tag=nanobot \
  "${SFT_EXTRA_ARGS[@]}"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_eval -- -i sft --model-tag=nanobot

# Serve: python -m scripts.chat_web  (use nanobot checkpoint via env or default latest)

python -m nanochat.report generate
