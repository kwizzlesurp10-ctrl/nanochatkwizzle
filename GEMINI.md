# GEMINI.md - Project Context: nanochat

## Project Overview
`nanochat` is a minimal, full-stack ChatGPT clone designed for efficiency and accessibility. This project is a fork of [karpathy/nanochat](https://github.com/karpathy/nanochat), maintaining the core training and evaluation codebase while prioritizing **Ollama** as the default chat backend. It enables users to run a ChatGPT-style UI that interacts with any Ollama model, supports RAG (Retrieval-Augmented Generation), and provides a complete pipeline for training local models (Pretrain → SFT → RL).

The project is optimized for both high-end hardware (8xH100/A100) and smaller setups, including CPU/MPS, through heuristic-based configuration downshifting.

## Main Technologies
- **Language:** Python 3.10+
- **Deep Learning:** PyTorch (with Flash Attention 3 support)
- **Dependency Management:** `uv`
- **Web Framework:** FastAPI + Uvicorn
- **Chat Backend:** Ollama (default) or local NanoChat checkpoints
- **Experiment Tracking:** Weights & Biases (wandb)
- **Data:** Hugging Face `datasets` (FineWeb, SmolTalk)
- **Vector DB:** Chroma (optional, for RAG)

## Key Architecture & Features
- **Transformer Design:** Modern GPT implementation featuring:
    - Rotary Embeddings (RoPE)
    - RMSNorm (no learnable parameters)
    - QK Norm (to stabilize training)
    - Group-Query Attention (GQA)
    - Flash Attention 3 (on Hopper+ GPUs, with SDPA fallback)
    - Causal Sliding Window Attention
    - ReLU² activation in MLP
- **Precision Management:** Explicit `COMPUTE_DTYPE` (bf16/fp32) managed in `nanochat/common.py`. Model weights are stored in fp32, but matmuls run in `COMPUTE_DTYPE`.
- **Unified Web UI:** A single FastAPI instance serves both the chat interface and the streaming API.
- **RAG Support:** Integrates Ollama embeddings with either a plain-text corpus or a Chroma database.

## Building and Running

### Environment Setup
```bash
# Install dependencies with uv
uv sync --extra gpu   # Use --extra cpu for machines without CUDA
source .venv/bin/activate
```

### Running the Chat UI
```bash
# Default: Chat with Ollama (requires 'ollama serve')
python -m scripts.chat_web

# Chat with local nanochat SFT checkpoints
python -m scripts.chat_web --backend nanochat --model-tag nanobot --source sft

# Enable RAG with a text corpus
python -m scripts.chat_web --rag --rag-corpus path/to/corpus.txt
```

### Training Pipeline
The project provides several pre-configured scripts in `runs/`:
- `bash runs/nanobot.sh`: Full pipeline (Pretrain + SFT) with automatic hardware-aware scaling.
- `bash runs/speedrun.sh`: Train a ~$100 nanochat model (d20).
- `bash runs/runcpu.sh`: Small example for CPU/MPS execution.

Individual scripts for specific stages:
- **Pretraining:** `torchrun ... -m scripts.base_train`
- **SFT (Supervised Fine-Tuning):** `torchrun ... -m scripts.chat_sft`
- **RL (Reinforcement Learning):** `torchrun ... -m scripts.chat_rl`

## Development Conventions
- **Model Configuration:** Centered around the `--depth` parameter, which automatically determines other hyperparameters (width, heads, learning rate, etc.) to maintain compute optimality.
- **DType Control:** Use the `NANOCHAT_DTYPE` environment variable to override auto-detected compute precision.
- **Logging:** Uses `wandb` for training metrics. Default projects are `nanochat`, `nanochat-sft`, and `nanochat-rl`.
- **Checkpoints:** Local checkpoints are stored in `~/.cache/nanochat/` by default. This can be overridden via `NANOCHAT_BASE_DIR`.
- **Code Style:** Vanilla PyTorch with explicit control over precision and distributed training logic.

## Directory Structure Highlights
- `nanochat/`: Core library (model architecture, engine, dataloader).
- `scripts/`: CLI entry points for training, evaluation, and serving.
- `tasks/`: Evaluation benchmarks (GSM8K, MMLU, ARC, etc.).
- `runs/`: Shell scripts for end-to-end training runs.
- `dev/`: Synthetic data generation and utility scripts.
