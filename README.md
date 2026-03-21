# nanochat

![nanochat logo](dev/nanochat.png)
![scaling laws](dev/scaling_laws_jan26.png)

This fork keeps the **nanochat training/eval codebase** from upstream but uses **Ollama as the default chat backend** (no local transformer weights required for the web UI). You get a ChatGPT-style UI that talks to any Ollama model, with optional RAG (Ollama embeddings + text corpus or Chroma).

**This fork:** [kwizzlesurp10-ctrl/nanochatkwizzle](https://github.com/kwizzlesurp10-ctrl/nanochatkwizzle). Upstream: [karpathy/nanochat](https://github.com/karpathy/nanochat). Community: [DeepWiki](https://deepwiki.com/karpathy/nanochat), [Discussions](https://github.com/karpathy/nanochat/discussions), [#nanochat on Discord](https://discord.com/channels/1020383067459821711/1427295580895314031).

## Getting started

### Chat with Ollama (default)

1. Install [Ollama](https://ollama.com/) and start it (`ollama serve`).
2. Pull a chat model and (if using RAG) an embedding model:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

3. Install deps and run the UI:

```bash
uv sync --extra gpu   # or --extra cpu on machines without CUDA; Ollama chat does not need a GPU in this process
source .venv/bin/activate
python -m scripts.chat_web
```

Optional: `OLLAMA_CHAT_MODEL=mistral python -m scripts.chat_web` or `--ollama-chat-model mistral`.

**RAG + Ollama in one command** (uses sample corpus if you do not pass `--rag-corpus`):

```bash
python chat_ollama_rag.py
```

Or manually: `python -m scripts.chat_web --rag --rag-corpus path/to.txt` (see below for Chroma).

### Local nanochat checkpoints (optional)

To serve a model you trained with this repo (SFT/RL checkpoints under `~/.cache/nanochat`):

```bash
python -m scripts.chat_web --backend nanochat --model-tag nanobot --source sft
```

### Nanobot / speedrun (optional GPU training)

Pipeline: **Pretrain (base) → SFT (chat) → serve**. [runs/nanobot.sh](runs/nanobot.sh) runs pretrain + SFT; when it finishes you get SFT checkpoints you can serve. Checkpoints use model-tag `nanobot` (separate from speedrun). Same hardware and rough duration as speedrun (~3 hr on 8×H100).

```bash
bash runs/nanobot.sh
```

Serve the nanobot SFT checkpoint:

```bash
python -m scripts.chat_web --backend nanochat --model-tag nanobot --source sft
```

**If the model outputs gibberish or special tokens** (e.g. `<|python_start|>`): you are likely serving a pretrain-only checkpoint. Use an SFT checkpoint (e.g. from a completed `runs/speedrun.sh` or `runs/nanobot.sh`) and pass the correct `--model-tag` and `--source sft`. Generation now stops when the model emits tool/code tokens so less garbage is streamed.

**CUDA OOM on a small GPU (e.g. 6 GB) while using `--backend nanochat`:** Only one heavy CUDA workload should use the GPU at a time — stop `base_train` / `chat_sft` / other training shells (`nvidia-smi` shows PIDs) before serving chat. Cursor and the browser are fine; a training run alone can take 2–3 GB+ and leave no room for inference. Optional: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation.

**Flash Attention 3 vs PyTorch SDPA:** FA3 only runs on **Hopper (sm90+)** with **bf16**. Other GPUs (e.g. RTX 16xx–40xx, most consumer cards) correctly use **SDPA** — that is not an error. Sliding-window patterns (`S` in `--window-pattern`) are **automatically coerced to `L`** on the SDPA path so training stays fast; on Hopper you can keep patterns like `SSSL` for FA3.

**RAG (`--rag`)** uses Ollama embeddings (`ollama pull nomic-embed-text`). You can use a plain-text corpus (`--rag-corpus FILE`) or a Chroma persist directory (`--rag-db`).

**Why `uv add chromadb` fails on Python 3.10:** current `chromadb` pulls `onnxruntime`, which only ships wheels for **cp311+**, not cp310. Workarounds:

1. **Stay on 3.10 — no Chroma:** run with `--rag --rag-corpus /path/to.txt` (or rely on `dev/rag_sample_corpus.txt` for a quick test). No `chromadb` install needed.
2. **Use Chroma:** move the project venv to **Python 3.11+** (e.g. `uv python install 3.12`, recreate `.venv`, `uv sync --extra gpu`), then `uv add chromadb` and point `--rag-db` at your Chroma folder.

### Weights & Biases (entity / project)

Training scripts (`base_train`, `chat_sft`, `chat_rl`) log to wandb with default projects `nanochat`, `nanochat-sft`, and `nanochat-rl`. Send runs to a team or custom project with env vars or flags:

```bash
export WANDB_ENTITY=kwizzlesurp10-sevtech   # optional; else your login default
export WANDB_PROJECT=nanochat-sft            # optional for SFT; overrides default project name
torchrun ... -m scripts.chat_sft -- --run=my-run
# or per-invocation:
torchrun ... -m scripts.chat_sft -- --run=my-run --wandb-entity=kwizzlesurp10-sevtech --wandb-project=nanochat-sft
```

Patch an existing run’s config via the public API (`run_id` is the short id in the run URL). Requires `wandb login` (or `WANDB_API_KEY`) with access to that entity/project.

**Generic path** — swap entity / project / run id (no code change besides those strings):

```python
import wandb

api = wandb.Api()
entity, project, run_id = "kwizzlesurp10-sevtech", "nanochat", "YOUR_RUN_ID"
run = api.run(f"{entity}/{project}/{run_id}")
run.config["architecture"] = "8L-512D-8H"
run.config["max_seq_len"] = 256
run.update()
```

**Example** (run `jvwxyzr3` in project `nanochat`):

```python
import wandb

api = wandb.Api()
run = api.run("kwizzlesurp10-sevtech/nanochat/jvwxyzr3")
run.config["architecture"] = "8L-512D-8H"
run.config["max_seq_len"] = 256
run.update()
```

SFT/RL runs live under `nanochat-sft` / `nanochat-rl` instead of `nanochat` — use the same pattern with the correct project name from the URL.

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

A few more notes:

- The code will run just fine on the Ampere 8XA100 GPU node as well, but a bit slower.
- All code will run just fine on even a single GPU by omitting `torchrun`, and will produce ~identical results (code will automatically switch to gradient accumulation), but you'll have to wait 8 times longer.
- If your GPU(s) have less than 80GB, you'll have to tune some of the hyperparameters or you will OOM / run out of VRAM. Look for `--device_batch_size` in the scripts and reduce it until things fit. E.g. from 32 (default) to 16, 8, 4, 2, or even 1. Less than that you'll have to know a bit more what you're doing and get more creative.
- Most of the code is fairly vanilla PyTorch so it should run on anything that supports that - xpu, mps, or etc, but I haven't personally exercised all of these code paths so there might be sharp edges.

## Research

If you are a researcher and wish to help improve nanochat, two scripts of interest are [runs/scaling_laws.sh](runs/scaling_laws.sh) and [runs/miniseries.sh](runs/miniseries.sh). See [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) for related documentation. For quick experimentation (~5 min pretraining runs) my favorite scale is to train a 12-layer model (GPT-1 sized), e.g. like this:

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \
```

This uses wandb (run name "d12"), only runs the CORE metric on last step, and it doesn't sample and save intermediate checkpoints. I like to change something in the code, re-run a d12 (or a d16 etc) and see if it helped, in an iteration loop. To see if a run helps, I like to monitor the wandb plots for:

1. `val_bpb` (validation loss in vocab-size-invariant units of bits per byte) as a function of `step`, `total_training_time` and `total_training_flops`.
2. `core_metric` (the DCLM CORE socre)
3. VRAM utilization, `train/mfu` (Model FLOPS utilization), `train/tok_per_sec` (training throughput)

See an example [here](https://github.com/karpathy/nanochat/pull/498#issuecomment-3850720044).

The important thing to note is that nanochat is written and configured around one single dial of complexity - the depth of the transformer. This single integer automatically determines all other hyperparameters (the width of the transformer, number of heads, learning rate adjustments, training horizons, weight decays, ...) so that the trained model comes out compute optimal. The idea is that the user doesn't have to think about or set any of this, they are simply asking for a smaller or bigger model using `--depth`, and everything "just works". By sweeping out the depth, you achieve the nanochat miniseries of compute optimal models at various sizes. Strong reference-scale models in upstream benchmarks often sit around d24–d26 with the current code. Any candidate changes have to be principled enough that they work for all settings of depth.

## Running on CPU / MPS

The script [runs/runcpu.sh](runs/runcpu.sh) shows a very simple example of running on CPU or Apple Silicon. It dramatically shrinks the LLM that is being trained to make things fit into a reasonable time interval of a few ten minutes of training. You will not get strong results in this way.

## Precision / dtype

nanochat does not use `torch.amp.autocast`. Instead, precision is managed explicitly through a single global `COMPUTE_DTYPE` (defined in `nanochat/common.py`). By default this is auto-detected based on your hardware:

| Hardware | Default dtype | Why |
|----------|--------------|-----|
| CUDA SM 80+ (A100, H100, ...) | `bfloat16` | Native bf16 tensor cores |
| CUDA SM < 80 (V100, T4, ...) | `float32` | No bf16; fp16 available via `NANOCHAT_DTYPE=float16` (uses GradScaler) |
| CPU / MPS | `float32` | No reduced-precision tensor cores |

You can override the default with the `NANOCHAT_DTYPE` environment variable:

```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"   # force fp32
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=8 -m scripts.base_train  # force bf16
```

How it works: model weights are stored in fp32 (for optimizer precision), but our custom `Linear` layer casts them to `COMPUTE_DTYPE` during the forward pass. Embeddings are stored directly in `COMPUTE_DTYPE` to save memory. This gives us the same mixed-precision benefit as autocast but with full explicit control over what runs in which precision.

Note: `float16` training automatically enables a `GradScaler` in `base_train.py` to prevent gradient underflow. SFT suppors this too but RL currently does not. Inference in fp16 works fine everywhere.

## Guides

I've published a number of guides that might contain helpful information, most recent to least recent:

- [Feb 1 2026: Beating GPT-2 for <<$100: the nanochat journey](https://github.com/karpathy/nanochat/discussions/481)
- [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) documents the first nanochat miniseries of models.
- To add new abilities to nanochat, see [Guide: counting r in strawberry (and how to add abilities generally)](https://github.com/karpathy/nanochat/discussions/164).
- To customize your nanochat, see [Guide: infusing identity to your nanochat](https://github.com/karpathy/nanochat/discussions/139) in Discussions, which describes how you can tune your nanochat's personality through synthetic data generation and mixing that data into the SFT stage.
- [Oct 13 2025: original nanochat post](https://github.com/karpathy/nanochat/discussions/1) introducing nanochat, though now it contains some deprecated information and the model is a lot older (with worse results) than current master.

## File structure

```
.
├── LICENSE
├── README.md
├── dev
│   ├── gen_synthetic_data.py       # Example synthetic data for identity
│   ├── generate_logo.html
│   ├── nanochat.png
│   └── repackage_data_reference.py # Pretraining data shard generation
├── nanochat
│   ├── __init__.py                 # empty
│   ├── checkpoint_manager.py       # Save/Load model checkpoints
│   ├── common.py                   # Misc small utilities, quality of life
│   ├── core_eval.py                # Evaluates base model CORE score (DCLM paper)
│   ├── dataloader.py               # Tokenizing Distributed Data Loader
│   ├── dataset.py                  # Download/read utils for pretraining data
│   ├── engine.py                   # Efficient model inference with KV Cache
│   ├── execution.py                # Allows the LLM to execute Python code as tool
│   ├── gpt.py                      # The GPT nn.Module Transformer
│   ├── logo.svg
│   ├── loss_eval.py                # Evaluate bits per byte (instead of loss)
│   ├── optim.py                    # AdamW + Muon optimizer, 1GPU and distributed
│   ├── report.py                   # Utilities for writing the nanochat Report
│   ├── tokenizer.py                # BPE Tokenizer wrapper in style of GPT-4
│   └── ui.html                     # HTML/CSS/JS for nanochat frontend
├── pyproject.toml
├── runs
│   ├── miniseries.sh               # Miniseries training script
│   ├── nanobot.sh                  # Pretrain + SFT (model-tag nanobot), then serve
│   ├── runcpu.sh                   # Small example of how to run on CPU/MPS
│   ├── scaling_laws.sh             # Scaling laws experiments
│   └── speedrun.sh                 # Train the ~$100 nanochat d20
├── scripts
│   ├── base_eval.py                # Base model: CORE score, bits per byte, samples
│   ├── base_train.py               # Base model: train
│   ├── chat_cli.py                 # Chat model: talk to over CLI
│   ├── chat_eval.py                # Chat model: eval tasks
│   ├── chat_rl.py                  # Chat model: reinforcement learning
│   ├── chat_sft.py                 # Chat model: train SFT
│   ├── chat_web.py                 # Chat model: talk to over WebUI
│   ├── tok_eval.py                 # Tokenizer: evaluate compression rate
│   └── tok_train.py                # Tokenizer: train it
├── tasks
│   ├── arc.py                      # Multiple choice science questions
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── customjson.py               # Make Task from arbitrary jsonl convos
│   ├── gsm8k.py                    # 8K Grade School Math questions
│   ├── humaneval.py                # Misnomer; Simple Python coding task
│   ├── mmlu.py                     # Multiple choice questions, broad topics
│   ├── smoltalk.py                 # Conglomerate dataset of SmolTalk from HF
│   └── spellingbee.py              # Task teaching model to spell/count letters
├── tests
│   └── test_engine.py
└── uv.lock
```

## Contributing

The goal of nanochat (upstream) is to improve the state of the art in micro models that are accessible to work with end to end on modest budgets. This fork adds an Ollama-first chat path so you can use that stack without training; the training code remains available for experiments. Upstream history and CORE / leaderboard context live in [dev/LEADERBOARD.md](dev/LEADERBOARD.md).

Current AI policy: disclosure. When submitting a PR, please declare any parts that had substantial LLM contribution and that you have not written or that you do not fully understand.

## Acknowledgements

- The name (nanochat) derives from my earlier project [nanoGPT](https://github.com/karpathy/nanoGPT), which only covered pretraining.
- nanochat is also inspired by [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt), which gamified the nanoGPT repo with clear metrics and a leaderboard, and borrows a lot of its ideas and some implementation for pretraining.
- Thank you to [HuggingFace](https://huggingface.co/) for fineweb and smoltalk.
- Thank you [Lambda](https://lambda.ai/service/gpu-cloud) for the compute used in developing this project.
- Thank you to chief LLM whisperer 🧙‍♂️ Alec Radford for advice/guidance.
- Thank you to the repo czar Sofie [@svlandeg](https://github.com/svlandeg) for help with managing issues, pull requests and discussions of nanochat.

## Cite

If you find nanochat helpful in your research cite simply as:

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/kwizzlesurp10-ctrl/nanochatkwizzle}
}
```

## License

MIT
