import os
import subprocess
import sys
import torch
import wandb
import argparse
from nanochat.common import estimate_model_vram, autodetect_device_type, COMPUTE_DTYPE

def validate_config(config):
    """Check if the configuration is likely to fit in VRAM."""
    device_type = autodetect_device_type()
    if device_type != "cuda":
        return True, "" # CPU/MPS estimation not implemented/required
    
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    # Extract params from config, with fallbacks to base_train.py defaults
    depth = int(config.get("depth", 8))
    aspect_ratio = int(config.get("aspect_ratio", 64))
    head_dim = int(config.get("head_dim", 64))
    max_seq_len = int(config.get("max_seq_len", 1024))
    device_batch_size = int(config.get("device_batch_size", 128))
    vocab_size = 128320 # approx for nanochat tokenizer
    
    model_dim = ((depth * aspect_ratio + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    
    est_vram = estimate_model_vram(
        depth, model_dim, num_heads, vocab_size, max_seq_len, device_batch_size,
        dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32,
        training=True
    )
    
    # print(f"DEBUG PRE-FLIGHT: depth={depth}, model_dim={model_dim}, num_heads={num_heads}, vocab_size={vocab_size}, max_seq_len={max_seq_len}, device_batch_size={device_batch_size}")
    # print(f"DEBUG PRE-FLIGHT: est_vram={est_vram:.2f}GB")

    if est_vram > total_vram * 0.95: # 5% headroom (tight for 1660 SUPER)
        return False, f"Estimated VRAM {est_vram:.2f}GB exceeds available {total_vram:.2f}GB"
    
    # Constraint check: num_iterations and resume_from_step should be non-negative (except -1 for disabled)
    if int(config.get("num_iterations", 0)) < -1:
         return False, f"Invalid num_iterations: {config['num_iterations']}"
    if int(config.get("resume_from_step", 0)) < -1:
         return False, f"Invalid resume_from_step: {config['resume_from_step']}"

    return True, ""

if __name__ == "__main__":
    # 1. Parse known args to get a preliminary config for validation
    # This handles both manual runs and sweep agent passes.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--aspect-ratio", "--aspect_ratio", type=int, default=64)
    parser.add_argument("--head-dim", "--head_dim", type=int, default=64)
    parser.add_argument("--max-seq-len", "--max_seq_len", type=int, default=1024)
    parser.add_argument("--device-batch-size", "--device_batch_size", "--batch-size", type=int, default=128)
    parser.add_argument("--num-iterations", "--num_iterations", type=int, default=-1)
    parser.add_argument("--resume-from-step", "--resume_from_step", type=int, default=-1)
    parser.add_argument("--run", type=str, default="dummy")
    
    args, unknown = parser.parse_known_args()
    user_config = vars(args)

    # 2. Start a wandb run to get the FULL config (especially if running in a sweep)
    # If in a sweep, wandb.init() will merge the sweep config into what we provide.
    run = wandb.init(config=user_config)
    full_config = dict(run.config)
    
    # 3. Pre-flight validation
    is_valid, error_msg = validate_config(full_config)
    if not is_valid:
        print(f"PRE-FLIGHT ERROR: {error_msg}")
        run.finish(exit_code=1)
        # We exit with 0 to NOT crash the sweep agent, just skip this doomed run.
        # However, if it's a manual run, the user might want to know.
        if full_config.get("run") == "dummy":
             sys.exit(1)
        sys.exit(0) 

    # 4. Build command line for the subprocess
    # We use sys.argv[1:] but we must fix the keys (underscores to hyphens) 
    # and handle boolean flags (=true/false) as we did before.
    raw_args = sys.argv[1:]
    processed_args = []
    i = 0
    while i < len(raw_args):
        arg = raw_args[i]
        if arg.startswith("--"):
            if "=" in arg:
                key, val = arg.split("=", 1)
                key = key.replace("_", "-")
                val_lower = val.lower()
                if val_lower == "true":
                    processed_args.append(key)
                elif val_lower == "false":
                    pass 
                else:
                    processed_args.append(f"{key}={val}")
            else:
                key = arg.replace("_", "-")
                if i + 1 < len(raw_args) and raw_args[i+1].lower() in ["true", "false"]:
                    if raw_args[i+1].lower() == "true":
                        processed_args.append(key)
                    i += 1 
                else:
                    processed_args.append(key)
        else:
            processed_args.append(arg)
        i += 1

    print(f"DEBUG: Launching base_train with args: {processed_args}")
    sys.stdout.flush()

    # 5. Forward all arguments to scripts.base_train
    cmd = [sys.executable, "-u", "-m", "scripts.base_train"] + processed_args
    run.finish()

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with exit code {e.returncode}")
        sys.exit(e.returncode)
