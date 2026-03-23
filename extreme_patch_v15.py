import sys
import torch

def dummy_sample(logits, rng, temperature=1.0, top_k=None):
    print("DEBUG: EXTREME PATCH V15 CALLED (ARGMAX)", flush=True)
    return torch.argmax(logits, dim=-1, keepdim=True)

# Patch Engine.generate globals AND sample_next_token in that module
import nanochat.engine
from nanochat.engine import Engine

# 1) Direct overwrite
nanochat.engine.sample_next_token = dummy_sample

# 2) sys.modules injection
if "nanochat.engine" in sys.modules:
    sys.modules["nanochat.engine"].sample_next_token = dummy_sample

# 3) Patch Engine.generate globals
if hasattr(Engine.generate, "__globals__"):
    Engine.generate.__globals__["sample_next_token"] = dummy_sample

# 4) Builtins override
import builtins
builtins.sample_next_token = dummy_sample

print(f"DEBUG: V15 Patch finished. engine.sample_next_token is {nanochat.engine.sample_next_token}")
