import sys
import torch

def dummy_sample(logits, rng, temperature=1.0, top_k=None):
    print("DEBUG: EXTREME PATCH V17 CALLED (ARGMAX)", flush=True)
    return torch.argmax(logits, dim=-1, keepdim=True)

# Patch the module in sys.modules with a completely new fake module
import types
fake_engine = types.ModuleType("nanochat.engine")
fake_engine.sample_next_token = dummy_sample

# Overwrite everything
sys.modules["nanochat.engine"] = fake_engine

print("DEBUG: Extreme patch V17 applied. sys.modules['nanochat.engine'] replaced.")
