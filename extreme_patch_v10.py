import sys
import torch

def dummy_sample(logits, rng, temperature=1.0, top_k=None):
    print("DEBUG: EXTREME PATCH V10 CALLED (ARGMAX)", flush=True)
    return torch.argmax(logits, dim=-1, keepdim=True)

# Patch in sys.modules
import nanochat.engine
sys.modules["nanochat.engine"].sample_next_token = dummy_sample

# Also patch the global in the module itself
nanochat.engine.sample_next_token = dummy_sample

print(f"DEBUG: Patch V10 applied to {nanochat.engine.sample_next_token}")
