print("STARTING EXTREME PATCH", flush=True)
import sys
import torch

def dummy_sample(logits, rng, temperature=1.0, top_k=None):
    print("DEBUG: EXTREME PATCH CALLED", flush=True)
    return torch.argmax(logits, dim=-1, keepdim=True)

class ModulePatch:
    def __init__(self, name):
        self.name = name
    def __getattr__(self, item):
        if item == "sample_next_token":
            return dummy_sample
        raise AttributeError(f"Module {self.name} has no attribute {item}")

# Forcefully inject into sys.modules
import nanochat.engine
sys.modules["nanochat.engine"].sample_next_token = dummy_sample
print(f"DEBUG: Injected into sys.modules['nanochat.engine']")
