import sys
import torch

def dummy_sample(logits, rng, temperature=1.0, top_k=None):
    print("DEBUG: EXTREME PATCH V2 CALLED", flush=True)
    return torch.argmax(logits, dim=-1, keepdim=True)

# Patch everything that looks like nanochat.engine
for name, mod in list(sys.modules.items()):
    if "nanochat.engine" in name:
        print(f"DEBUG: Patching mod: {name}")
        mod.sample_next_token = dummy_sample

# In case it's not loaded yet, set it in sys.modules
import types
m = types.ModuleType("nanochat.engine")
m.sample_next_token = dummy_sample
# sys.modules["nanochat.engine"] = m # This might be too dangerous
