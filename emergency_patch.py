import sys
import torch

def dummy_sample(logits, rng, temperature=1.0, top_k=None):
    print("DEBUG: EMERGENCY PATCH EXECUTING (ARGMAX)", flush=True)
    return torch.argmax(logits, dim=-1, keepdim=True)

# 1) Overwrite in the module itself
import nanochat.engine
nanochat.engine.sample_next_token = dummy_sample

# 2) Overwrite in ALL modules currently in sys.modules
for name, mod in list(sys.modules.items()):
    if "nanochat" in name:
        if hasattr(mod, "sample_next_token"):
            setattr(mod, "sample_next_token", dummy_sample)

# 3) Overwrite in Engine.generate globals specifically
from nanochat.engine import Engine
Engine.generate.__globals__["sample_next_token"] = dummy_sample

print(f"DEBUG: Emergency patch applied. engine.sample_next_token is now {nanochat.engine.sample_next_token}")