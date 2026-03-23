import sys
import torch

def dummy_sample(logits, rng, temperature=1.0, top_k=None):
    print("DEBUG: FINAL PATCH V3 CALLED (ARGMAX)", flush=True)
    return torch.argmax(logits, dim=-1, keepdim=True)

# 1) Direct overwrite in Engine class
import nanochat.engine
from nanochat.engine import Engine
Engine.sample_next_token = staticmethod(dummy_sample)

# 2) Patch global in that module
setattr(nanochat.engine, "sample_next_token", dummy_sample)

# 3) sys.modules injection
sys.modules['nanochat.engine'].sample_next_token = dummy_sample

# 4) Patch generate globals
Engine.generate.__globals__['sample_next_token'] = dummy_sample

# 5) Builtins override
import builtins
builtins.sample_next_token = dummy_sample

# 6) Patch in ALL modules
for name, mod in list(sys.modules.items()):
    if hasattr(mod, 'sample_next_token'):
        try:
            setattr(mod, 'sample_next_token', dummy_sample)
        except Exception:
            pass

print(f"DEBUG: V3 Patch finished. Engine.sample_next_token is {Engine.sample_next_token}")
