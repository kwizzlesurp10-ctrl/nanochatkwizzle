import sys
import torch

def dummy_sample(logits, rng, temperature=1.0, top_k=None):
    print("DEBUG: EXTREME PATCH V13 CALLED (ARGMAX)", flush=True)
    return torch.argmax(logits, dim=-1, keepdim=True)

# 1) Force into sys.modules
import nanochat.engine
sys.modules['nanochat.engine'].sample_next_token = dummy_sample

# 2) Patch Engine class in that module
from nanochat.engine import Engine
Engine.sample_next_token = staticmethod(dummy_sample)

# 3) Patch Engine.generate globals
Engine.generate.__globals__['sample_next_token'] = dummy_sample

# 4) Patch scripts.chat_web
import scripts.chat_web
scripts.chat_web.sample_next_token = dummy_sample

# 5) Iterative patch
for name, mod in list(sys.modules.items()):
    if name.startswith("nanochat") or name.startswith("scripts"):
        if hasattr(mod, "sample_next_token"):
            print(f"DEBUG: V13 Patching {name}.sample_next_token")
            setattr(mod, "sample_next_token", dummy_sample)

print("DEBUG: Extreme patch V13 finished")
