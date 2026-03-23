import sys
import torch

def dummy_sample(logits, rng, temperature=1.0, top_k=None):
    print("DEBUG: EXTREME PATCH V7 CALLED (BYPASSING MULTINOMIAL)", flush=True)
    return torch.argmax(logits, dim=-1, keepdim=True)

# Patch in sys.modules
import nanochat.engine
sys.modules["nanochat.engine"].sample_next_token = dummy_sample

# Patch all nanochat modules
def patch_all():
    for name, mod in list(sys.modules.items()):
        if name.startswith("nanochat"):
            print(f"DEBUG: Force-injecting sample_next_token into mod: {name}")
            setattr(mod, "sample_next_token", dummy_sample)

patch_all()

# Override the reference in Engine.generate itself
from nanochat.engine import Engine
import inspect

print(f"DEBUG: Engine.generate globals: {Engine.generate.__globals__.keys()}")
Engine.generate.__globals__["sample_next_token"] = dummy_sample
print(f"DEBUG: Patched Engine.generate globals['sample_next_token']")
