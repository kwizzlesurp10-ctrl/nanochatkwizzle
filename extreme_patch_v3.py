import sys
import torch

def dummy_sample(logits, rng, temperature=1.0, top_k=None):
    print("DEBUG: EXTREME PATCH V3 CALLED", flush=True)
    return torch.argmax(logits, dim=-1, keepdim=True)

# Patch in sys.modules
import nanochat.engine
sys.modules["nanochat.engine"].sample_next_token = dummy_sample

# Also patch in the current module's namespace if it's already imported elsewhere
def patch_all():
    print("INJECTING DEFINITIVE PATCH NOW", flush=True)
    for name, mod in list(sys.modules.items()):
        if hasattr(mod, "sample_next_token") and name.startswith("nanochat"):
            print(f"DEBUG: Patching mod: {name}")
            setattr(mod, "sample_next_token", dummy_sample)

patch_all()
