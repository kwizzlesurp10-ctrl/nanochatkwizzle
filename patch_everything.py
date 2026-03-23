import sys
import torch

def dummy_sample(logits, rng, temperature=1.0, top_k=None):
    print("DEBUG: GLOBAL EMERGENCY PATCH CALLED", flush=True)
    return torch.argmax(logits, dim=-1, keepdim=True)

# Patch it in sys.modules if it's already there
for name, mod in sys.modules.items():
    if "nanochat.engine" in name:
        print(f"DEBUG: Patching already loaded module: {name}")
        mod.sample_next_token = dummy_sample

# Define a hook or just patch the module directly if we can import it
try:
    import nanochat.engine
    nanochat.engine.sample_next_token = dummy_sample
    print("DEBUG: Successfully patched nanochat.engine.sample_next_token")
except Exception as e:
    print(f"DEBUG: Could not patch nanochat.engine yet: {e}")
