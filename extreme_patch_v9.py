import sys
import torch

def dummy_sample(logits, rng, temperature=1.0, top_k=None):
    print("DEBUG: EXTREME PATCH V9 CALLED (ARGMAX)", flush=True)
    return torch.argmax(logits, dim=-1, keepdim=True)

# 1) Force patch in sys.modules
import nanochat.engine
sys.modules['nanochat.engine'].sample_next_token = dummy_sample

# 2) Patch Engine.generate globals
from nanochat.engine import Engine
Engine.generate.__globals__['sample_next_token'] = dummy_sample

# 3) Patch all known locations
import scripts.chat_web
scripts.chat_web.sample_next_token = dummy_sample

# 4) Patch in all modules that might have imported it
for name, mod in list(sys.modules.items()):
    if hasattr(mod, 'sample_next_token'):
        print(f"DEBUG: Patching mod {name}")
        setattr(mod, 'sample_next_token', dummy_sample)

print("DEBUG: Extreme patch V9 applied")
