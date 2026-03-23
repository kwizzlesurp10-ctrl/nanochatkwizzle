import sys
import torch

def dummy_sample(logits, rng, temperature=1.0, top_k=None):
    print("DEBUG: EXTREME PATCH V8 CALLED (ARGMAX)", flush=True)
    return torch.argmax(logits, dim=-1, keepdim=True)

# Patch Engine.generate directly in its own module
import nanochat.engine
from nanochat.engine import Engine

print(f"DEBUG: Engine.generate is {Engine.generate}")
print(f"DEBUG: Engine module is {sys.modules['nanochat.engine']}")

# Overwrite the global in that module
sys.modules['nanochat.engine'].sample_next_token = dummy_sample
nanochat.engine.sample_next_token = dummy_sample

# ALSO patch the function if it was imported into chat_web
import scripts.chat_web
if hasattr(scripts.chat_web, "sample_next_token"):
    scripts.chat_web.sample_next_token = dummy_sample

print("DEBUG: Extreme patch V8 finished")
