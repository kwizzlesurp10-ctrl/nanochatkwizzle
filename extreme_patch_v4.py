import sys
import torch

def dummy_sample(logits, rng, temperature=1.0, top_k=None):
    print("DEBUG: EXTREME PATCH V4 CALLED (BYPASSING MULTINOMIAL)", flush=True)
    return torch.argmax(logits, dim=-1, keepdim=True)

import nanochat.engine
# Replace the function itself
nanochat.engine.sample_next_token = dummy_sample

# Replace it in sys.modules just to be sure
if "nanochat.engine" in sys.modules:
    sys.modules["nanochat.engine"].sample_next_token = dummy_sample

print(f"DEBUG: Definitive patch applied to {nanochat.engine.sample_next_token}")
