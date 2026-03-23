import sys
import torch

def dummy_sample(logits, rng, temperature=1.0, top_k=None):
    print("DEBUG: EXTREME PATCH V5 CALLED (BYPASSING MULTINOMIAL)", flush=True)
    return torch.argmax(logits, dim=-1, keepdim=True)

import nanochat.engine
# MANUALLY ASSIGN TO THE MODULE
nanochat.engine.sample_next_token = dummy_sample

# ALSO PATCH IN SYS.MODULES
if "nanochat.engine" in sys.modules:
    sys.modules["nanochat.engine"].sample_next_token = dummy_sample

# ALSO PATCH THE ENTIRE NANOCHAT PACKAGE
import nanochat
nanochat.sample_next_token = dummy_sample

print(f"DEBUG: Definitive patch V5 applied to {nanochat.engine.sample_next_token}")
