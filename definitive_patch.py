import torch
import builtins
import sys

def definitive_argmax_patch(logits, *args, **kwargs):
    print("DEBUG: DEFINITIVE PATCH EXECUTING (ARGMAX)", flush=True)
    # Be flexible about how logits are passed
    l = kwargs.get('logits', logits)
    return torch.argmax(l.detach().cpu().to(torch.float32), dim=-1, keepdim=True)

# 1) builtins
builtins.sample_next_token = definitive_argmax_patch

# 2) sys.modules
import nanochat.engine
sys.modules['nanochat.engine'].sample_next_token = definitive_argmax_patch

# 3) Engine class
from nanochat.engine import Engine
Engine.sample_next_token = staticmethod(definitive_argmax_patch)

# 4) generate globals
Engine.generate.__globals__['sample_next_token'] = definitive_argmax_patch

print("DEBUG: Definitive patch fully applied")
