import torch
import sys

def patch_sample():
    def definitive_argmax_patch(logits, *args, **kwargs):
        print("DEBUG: EMERGENCY PATCH V2 EXECUTING", flush=True)
        return torch.argmax(logits.detach().cpu().to(torch.float32), dim=-1, keepdim=True)

    # 1) Force into sys.modules
    import nanochat.engine
    nanochat.engine.sample_next_token = definitive_argmax_patch
    sys.modules['nanochat.engine'].sample_next_token = definitive_argmax_patch

    # 2) Patch Engine class
    from nanochat.engine import Engine
    Engine.sample_next_token = staticmethod(definitive_argmax_patch)
    
    # 3) Patch globals
    Engine.generate.__globals__['sample_next_token'] = definitive_argmax_patch
    
    # 4) Patch in ALL modules
    for name, mod in list(sys.modules.items()):
        if name.startswith("nanochat"):
            if hasattr(mod, "sample_next_token"):
                setattr(mod, "sample_next_token", definitive_argmax_patch)

    print("DEBUG: Emergency patch V2 fully applied", flush=True)

if __name__ == "__main__":
    patch_sample()
else:
    patch_sample()
