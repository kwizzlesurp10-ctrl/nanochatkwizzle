import sys
import torch

def dummy_sample(logits, rng, temperature=1.0, top_k=None):
    print("DEBUG: FINAL PATCH CALLED (ARGMAX)", flush=True)
    return torch.argmax(logits, dim=-1, keepdim=True)

# 1) Direct overwrite
import nanochat.engine
nanochat.engine.sample_next_token = dummy_sample

# 2) Global hook to catch any new imports
class PatchImporter:
    def find_spec(self, fullname, path, target=None):
        if fullname == "nanochat.engine":
            print(f"DEBUG: PatchImporter caught {fullname}")
        return None

sys.meta_path.insert(0, PatchImporter())

# 3) sys.modules injection
sys.modules["nanochat.engine"].sample_next_token = dummy_sample

print(f"DEBUG: Final patch applied. engine.sample_next_token is now {nanochat.engine.sample_next_token}")
