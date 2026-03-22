import torch
import os
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.common import get_checkpoint_base_dir

def test():
    base_dir = get_checkpoint_base_dir()
    checkpoint_dir = os.path.join(base_dir, "test_checkpoints", "test_model")
    model_data = {"weight": torch.randn(10, 10)}
    optimizer_data = {"state": {}}
    meta_data = {"config": {}}
    
    print(f"Attempting to save to {checkpoint_dir}...")
    save_checkpoint(checkpoint_dir, 1, model_data, optimizer_data, meta_data, rank=0)
    print("Save completed successfully!")

if __name__ == "__main__":
    test()
