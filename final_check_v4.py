import os
import sys
sys.path.insert(0, os.getcwd())
import nanochat.engine
print(f"nanochat.engine file: {nanochat.engine.__file__}")
with open(nanochat.engine.__file__, "r") as f:
    src = f.read()
    if "choice = torch.multinomial(probs, num_samples=1, generator=rng)" in src:
        print("BAD CODE FOUND IN LOADED FILE!")
    else:
        print("BAD CODE NOT FOUND IN LOADED FILE.")
