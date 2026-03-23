import os
import sys
sys.path.insert(0, os.getcwd())
import nanochat.engine
import inspect

print(f"nanochat.engine file: {nanochat.engine.__file__}")
with open(nanochat.engine.__file__, "r") as f:
    lines = f.readlines()
    for i in range(140, 160):
        if i < len(lines):
            print(f"{i+1}: {lines[i]}", end="")

print("\nsample_next_token in sys.modules:")
print(nanochat.engine.sample_next_token)