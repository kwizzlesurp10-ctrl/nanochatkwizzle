import os
import sys
sys.path.insert(0, os.getcwd())
import nanochat.engine
import inspect

print(f"nanochat.engine file: {nanochat.engine.__file__}")
print(f"sample_next_token location: {inspect.getfile(nanochat.engine.sample_next_token)}")

with open(nanochat.engine.__file__, "r") as f:
    lines = f.readlines()
    print("--- Line 145-155 ---")
    for i in range(144, min(155, len(lines))):
        print(f"{i+1}: {lines[i]}", end="")
