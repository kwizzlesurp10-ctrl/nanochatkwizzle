import os
import sys
sys.path.insert(0, os.getcwd())
import nanochat.engine
print(f"nanochat.engine file: {nanochat.engine.__file__}")
with open(nanochat.engine.__file__, "r") as f:
    lines = f.readlines()
    print(f"Line 151: {lines[150] if len(lines) > 150 else 'N/A'}")
    # Search for multinomial
    for i, line in enumerate(lines):
        if "multinomial" in line:
            print(f"Line {i+1}: {line.strip()}")
