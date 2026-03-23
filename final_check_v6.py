import os
import sys
sys.path.insert(0, os.getcwd())
import nanochat.engine
import inspect

print(f"nanochat.engine file: {nanochat.engine.__file__}")
print(f"sample_next_token location: {inspect.getfile(nanochat.engine.sample_next_token)}")
print(f"Engine.generate location: {inspect.getfile(nanochat.engine.Engine.generate)}")

# Get the actual source code of Engine.generate
src = inspect.getsource(nanochat.engine.Engine.generate)
print("\nEngine.generate logic for sampling:")
import re
match = re.search(r"next_ids = .*?\n", src)
if match:
    print(match.group(0))
else:
    print("Could not find sampling line in source")