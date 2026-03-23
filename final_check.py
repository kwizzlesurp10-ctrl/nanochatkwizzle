import os
import sys
sys.path.insert(0, os.getcwd())
import nanochat
import nanochat.engine
import inspect

print(f"nanochat file: {nanochat.__file__}")
print(f"nanochat.engine file: {nanochat.engine.__file__}")
print("Engine.generate source code snippet:")
try:
    source = inspect.getsource(nanochat.engine.Engine.generate)
    print(source[:500])
except Exception as e:
    print(f"Error getting source: {e}")

print("\nsample_next_token location:")
try:
    print(inspect.getfile(nanochat.engine.sample_next_token))
except Exception as e:
    print(f"Error getting sample_next_token file: {e}")