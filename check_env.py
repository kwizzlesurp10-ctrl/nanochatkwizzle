import os
import sys
import nanochat
import nanochat.engine
import inspect

print(f"sys.path: {sys.path}")
print(f"nanochat file: {nanochat.__file__}")
print(f"nanochat.engine file: {nanochat.engine.__file__}")
try:
    print("Engine source snippet:")
    src = inspect.getsource(nanochat.engine.Engine)
    print(src[:200])
except Exception as e:
    print(f"Error: {e}")
