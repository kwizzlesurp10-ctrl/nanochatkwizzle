import os
import sys
sys.path.insert(0, os.getcwd())
import nanochat.engine
print(f"nanochat.engine module: {nanochat.engine}")
print(f"Engine class: {nanochat.engine.Engine}")
print(f"Engine file path: {nanochat.engine.__file__}")

import inspect
print("Engine.generate source code:")
print(inspect.getsource(nanochat.engine.Engine.generate))
