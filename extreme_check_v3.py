import os
import sys
import nanochat
import nanochat.engine
import inspect

print(f"nanochat.engine file: {nanochat.engine.__file__}")
print(f"sample_next_token location: {inspect.getfile(nanochat.engine.sample_next_token)}")

# Check for all nanochat modules in sys.modules
print("\nNanochat modules in sys.modules:")
for name in sorted(sys.modules.keys()):
    if "nanochat" in name:
        mod = sys.modules[name]
        try:
            path = getattr(mod, "__file__", "N/A")
            print(f"  {name}: {path}")
        except Exception:
            print(f"  {name}: <error>")

# List all files in the nanochat package
print("\nFiles in nanochat/ directory:")
try:
    for f in os.listdir("nanochat"):
        print(f"  {f}")
except Exception as e:
    print(f"Error listing nanochat/: {e}")