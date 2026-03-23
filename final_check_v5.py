import os
import sys
sys.path.insert(0, os.getcwd())
import nanochat.engine
import inspect

print(f"nanochat.engine file: {nanochat.engine.__file__}")
with open(nanochat.engine.__file__, "r") as f:
    lines = f.readlines()
    print(f"Total lines: {len(lines)}")
    if len(lines) >= 151:
        print(f"Line 151: {lines[150].strip()}")
    
    # Show generate() body around where sample_next_token is called
    try:
        src = inspect.getsource(nanochat.engine.Engine.generate)
        print("\nEngine.generate source (partial):")
        # Find 'sample_next_token' in source
        idx = src.find('sample_next_token')
        if idx != -1:
            print(src[max(0, idx-100):idx+200])
        else:
            print("sample_next_token NOT FOUND in Engine.generate source!")
            print(src[:500])
    except Exception as e:
        print(f"Error getting source: {e}")