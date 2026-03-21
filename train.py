import subprocess
import sys

if __name__ == "__main__":
    # Pre-process arguments to handle wandb sweep format
    raw_args = sys.argv[1:]
    processed_args = []
    
    # List of known hyphenated arguments in base_train.py
    # (Easier to just replace all underscores in keys starting with --)
    
    i = 0
    while i < len(raw_args):
        arg = raw_args[i]
        
        # 1. Standardize key format (convert --key_name to --key-name)
        if arg.startswith("--"):
            if "=" in arg:
                key, val = arg.split("=", 1)
                key = key.replace("_", "-")
                
                # 2. Handle boolean-style values
                val_lower = val.lower()
                if val_lower == "true":
                    processed_args.append(key)
                elif val_lower == "false":
                    pass # skip
                else:
                    processed_args.append(f"{key}={val}")
            else:
                key = arg.replace("_", "-")
                # Check for --key true/false
                if i + 1 < len(raw_args) and raw_args[i+1].lower() in ["true", "false"]:
                    if raw_args[i+1].lower() == "true":
                        processed_args.append(key)
                    i += 1 # skip val
                else:
                    processed_args.append(key)
        else:
            processed_args.append(arg)
        i += 1

    print(f"DEBUG: Raw args: {raw_args}")
    print(f"DEBUG: Processed args: {processed_args}")
    sys.stdout.flush()

    # Forward all arguments to scripts.base_train
    cmd = [sys.executable, "-u", "-m", "scripts.base_train"] + processed_args
    subprocess.run(cmd, check=True)
