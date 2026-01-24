# list_weights.py
#python list_weights.py path/to/smooth-mix-wan-22.safetensors > keys.txt

from safetensors import safe_open
import sys

if len(sys.argv) < 2:
    print("Usage: python list_weights.py <path_to_safetensors>")
    sys.exit(1)

path = sys.argv[1]

with safe_open(path, framework="pt", device="cpu") as f:
    for k in f.keys():
        # Newer versions use get_tensor; older ones can slice
        try:
            shape = f.get_tensor(k).shape
        except Exception:
            # fallback if tensor too large or framework mismatch
            try:
                shape = f.get_slice(k).get_shape()
            except Exception:
                shape = "(unknown)"
        print(k, shape)
