import sys
from safetensors import safe_open
from tqdm import tqdm
import argparse

def find_scaled_fp8_keys(model_path: str):
    """
    Scans a .safetensors file to find the locations of any 'scaled_fp8' tensors
    without loading the entire model into memory.
    """
    found_locations = []
    
    print(f"🔍 Scanning '{model_path}' for 'scaled_fp8' tensors...")
    
    try:
        # Use safe_open to iterate over tensors without loading them into RAM
        with safe_open(model_path, framework="pt", device="cpu") as f:
            # Iterate through all tensor names in the file
            for key in tqdm(f.keys(), desc="Scanning keys"):
                # Check if the key is exactly 'scaled_fp8' or ends with '.scaled_fp8'
                if key.endswith("scaled_fp8"):
                    # The "location" is the part of the name before 'scaled_fp8'
                    # rstrip('.') handles cases where the key is just 'scaled_fp8'
                    location = key.removesuffix("scaled_fp8").rstrip('.')
                    if not location:
                        location = "[root level]"
                    found_locations.append(location)

    except Exception as e:
        print(f"❌ An error occurred while reading the file: {e}")
        return

    if not found_locations:
        print("\n✅ No 'scaled_fp8' tensors were found in this model.")
    else:
        print(f"\n✅ Found {len(found_locations)} instance(s) of 'scaled_fp8' metadata in the following modules:")
        for loc in sorted(found_locations): # Sort for cleaner output
            print(f"  - {loc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find all occurrences of 'scaled_fp8' tensors in a .safetensors model file."
    )
    parser.add_argument(
        "model_path", 
        type=str, 
        help="The path to the .safetensors model file you want to inspect."
    )
    
    args = parser.parse_args()
    find_scaled_fp8_keys(args.model_path)