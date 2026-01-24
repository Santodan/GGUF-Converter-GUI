import torch
import argparse
import numpy as np
from safetensors.torch import load_file, save_file
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Isolate UNet, merge 5D fix, and flatten tensors for GGUF conversion.")
    parser.add_argument("--model", required=True, help="Path to the UNET-ONLY safetensors file from ComfyUI.")
    parser.add_argument("--fix", required=True, help="Path to the 5D tensor fix safetensors file.")
    parser.add_argument("--output", required=True, help="Path to save the new, prepared safetensors file.")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    print("Loading UNET-only model...")
    state_dict = load_file(args.model, device="cpu")
    
    print("Loading 5D tensor fix file...")
    fix_dict = load_file(args.fix, device="cpu")

    # --- START OF FINAL CRITICAL FIX ---
    # The UNET has the prefix 'model.diffusion_model.' but the fix file does not.
    # We must add the prefix to the fix keys before merging.
    prefixed_fix_dict = {}
    prefix = "model.diffusion_model."
    print(f"Adding prefix '{prefix}' to 5D fix tensor keys for correct merging...")
    for key, tensor in fix_dict.items():
        if not key.startswith(prefix):
            prefixed_key = prefix + key
            prefixed_fix_dict[prefixed_key] = tensor
            print(f"  '{key}' -> '{prefixed_key}'")
        else:
            prefixed_fix_dict[key] = tensor # Already has prefix, copy as-is
    # --- END OF FINAL CRITICAL FIX ---
    
    state_dict.update(prefixed_fix_dict)
    print("Models merged correctly.")

    metadata = {}
    
    for key in list(state_dict.keys()):
        tensor = state_dict[key]
        if tensor.ndim > 4:
            print(f"Processing tensor '{key}' with shape {tensor.shape} (ndim={tensor.ndim})")
            
            orig_shape = tensor.shape
            metadata[key] = {"orig_shape": [int(d) for d in orig_shape]}
            
            state_dict[key] = tensor.flatten()
            print(f"Flattened '{key}' to shape {state_dict[key].shape}")

    print(f"\nSaving prepared model to: {args.output}")
    save_file(state_dict, args.output, metadata={"gguf_metadata": str(metadata)})
    
    print("\nPreparation complete.")