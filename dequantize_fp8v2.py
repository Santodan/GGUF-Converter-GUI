#!/usr/bin/env python
"""dequantize_fp8v2.py — Universal (ComfyUI & Standard) Dequantizer
* Auto-detects ComfyUI .weight_scale format
* Auto-detects Standard .scale format
* Aggressive memory cleanup for low-RAM environments
"""

import argparse
import re
import sys
import torch
import gc
from safetensors.torch import load_file, save_file

# --------- helpers & constants ---------
_WEIGHT_RE       = re.compile(r"\.weight$")
_FP8_DTYPES      = {torch.float8_e4m3fn, torch.float8_e5m2}
DTYPE_MAP        = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

def find_reciprocal_scale(state: dict[str, torch.Tensor], base: str) -> float:
    """
    Smart detection of scale factors.
    Returns the float value to MULTIPLY the weight by.
    """
    
    # --- STRATEGY 1: ComfyUI Native Format ---
    # Key is ".weight_scale" and it is a multiplier.
    if f"{base}.weight_scale" in state:
        return state[f"{base}.weight_scale"].float().item()

    # --- STRATEGY 2: Standard/Diffusers Format ---
    # Keys like ".scale", ".scale_weight". 
    # In many FP8 implementations, 'scale' implies we divide by it, 
    # so we return 1.0 / scale for multiplication.
    for suffix in ("scale", "scale_weight", "scale_reciprocal"):
        key = f"{base}.{suffix}"
        if key in state:
            scale_val = state[key].float().item()
            # Avoid division by zero
            if scale_val == 0: return 1.0
            return 1.0 / scale_val

    # --- STRATEGY 3: Explicit Inverse Scale ---
    if f"{base}.scale_inv" in state:
        return state[f"{base}.scale_inv"].float().item()

    # Failure
    return None

@torch.inference_mode()
def in_place_convert(state: dict[str, torch.Tensor], *, out_dtype: torch.dtype, strip_fp8: bool):
    """Cast **all** tensors to *out_dtype* in‑place, with aggressive memory cleanup."""
    
    # 1. Identify FP8 keys
    fp8_weight_keys = [k for k, t in state.items() if _WEIGHT_RE.search(k) and t.dtype in _FP8_DTYPES]

    restored = 0
    print(f"Processing {len(fp8_weight_keys)} FP8 tensors...")

    # 2. Process one by one
    for i, key in enumerate(fp8_weight_keys):
        tensor = state[key]
        base   = key[:-7] # removes ".weight"
        
        # DETECT SCALE
        recip  = find_reciprocal_scale(state, base)

        if recip is None:
            # Only warn periodically to avoid log spam on massive failures
            if restored < 5 or restored % 100 == 0:
                print(f"⚠️ Warning: No scale found for '{base}'. Defaulting to 1.0")
            recip = 1.0
        
        # --- MATH & CLEANUP ---
        # 1. Cast to F32
        temp = tensor.to(torch.float32)
        
        # 2. Apply Scale
        temp.mul_(recip)
        
        # 3. Cast to final dtype & Replace
        state[key] = temp.to(out_dtype)
        
        # 4. Immediate Cleanup
        del temp
        del tensor
        
        # 5. Metadata Cleanup (Strip FP8 artifacts)
        if strip_fp8:
            # Standard artifacts
            for suf in ("weight_scale", "scale_weight", "scale_input", "scale", "scale_inv"):
                k_scale = f"{base}.{suf}"
                if k_scale in state:
                    del state[k_scale]
            
            # ComfyUI artifacts
            k_junk = f"{base}.comfy_quant"
            if k_junk in state:
                del state[k_junk]

        restored += 1
        
        # 6. Garbage Collection (Crucial for 90GB+ RAM usage prevention)
        if i % 100 == 0:
            gc.collect()

    # ---- 3) Cleanup remaining junk (Global Sweep) ----
    print("Cleaning up non-FP8 tensors...")
    keys_to_check = list(state.keys())
    for k in keys_to_check:
        if k not in state: continue
        
        # Always remove known metadata garbage even if the layer wasn't FP8
        if k.endswith(".comfy_quant") or k.endswith(".weight_scale") or k.endswith("scale_inv"):
            del state[k]
            continue

        t = state[k]
        
        # Cast other tensors (biases, norms) to target dtype
        # BUT: Do not touch integers (like 'orig_shape' or indices)
        if t.is_floating_point() and t.dtype != out_dtype:
            state[k] = t.to(out_dtype)

    # Final cleanup before save
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n―――――――― CONVERSION SUMMARY ―――――――")
    print(f"FP8 weights restored : {restored}")
    print(f"Total tensors         : {len(state)}")
    print("――――――――――――――――――――――――――――――――")


def main() -> None:
    ap = argparse.ArgumentParser(description="Universal (Comfy/Standard) Dequantizer")
    ap.add_argument("--src", required=True, help="Input FP8 .safetensors file")
    ap.add_argument("--dst", required=True, help="Output .safetensors file")
    ap.add_argument("--dtype", choices=DTYPE_MAP.keys(), default="bf16")
    ap.add_argument("--strip-fp8", action="store_true")
    args = ap.parse_args()

    out_dtype = DTYPE_MAP[args.dtype]

    print(f"Loading {args.src} ...")
    sd = load_file(args.src, device="cpu")

    in_place_convert(sd, out_dtype=out_dtype, strip_fp8=args.strip_fp8)

    print(f"Saving to {args.dst} ...")
    try:
        save_file(sd, args.dst)
    except Exception as err:
        print("❌ Failed to save .safetensors:", err, file=sys.stderr)
        sys.exit(1)

    print("Done ✅")


if __name__ == "__main__":
    main()
