import torch
from safetensors.torch import load_file, save_file
import argparse
import os

def extract_parts(src_path, dst_dir, extract_model, extract_clip, extract_vae):
    print(f"Loading {src_path}...")
    sd = load_file(src_path)
    basename = os.path.splitext(os.path.basename(src_path))[0]
    os.makedirs(dst_dir, exist_ok=True)

    # 1. EXTRACT MODEL (Diffusion/Transformer)
    if extract_model:
        print("Extracting Diffusion Model...")
        model_sd = {k.replace("model.diffusion_model.", ""): v for k, v in sd.items() if k.startswith("model.diffusion_model.")}
        if not model_sd: # For Flux/Wan/etc
             model_sd = {k: v for k, v in sd.items() if "transformer." in k or "diffusion_model." in k}
        if model_sd:
            out_path = os.path.join(dst_dir, f"{basename}_model.safetensors")
            save_file(model_sd, out_path)
            print(f"Saved Model to: {out_path}")

    # 2. EXTRACT VAE
    if extract_vae:
        print("Extracting VAE...")
        vae_sd = {k.replace("first_stage_model.", "").replace("vae.", ""): v for k, v in sd.items() if k.startswith("first_stage_model.") or k.startswith("vae.")}
        if vae_sd:
            out_path = os.path.join(dst_dir, f"{basename}_vae.safetensors")
            save_file(vae_sd, out_path)
            print(f"Saved VAE to: {out_path}")

    # 3. EXTRACT CLIP (Enhanced Search Logic)
    if extract_clip:
        print("Extracting CLIP(s)...")
        # These are the standard prefixes ComfyUI looks for
        prefixes = ["clip_l.", "clip_g.", "clip_h.", "t5xxl.", "pile_t5xl.", "mt5xl.", "umt5xxl.", "t5base.", "gemma2_2b.", "llama.", "hydit_clip."]
        
        # Baked-in prefixes often used in SDXL/Illustrious/Flux
        baked_containers = ["cond_stage_model.", "conditioner.embedders.0.", "conditioner.embedders.1.", "conditioner.embedders.2."]

        found_any = False

        for prefix in prefixes:
            current_clip_sd = {}
            for k in list(sd.keys()):
                target_key = None
                
                # Check for standard prefix
                if k.startswith(prefix):
                    target_key = k.replace(prefix, "")
                else:
                    # Check inside baked containers
                    for container in baked_containers:
                        if k.startswith(container + prefix):
                            target_key = k.replace(container + prefix, "")
                            break
                        # Special case: SDXL baked models often just have 'transformer' inside the container
                        elif k.startswith(container + "transformer.") and prefix in ["clip_l.", "clip_g."]:
                             # We only do this if we haven't found a better match
                             target_key = k.replace(container, "")

                if target_key:
                    # ComfyUI strips 'transformer.' and 'model.' from the internal CLIP sd
                    target_key = target_key.replace("transformer.", "").replace("model.", "")
                    current_clip_sd[target_key] = sd[k]

            if current_clip_sd and len(current_clip_sd) > 5: # Ignore tiny fragments
                p_name = prefix.rstrip('.')
                out_path = os.path.join(dst_dir, f"{basename}_{p_name}.safetensors")
                save_file(current_clip_sd, out_path)
                print(f"Saved CLIP {p_name} to: {out_path}")
                found_any = True

        # Fallback for SD1.5 or models where CLIP has no prefix inside cond_stage_model
        if not found_any:
            clip_sd = {k.replace("cond_stage_model.transformer.", "").replace("cond_stage_model.", ""): v 
                       for k, v in sd.items() if k.startswith("cond_stage_model.") and "diffusion_model" not in k}
            if clip_sd:
                out_path = os.path.join(dst_dir, f"{basename}_clip.safetensors")
                save_file(clip_sd, out_path)
                print(f"Saved CLIP (Standard) to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst_dir", type=str, required=True)
    parser.add_argument("--model", action="store_true")
    parser.add_argument("--clip", action="store_true")
    parser.add_argument("--vae", action="store_true")
    args = parser.parse_args()
    extract_parts(args.src, args.dst_dir, args.model, args.clip, args.vae)