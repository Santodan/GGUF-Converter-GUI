import torch
from safetensors.torch import load_file, save_file
import argparse
import os

def extract_parts(src_path, dst_dir, extract_model, extract_clip, extract_vae):
    print(f"Loading {src_path}...")
    sd = load_file(src_path)
    basename = os.path.splitext(os.path.basename(src_path))[0]
    os.makedirs(dst_dir, exist_ok=True)

    # 1. EXTRACT VAE
    if extract_vae:
        print("Extracting VAE...")
        vae_sd = {}
        for k in list(sd.keys()):
            if k.startswith("first_stage_model."):
                vae_sd[k.replace("first_stage_model.", "")] = sd.pop(k)
            elif k.startswith("vae."):
                vae_sd[k.replace("vae.", "")] = sd.pop(k)
        if vae_sd:
            out_path = os.path.join(dst_dir, f"{basename}_vae.safetensors")
            save_file(vae_sd, out_path)
            print(f"✅ Saved VAE")

    # 2. EXTRACT MODEL (Diffusion)
    if extract_model:
        print("Extracting Diffusion Model...")
        model_sd = {}
        for k in list(sd.keys()):
            if k.startswith("model.diffusion_model."):
                model_sd[k.replace("model.diffusion_model.", "")] = sd.pop(k)
        
        if not model_sd:
             for k in list(sd.keys()):
                 if "transformer." in k or "diffusion_model." in k:
                     model_sd[k] = sd.pop(k)

        if model_sd:
            out_path = os.path.join(dst_dir, f"{basename}_model.safetensors")
            save_file(model_sd, out_path)
            print(f"✅ Saved Model")

    # 3. EXTRACT CLIP (Exact Parity with ComfyUI CLIPSave logic)
    if extract_clip:
        print("Extracting CLIP components...")
        
        # In SDXL/Illustrious: 
        # conditioner.embedders.0 = CLIP-L
        # conditioner.embedders.1 = CLIP-G
        
        # We define search patterns that match ComfyUI's internal state_dict_prefix_replace
        search_targets = [
            ("conditioner.embedders.0.", "clip_l"),
            ("conditioner.embedders.1.", "clip_g"),
            ("conditioner.embedders.2.", "t5xxl"),
            ("cond_stage_model.clip_l.", "clip_l"),
            ("cond_stage_model.clip_g.", "clip_g"),
            ("cond_stage_model.transformer.", "clip_l"),
            ("cond_stage_model.", "clip_l"),
        ]

        for prefix, label in search_targets:
            current_clip_sd = {}
            # Step 1: Find all keys belonging to this embedder
            target_keys = [k for k in sd.keys() if k.startswith(prefix)]
            if not target_keys:
                continue

            for k in target_keys:
                # Step 2: Pop from master to prevent "leaks" or duplicates
                tensor = sd.pop(k)
                
                # Step 3: Strip the root prefix (e.g., 'conditioner.embedders.0.')
                clean_key = k[len(prefix):]
                
                # Step 4: Strip 'transformer.' or 'model.' if it exists at the start
                # This matches ComfyUI's replace_prefix["transformer."] = ""
                if clean_key.startswith("transformer."):
                    clean_key = clean_key[12:]
                elif clean_key.startswith("model."):
                    clean_key = clean_key[6:]
                
                current_clip_sd[clean_key] = tensor

            if current_clip_sd:
                # Step 5: Final validation of the label by checking tensor shapes
                # This prevents the "1280 vs 768" error if a model has swapped indices
                actual_label = label
                test_key = "text_model.embeddings.token_embedding.weight"
                if test_key not in current_clip_sd:
                    # Fallback key check
                    test_key = "token_embedding.weight"
                
                if test_key in current_clip_sd:
                    dim = current_clip_sd[test_key].shape[-1]
                    if dim == 768: actual_label = "clip_l"
                    elif dim == 1280: actual_label = "clip_g"
                    elif dim == 4096: actual_label = "t5xxl"

                out_path = os.path.join(dst_dir, f"{basename}_{actual_label}.safetensors")
                save_file(current_clip_sd, out_path)
                print(f"✅ Saved {actual_label.upper()} ({len(current_clip_sd)} tensors)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst_dir", type=str, required=True)
    parser.add_argument("--model", action="store_true")
    parser.add_argument("--clip", action="store_true")
    parser.add_argument("--vae", action="store_true")
    args = parser.parse_args()
    extract_parts(args.src, args.dst_dir, args.model, args.clip, args.vae)