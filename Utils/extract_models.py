import sys
import subprocess
import os

# --- Dependency Management ---
required_packages = ['torch', 'safetensors', 'prompt-toolkit', 'tqdm']

try:
    import torch
    from safetensors.torch import load_file, save_file
    from prompt_toolkit import prompt
    from prompt_toolkit.completion import PathCompleter
    from tqdm import tqdm
except ImportError:
    print("One or more required libraries are not installed. Attempting to install...")
    python_executable = sys.executable
    try:
        for package in required_packages:
            print(f"Installing {package}...")
            subprocess.check_call([python_executable, "-m", "pip", "install", package])
        print("\nDependencies installed successfully. Please re-run the script.")
        sys.exit()
    except Exception as e:
        print(f"\nAn error occurred: {e}\nPlease install manually: pip install {' '.join(required_packages)}")
        sys.exit(1)

# --- Core Functions ---

def get_paths():
    """Prompts the user for the input checkpoint file and the output directory."""
    print("\nPlease provide the following paths. Use Tab for completion.")
    file_completer = PathCompleter(expanduser=True)
    folder_completer = PathCompleter(only_directories=True, expanduser=True)
    checkpoint_path = prompt("Enter path to the checkpoint file (.safetensors): ", completer=file_completer)
    output_folder = prompt("Enter path to the output folder: ", completer=folder_completer)
    return checkpoint_path, output_folder

def get_component_choices(is_sdxl):
    """Prompts the user to select components based on the model type."""
    if is_sdxl:
        print("\nSDXL Model Detected. Which components would you like to save?")
        print("  1: CLIP-L (Text Encoder 1)")
        print("  2: CLIP-G (Text Encoder 2, OpenCLIP)")
        print("  3: UNET (Denoising Model)")
        print("  4: VAE (Image Encoder/Decoder)")
        print("  5: All of the above")
        component_map = {'1': 'CLIP_L', '2': 'CLIP_G', '3': 'UNET', '4': 'VAE'}
        all_option = '5'
        all_components = ['CLIP_L', 'CLIP_G', 'UNET', 'VAE']
    else:
        print("\nSD 1.5 Model Detected. Which components would you like to save?")
        print("  1: CLIP (Text Encoder)")
        print("  2: UNET (Denoising Model)")
        print("  3: VAE (Image Encoder/Decoder)")
        print("  4: All of the above")
        component_map = {'1': 'CLIP', '2': 'UNET', '3': 'VAE'}
        all_option = '4'
        all_components = ['CLIP', 'UNET', 'VAE']
    
    while True:
        user_input = prompt(f"Enter your choice(s), separated by commas (e.g., 1,{len(component_map) - 1}): ")
        choices = {c.strip() for c in user_input.split(',')}
        
        if all_option in choices:
            return all_components
            
        selected_components = []
        valid_choices = True
        for choice in choices:
            if choice in component_map:
                selected_components.append(component_map[choice])
            else:
                print(f"Invalid choice: '{choice}'. Please use numbers from the menu.")
                valid_choices = False
                break
        
        if valid_choices and selected_components:
            return list(set(selected_components))
        elif valid_choices and not selected_components:
             print("No choice entered. Please try again.")

def extract_and_save_models(checkpoint_path, output_folder, components_to_save):
    """Loads a checkpoint, detects model type, separates tensors, and saves selected components."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    try:
        print(f"\nLoading checkpoint: {checkpoint_path} (this may take a while)...")
        state_dict = load_file(checkpoint_path, device="cpu")
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # --- Model Type Detection ---
    is_sdxl = any(key.startswith("conditioner.embedders.1.") for key in state_dict.keys())
    
    # Get user choices *after* detecting model type
    user_choices = get_component_choices(is_sdxl)

    # --- Tensor Separation ---
    model_state_dicts = {comp: {} for comp in ['CLIP', 'CLIP_L', 'CLIP_G', 'UNET', 'VAE']}
    
    print("Separating model components...")
    for key, tensor in tqdm(state_dict.items(), desc="Processing Tensors"):
        if key.startswith("model.diffusion_model."):
            model_state_dicts['UNET'][key] = tensor
        elif key.startswith("first_stage_model."):
            model_state_dicts['VAE'][key] = tensor
        # SD 1.5 CLIP
        elif key.startswith("cond_stage_model."):
            model_state_dicts['CLIP'][key] = tensor
        # SDXL CLIPs
        elif key.startswith("conditioner.embedders.0."):
            model_state_dicts['CLIP_L'][key] = tensor
        elif key.startswith("conditioner.embedders.1."):
            model_state_dicts['CLIP_G'][key] = tensor
            
    # --- Saving Logic ---
    base_filename = os.path.splitext(os.path.basename(checkpoint_path))[0]
    saved_count = 0
    print("\nSaving selected components...")

    component_save_map = {
        'CLIP': ('_clip', model_state_dicts['CLIP']),
        'CLIP_L': ('_clip_l', model_state_dicts['CLIP_L']),
        'CLIP_G': ('_clip_g', model_state_dicts['CLIP_G']),
        'UNET': ('_unet', model_state_dicts['UNET']),
        'VAE': ('_vae', model_state_dicts['VAE']),
    }

    for component_name in user_choices:
        suffix, state_dict_to_save = component_save_map[component_name]
        
        if state_dict_to_save:
            output_path = os.path.join(output_folder, f"{base_filename}{suffix}.safetensors")
            print(f"  -> Saving {component_name} to: {output_path} ...", end="", flush=True)
            save_file(state_dict_to_save, output_path)
            print(" Done.")
            saved_count += 1
        else:
            print(f"  -> Warning: No '{component_name}' tensors were found to save.")

    if saved_count > 0:
        print(f"\nExtraction complete! Saved {saved_count} component(s).")
    else:
        print("\nNo components were saved.")

# --- Main Execution ---

if __name__ == "__main__":
    checkpoint_file, output_dir = get_paths()
    if os.path.isfile(checkpoint_file):
        # We now pass the choices from the main function down
        # The main extract function will handle calling get_component_choices
        extract_and_save_models(checkpoint_file, output_dir, []) # Pass empty list initially
    else:
        print("\nError: The specified checkpoint file does not exist.")