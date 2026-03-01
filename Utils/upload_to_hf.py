import os
import sys
import argparse
import glob
import importlib
import time
import psutil
import subprocess

# Dependency Checker
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_TRANSFER_CONCURRENCY"] = "1"
os.environ["HF_HUB_TRANSFER_CHUNK_SIZE"] = "5242880"  # 5 MB chunks
os.environ["HUGGINGFACE_HUB_READ_TIMEOUT"] = "600"
os.environ["TOKIO_WORKER_THREADS"] = "1"

# === DIAGNOSTICS ===
print(f"[DEBUG] System Memory: {psutil.virtual_memory().percent}% used")
print(f"[DEBUG] HF_HUB_ENABLE_HF_TRANSFER: {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER')}")
print(f"[DEBUG] HF_HUB_TRANSFER_CONCURRENCY: {os.environ.get('HF_HUB_TRANSFER_CONCURRENCY')}")
print(f"[DEBUG] HF_HUB_TRANSFER_CHUNK_SIZE: {os.environ.get('HF_HUB_TRANSFER_CHUNK_SIZE')}")
REQUIRED_PACKAGES = {"hf-transfer": "hf_transfer", "huggingface-hub": "huggingface_hub", "prompt-toolkit": "prompt_toolkit"}
missing_packages = []
for package, import_name in REQUIRED_PACKAGES.items():
    try:
        importlib.import_module(import_name)
    except ImportError:
        missing_packages.append(package)
if missing_packages:
    print(f"❌ Missing packages: {', '.join(missing_packages)}. Please run: pip install {' '.join(missing_packages)}")
    sys.exit(1)

from huggingface_hub import HfApi, login, whoami, create_repo, repo_exists
from huggingface_hub.errors import HfHubHTTPError
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import PathCompleter

# === CHECK HF-TRANSFER STATUS ===
try:
    import hf_transfer
    print(f"[DEBUG] hf-transfer is installed: {hf_transfer.__version__}")
except ImportError:
    print("[DEBUG] ⚠️  hf-transfer NOT installed - will use slower requests library!")

def expand_paths(path_patterns):
    """Takes a list of path patterns and returns a flat list of existing files/folders."""
    expanded_paths = []
    for part in path_patterns:
        expanded_part = os.path.expanduser(part)
        matches = glob.glob(expanded_part)
        if matches:
            expanded_paths.extend(matches)
        elif os.path.exists(expanded_part):
            expanded_paths.append(expanded_part)
    return expanded_paths

def get_upload_paths_interactive():
    """Interactively prompts for local paths using advanced completion."""
    completer = PathCompleter()
    session = PromptSession(completer=completer)
    while True:
        try:
            input_str = session.prompt("\nEnter local path(s) to upload (use Tab, spaces, wildcards): ")
            if not input_str: continue
            expanded_paths = expand_paths(input_str.strip().split())
            if expanded_paths: return expanded_paths
            else: print(f"❌ No files or folders found matching: {input_str}")
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled."); return []

def main(token: str = None, repo_id: str = None, local_paths_args: list = None, dest_folder: str = None, 
         non_interactive: bool = False, create_if_needed: bool = False, is_private: bool = False, repo_type: str = 'model'):
    """Main function to handle authentication, repo selection/creation, and upload."""
    try:
        login(token=token, add_to_git_credential=False)
        username = whoami(token=token)['name']
        print(f"✅ Logged in as: {username}")
    except Exception as e:
        print(f"❌ Authentication failed: {e}"); return

    api = HfApi(token=token)
    selected_repo = repo_id

    # --- REPO SELECTION / CREATION ---
    # Non-interactive mode (used by the conversion script)
    if selected_repo and non_interactive:
        try:
            if not repo_exists(repo_id=selected_repo, repo_type=repo_type, token=token):
                if create_if_needed:
                    print(f" Repository '{selected_repo}' not found. Creating it now...")
                    create_repo(repo_id=selected_repo, private=is_private, repo_type=repo_type, token=token)
                    print(f"✅ Successfully created repository.")
                else:
                    print(f"❌ Repository '{selected_repo}' not found and creation was not requested."); return
        except Exception as e:
            print(f"❌ Error checking/creating repository: {e}"); return
    # Interactive mode (when running this script directly)
    elif not selected_repo:
        while not selected_repo:
            try:
                print("\nFetching your repositories...")
                repos = sorted([r.id for r in api.list_models(author=username)])
                if repos:
                    print("Please select a repository:")
                    for i, r_id in enumerate(repos): print(f"  {i + 1}. {r_id}")
                else: print("You have no model repositories.")
                print("\n  N. Create a new repository")
                choice = input("Enter number, 'N' to create, or 'Q' to quit: ").strip().upper()
                if choice == 'Q': return
                elif choice == 'N':
                    new_repo = input("Enter a name for the new repository: ").strip()
                    if not new_repo: print("❌ Repo name cannot be empty."); continue
                    if '/' not in new_repo: new_repo = f"{username}/{new_repo}"
                    private = input("Make this repo private? [y/n] (default: n): ").strip().lower() == 'y'
                    try:
                        print(f"Creating {'private' if private else 'public'} repo: '{new_repo}'...")
                        selected_repo = create_repo(repo_id=new_repo, private=private, repo_type='model').repo_id
                        print(f"✅ Successfully created '{selected_repo}'")
                    except HfHubHTTPError as e: print(f"❌ Error creating repository: {e}")
                else:
                    try: selected_repo = repos[int(choice) - 1]
                    except (ValueError, IndexError): print("❌ Invalid selection.")
            except Exception as e: print(f"❌ An error occurred: {e}"); return

    if not selected_repo:
        print("❌ No repository selected. Aborting upload."); return

    # --- PATH SELECTION ---
    local_paths = expand_paths(local_paths_args) if local_paths_args else get_upload_paths_interactive()
    if not local_paths:
        print("No local files provided for upload."); return
        
    if dest_folder is None and not non_interactive:
        dest_folder = input(f"\nEnter destination folder in '{selected_repo}' (press Enter for root): ").strip()

    # --- UPLOAD ---
    print("\n--- UPLOAD SUMMARY ---")
    print(f"  - Target repository:   '{selected_repo}'")
    print(f"  - Destination folder:  '{dest_folder or 'root'}'")
    for path in local_paths: print(f"    - {path}")
    print("----------------------")

    if not non_interactive and input("\nProceed with upload? (y/n): ").strip().lower() != 'y':
        print("\nUpload cancelled."); return

    # === UPLOAD USING HF UPLOAD (per-file with destination path) ===
    INTER_FILE_DELAY = 3  # seconds between file uploads to avoid rate limits
    
    for file_idx, path in enumerate(local_paths):
        item_name = os.path.basename(path.rstrip('/\\'))
        path_in_repo = f"{dest_folder}/{item_name}" if dest_folder else item_name
        
        try:
            mem_before = psutil.virtual_memory().percent
            if os.path.isfile(path):
                file_size = os.path.getsize(path) / (1024**3)
                print(f"\n[DEBUG] Pre-upload - Mem: {mem_before}%, File Size: {file_size:.2f}GB")
                print(f"Uploading FILE '{item_name}' to '{path_in_repo}'...")
                
                upload_start = time.time()
                cmd = [
                    "hf", "upload",
                    selected_repo,
                    path,
                    path_in_repo
                ]
                
                result = subprocess.run(cmd)
                upload_elapsed = time.time() - upload_start
                
                if result.returncode == 0:
                    actual_speed = (file_size * 1024) / upload_elapsed if upload_elapsed > 0 else 0
                    mem_after = psutil.virtual_memory().percent
                    print(f"[DEBUG] Post-upload - Mem: {mem_after}%, Actual Speed: {actual_speed:.2f}MB/s, Time: {upload_elapsed:.1f}s")
                    print(f"  ✅ Successfully uploaded {item_name}")
                else:
                    print(f"  ❌ FAILED to upload {item_name}.")
            elif os.path.isdir(path):
                print(f"\nUploading FOLDER '{item_name}' to '{path_in_repo}'...")
                cmd = [
                    "hf", "upload",
                    selected_repo,
                    path,
                    path_in_repo
                ]
                result = subprocess.run(cmd)
                if result.returncode == 0:
                    print(f"  ✅ Successfully uploaded {item_name}")
                else:
                    print(f"  ❌ FAILED to upload {item_name}.")
        except Exception as e:
            print(f"  ❌ Error uploading {item_name}: {e}")
        
        # Add delay between files to avoid rate limiting
        if file_idx < len(local_paths) - 1:
            print(f"  ⏳ Waiting {INTER_FILE_DELAY}s before next file...")
            time.sleep(INTER_FILE_DELAY)

    print(f"\n🚀 All operations complete! View your repository at: https://huggingface.co/{selected_repo}/tree/main")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload files/folders to your Hugging Face repo.")
    parser.add_argument("--token", help="Your Hugging Face API token.")
    parser.add_argument("--repo", help="Target repository ID (e.g., 'username/my-model').")
    parser.add_argument("--path", nargs='+', help="One or more local paths to upload.")
    parser.add_argument("--dest", help="Destination folder in the repository.")
    parser.add_argument("-y", "--yes", action='store_true', help="Automatically confirm upload.")
    parser.add_argument("--create", action='store_true', help="Create the repository if it does not exist.")
    parser.add_argument("--private", action='store_true', help="When creating a repo, make it private.")
    parser.add_argument("--repo-type", choices=['model', 'dataset', 'space'], default='model', help="Type of repo to create.")
    args = parser.parse_args()
    
    main(token=(args.token or os.getenv("HUGGING_FACE_HUB_TOKEN")), repo_id=args.repo, local_paths_args=args.path, 
         dest_folder=args.dest, non_interactive=args.yes, create_if_needed=args.create, 
         is_private=args.private, repo_type=args.repo_type)
