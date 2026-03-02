import os
import sys
import subprocess
import argparse
import time
from datetime import datetime

# --- ENVIRONMENT OVERRIDES (Matching your successful manual test) ---
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "0"
os.environ["HF_HUB_MAX_CONCURRENCY"] = "4"

def format_duration(seconds):
    hours, rem = divmod(int(seconds), 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours}h {minutes}m {seconds}s" if hours > 0 else f"{minutes}m {seconds}s"

def upload_to_hf():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str)
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--dest", type=str, default="")
    parser.add_argument("--path", nargs="+", required=True)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    token = args.token or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # Path Discovery + Symlink Resolution (to ensure we use physical E:\ paths)
    final_upload_list = []
    comp_suffixes = ["_vae.safetensors", "_clip_g.safetensors", "_clip_l.safetensors", "_clip.safetensors", "_model.safetensors"]
    for p in args.path:
        p = os.path.realpath(os.path.normpath(p))
        if os.path.exists(p):
            if p not in final_upload_list: final_upload_list.append(p)
            base_name = os.path.basename(p).split('-')[0].split('_v')[0] 
            search_dir = os.path.dirname(p)
            for suffix in comp_suffixes:
                candidate = os.path.realpath(os.path.join(search_dir, f"{base_name}{suffix}"))
                if os.path.exists(candidate) and candidate not in final_upload_list:
                    print(f"[AUTO-DETECT] Found matching component: {os.path.basename(candidate)}")
                    final_upload_list.append(candidate)
    
    final_upload_list = list(dict.fromkeys(final_upload_list))

    batch_start_time = time.time()
    for i, local_path in enumerate(final_upload_list):
        file_name = os.path.basename(local_path)
        dest_path = f"{args.dest.strip('/')}/{file_name}".replace("\\", "/")

        file_start_time = time.time()
        print(f"[{i+1}/{len(final_upload_list)}] ⏩ UPLOADING: {file_name}")
        print(f"      🕒 Start: {datetime.now().strftime('%H:%M:%S')}")

        # --- THE FIX: CONSTRUCT THE EXACT COMMAND STRING ---
        # We wrap paths in double quotes to handle spaces correctly.
        cmd_string = f'hf upload {args.repo} "{local_path}" "{dest_path}" --token {token}'
        cmd_string_dummy = f'hf upload {args.repo} "{local_path}" "{dest_path}" --token <TOKEN>'
        print(f"      🔧 CMD: {cmd_string_dummy}")
        
        try:
            # We use shell=True to make it behave exactly like your manual CMD entry
            # This allows 'hf' to handle its own progress bars and network sockets
            subprocess.run(cmd_string, shell=True, check=True)
            
            duration = format_duration(time.time() - file_start_time)
            print(f"      ✅ SUCCESS: {file_name}")
            print(f"      🕒 End:   {datetime.now().strftime('%H:%M:%S')}")
            print(f"      ⏱️  Took:  {duration}\n")
        except subprocess.CalledProcessError as e:
            print(f"      ❌ FAILED: {file_name} (CLI returned error)")
        except Exception as e:
            print(f"      ⚠️  Unexpected Error: {e}")

if __name__ == "__main__":
    upload_to_hf()