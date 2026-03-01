import os
import sys
import subprocess
import argparse
import time
from datetime import datetime

# --- THE ULTIMATE STABILITY OVERRIDES ---
os.environ["HF_HUB_DISABLE_XET_UPLOADS"] = "1"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "0"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_UPLOAD_TIMEOUT"] = "600"
os.environ["HF_HUB_COMMIT_TIMEOUT"] = "600"

def format_duration(seconds):
    """Formats seconds into H:M:S."""
    hours, rem = divmod(int(seconds), 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    return f"{minutes}m {seconds}s"

def upload_to_hf():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str)
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--dest", type=str, default="")
    parser.add_argument("--path", nargs="+", required=True)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    token = args.token or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # Path discovery logic (Find matching VAE/CLIPs)
    final_upload_list = []
    comp_suffixes = ["_vae.safetensors", "_clip_g.safetensors", "_clip_l.safetensors", "_clip.safetensors", "_model.safetensors"]
    for p in args.path:
        p = os.path.normpath(p)
        if os.path.exists(p):
            if p not in final_upload_list: final_upload_list.append(p)
            base_filename = os.path.basename(p)
            base_name = base_filename.split('-')[0].split('_v')[0] 
            search_dir = os.path.dirname(p)
            for suffix in comp_suffixes:
                candidate = os.path.join(search_dir, f"{base_name}{suffix}")
                if os.path.exists(candidate) and candidate not in final_upload_list:
                    final_upload_list.append(candidate)
    
    final_upload_list = list(dict.fromkeys(final_upload_list))

    # Binary check
    venv_bin_dir = os.path.dirname(sys.executable)
    hf_binary = os.path.join(venv_bin_dir, "hf")
    if not os.path.exists(hf_binary):
        hf_binary = os.path.join(venv_bin_dir, "hf.exe")
    if not os.path.exists(hf_binary):
        hf_binary = "hf"

    batch_start_time = time.time()
    print(f"\n============================================================")
    print(f"🚀 BATCH UPLOAD STARTED AT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📦 Total Files in Queue: {len(final_upload_list)}")
    print(f"⚙️  Stability Mode: S3 Standard (Xet-Deduplication Disabled)")
    print(f"============================================================\n")

    success_count = 0
    fail_count = 0

    for i, local_path in enumerate(final_upload_list):
        file_name = os.path.basename(local_path)
        clean_dest = args.dest.strip("/")
        path_in_repo = f"{clean_dest}/{file_name}" if clean_dest else file_name
        path_in_repo = path_in_repo.replace("\\", "/")

        file_start_time = time.time()
        timestamp_start = datetime.now().strftime('%H:%M:%S')
        
        print(f"[{i+1}/{len(final_upload_list)}] ⏩ STARTING: {file_name}")
        print(f"      🕒 Start Time: {timestamp_start}")
        print(f"      📍 Destination: {path_in_repo}")

        cmd = [
            hf_binary, "upload",
            args.repo,
            local_path,
            path_in_repo,
            "--token", token,
            "--type", "model"
        ]
        
        try:
            # subprocess.run mirrors the 'hf' progress bars to your GUI
            subprocess.run(cmd, check=True)
            
            file_end_time = time.time()
            duration = format_duration(file_end_time - file_start_time)
            print(f"      ✅ SUCCESS: {file_name}")
            print(f"      🕒 End Time: {datetime.now().strftime('%H:%M:%S')}")
            print(f"      ⏱️  Duration: {duration}\n")
            success_count += 1
            # Brief pause to let server handle the commit
            time.sleep(2) 
        except Exception as e:
            print(f"      ❌ FAILED: {file_name}")
            print(f"      ⚠️  Error: {e}\n")
            fail_count += 1

    batch_end_time = time.time()
    total_duration = format_duration(batch_end_time - batch_start_time)

    print(f"============================================================")
    print(f"🏁 BATCH COMPLETED AT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total Time Elapsed: {total_duration}")
    print(f"✅ Successfully Uploaded: {success_count}")
    print(f"❌ Failed: {fail_count}")
    print(f"============================================================\n")

if __name__ == "__main__":
    upload_to_hf()