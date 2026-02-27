import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk, Menu, Canvas, Toplevel
import os
import sys
import subprocess
import threading
import logging
import re
import glob
import json
import platform
import queue
import time
import shutil
import urllib.request
from datetime import datetime
import math
import importlib.util

global THEMES
THEMES = {
    "Default Light": {
        "bg": "#f0f0f0", "fg": "#000000", "frame_bg": "#f0f0f0", "label_fg": "blue",
        "btn_bg": "#e1e1e1", "btn_fg": "#000000", "entry_bg": "#ffffff", "entry_fg": "#000000",
        "log_bg": "#1e1e1e", "log_fg": "#d4d4d4", "action_bg": "#ddffdd"
    },
    "Visual Studio Dark": {
        "bg": "#1e1e1e", "fg": "#cccccc", "frame_bg": "#1e1e1e", "label_fg": "#569cd6",
        "btn_bg": "#333333", "btn_fg": "#cccccc", "entry_bg": "#2d2d2d", "entry_fg": "#9cdcfe",
        "log_bg": "#000000", "log_fg": "#dcdcdc", "action_bg": "#0e639c"
    },
    "OLED Night": {
        "bg": "#000000", "fg": "#ffffff", "frame_bg": "#000000", "label_fg": "#bb86fc",
        "btn_bg": "#121212", "btn_fg": "#ffffff", "entry_bg": "#1e1e1e", "entry_fg": "#03dac6",
        "log_bg": "#000000", "log_fg": "#ffffff", "action_bg": "#3700b3"
    }
}

if platform.system() == "Windows":
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.windll.kernel32
    hStdOut = kernel32.GetStdHandle(-11)
    mode = wintypes.DWORD()
    if kernel32.GetConsoleMode(hStdOut, ctypes.byref(mode)):
        # 0x0004 = ENABLE_VIRTUAL_TERMINAL_PROCESSING
        # 0x0008 = DISABLE_NEWLINE_AUTO_RETURN (Stops Windows from adding \n to \r)
        kernel32.SetConsoleMode(hStdOut, mode.value | 0x0004 | 0x0008)

# --- 0. AUTO-RESTART IN VENV ---
def check_and_restart_in_venv():
    possible_venvs = ["venv", ".venv", "env"]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_venv_path = None
    for venv_name in possible_venvs:
        full_path = os.path.join(script_dir, venv_name)
        if os.path.isdir(full_path):
            target_venv_path = full_path
            break
    if not target_venv_path: return
    if sys.platform == "win32":
        venv_python = os.path.join(target_venv_path, "Scripts", "python.exe")
    else:
        venv_python = os.path.join(target_venv_path, "bin", "python")
    if not os.path.exists(venv_python): return
    current_exe = os.path.normpath(sys.executable).lower()
    target_exe = os.path.normpath(venv_python).lower()
    if current_exe != target_exe:
        print(f"[INFO] Auto-relaunching in venv: {venv_python}")
        try:
            subprocess.call([venv_python] + sys.argv)
            sys.exit()
        except Exception as e:
            print(f"[ERROR] Restart failed: {e}")

check_and_restart_in_venv()

def ensure_dependencies():
    """Checks for required packages and installs them if missing."""
    dependencies = {
        "safetensors": "safetensors",
        "huggingface_hub": "huggingface_hub",
        "hf_transfer": "hf_transfer",
        "tqdm": "tqdm",
        "sentencepiece": "sentencepiece",
        "numpy==1.26.4": "numpy",
        "gguf": "gguf",
        "prompt_toolkit": "prompt_toolkit",
        "requests": "requests",
        "torch": "torch"
    }
    missing_or_wrong = []
    for pkg_pip, mod_name in dependencies.items():
        spec = importlib.util.find_spec(mod_name)
        if spec is None:
            missing_or_wrong.append(pkg_pip)
        else:
            if "numpy==1.26.4" in pkg_pip:
                try:
                    import numpy
                    if numpy.__version__ != "1.26.4":
                        missing_or_wrong.append(pkg_pip)
                except: missing_or_wrong.append(pkg_pip)
    if missing_or_wrong:
        print(f"[INFO] Installing dependencies: {', '.join(missing_or_wrong)}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_or_wrong)
        except Exception as e:
            print(f"[ERROR] Pip failed: {e}")

ensure_dependencies()

# --- 1. DEPENDENCY MANAGER ---
class DependencyManager:
    """Checks for, downloads, and compiles required tools."""
    SOURCES = {
        "dequantize_fp8v2.py": "https://raw.githubusercontent.com/Santodan/GGUF-Converter-GUI/refs/heads/main/dequantize_fp8v2.py",
        "convert.py": "https://raw.githubusercontent.com/city96/ComfyUI-GGUF/refs/heads/auto_convert/tools/convert.py",
        "extract_components.py": "https://raw.githubusercontent.com/Santodan/GGUF-Converter-GUI/refs/heads/main/extract_components.py",
        "lcpp.patch": "https://raw.githubusercontent.com/city96/ComfyUI-GGUF/refs/heads/auto_convert/tools/lcpp.patch",
        "fix_5d_tensors.py": "https://raw.githubusercontent.com/city96/ComfyUI-GGUF/refs/heads/auto_convert/tools/fix_5d_tensors.py",
        "upload_to_hf.py": "https://raw.githubusercontent.com/Santodan/GGUF-Converter-GUI/refs/heads/main/upload_to_hf.py"
    }

    @staticmethod
    def check_and_setup(logger_callback):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for tool, url in DependencyManager.SOURCES.items():
            path = os.path.join(script_dir, tool)
            if not os.path.exists(path):
                logger_callback(f"[SETUP] Downloading {tool}...")
                try: urllib.request.urlretrieve(url, path)
                except Exception as e: logger_callback(f"[ERROR] Download failed: {e}")

        binary_name = "llama-quantize.exe" if platform.system() == "Windows" else "llama-quantize"
        if not os.path.exists(os.path.join(script_dir, binary_name)):
            if messagebox.askyesno("Setup", f"{binary_name} not found. Build it?"):
                threading.Thread(target=DependencyManager.build_llama_cpp, args=(script_dir, logger_callback), daemon=True).start()

    @staticmethod
    def build_llama_cpp(target_dir, logger_callback):
        original_cwd = os.getcwd()
        temp_build_dir = os.path.join(target_dir, "llama_cpp_build_temp")
        
        try:
            if os.path.exists(temp_build_dir):
                shutil.rmtree(temp_build_dir)
            os.makedirs(temp_build_dir)
            
            patch_path = os.path.join(temp_build_dir, "lcpp.patch")
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/city96/ComfyUI-GGUF/refs/heads/auto_convert/tools/lcpp.patch",
                patch_path
            )
            
            os.chdir(temp_build_dir)
            subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", "source"], check=True)
            os.chdir("source")
            
            # Use the exact tag your original code had
            subprocess.run(["git", "checkout", "tags/b3962"], check=True)
            subprocess.run(["git", "apply", patch_path], check=True)
            
            # ── Apply the exact log.cpp patch you want ───────────────────────────────
            log_cpp = os.path.join("common", "log.cpp")
            if os.path.exists(log_cpp):
                with open(log_cpp, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                with open(log_cpp, 'w', encoding='utf-8') as f:
                    inserted = False
                    for line in lines:
                        f.write(line)
                        if '#include "log.h"' in line and not inserted:
                            f.write('\n#define _SILENCE_CXX23_CHRONO_DEPRECATION_WARNING\n')
                            f.write('#include <chrono>\n')
                            inserted = True
            else:
                logger_callback("[WARNING] common/log.cpp not found — patch skipped")
            
            # ── CMake configure ──────────────────────────────────────────────────────
            # Platform-specific flags
            if platform.system() == "Windows":
                cmake_configure = [
                    "cmake", "-B", "build",
                    "-DCMAKE_CXX_STANDARD=17",
                    "-DCMAKE_CXX_STANDARD_REQUIRED=ON",
                    '-DCMAKE_CXX_FLAGS="-std=c++17"',
                    "-A", "x64"   # still needed for VS generator
                ]
            else:  # Linux / others
                cmake_configure = [
                    "cmake", "-B", "build"
                ]
            
            logger_callback("[BUILD] Running cmake configure...")
            subprocess.run(cmake_configure, check=True)
            
            # ── Build Debug ──────────────────────────────────────────────────────────
            jobs = "10" if platform.system() == "Windows" else str(os.cpu_count() or 4)
            if platform.system() == "Windows":
                build_cmd = [
                    "cmake", "--build", "build",
                    "--config", "Debug",
                    "-j", jobs,
                    "--target", "llama-quantize"
                ]
            else:  # Linux / others
                build_cmd = [
                    "cmake", "--build", "build",
                    "-j", jobs,
                    "--target", "llama-quantize"
                ]  # No --config on Linux
            
            logger_callback("[BUILD] Compiling llama-quantize (Debug)...")
            subprocess.run(build_cmd, check=True)
            
            # ── Locate and copy binary + DLLs ────────────────────────────────────────
            possible_bin_paths = []
            if platform.system() == "Windows":
                possible_bin_paths = [
                    os.path.join("build", "bin", "Debug", "llama-quantize.exe"),
                    os.path.join("build", "Debug", "llama-quantize.exe"),
                ]
                dll_patterns = ["ggml*.dll", "llama*.dll"]
            else:
                possible_bin_paths = [
                    os.path.join("build", "bin", "llama-quantize"),
                    os.path.join("build", "llama-quantize"),
                ]
                dll_patterns = ["libggml*.so*", "libllama*.so*"]
            
            built_binary = None
            for path in possible_bin_paths:
                if os.path.exists(path):
                    built_binary = path
                    break
            
            if built_binary:
                dest_name = "llama-quantize.exe" if platform.system() == "Windows" else "llama-quantize"
                dest_path = os.path.join(target_dir, dest_name)
                shutil.copy2(built_binary, dest_path)
                logger_callback(f"[BUILD] Copied binary → {dest_path}")
                
                # Copy nearby shared libs (platform-specific patterns)
                bin_dir = os.path.dirname(built_binary)
                for pattern in dll_patterns:
                    for lib in glob.glob(os.path.join(bin_dir, pattern)):
                        shutil.copy2(lib, target_dir)
                        logger_callback(f"[BUILD] Copied dependency: {os.path.basename(lib)}")
                
                messagebox.showinfo("Success", "llama-quantize (Debug) built and copied successfully")
            else:
                logger_callback("[ERROR] Could not find built llama-quantize binary in expected locations")
                messagebox.showerror("Build Failed", "Binary not found after compilation")

        except Exception as e:
            logger_callback(f"[ERROR] Build process failed: {e}")
            messagebox.showerror("Build Error", f"An error occurred during the build:\n{e}")
        finally:
            os.chdir(original_cwd)

# --- UPLOADER LOADER ---
UPLOADER_AVAILABLE = False; uploader = None
def try_load_uploader():
    global uploader, UPLOADER_AVAILABLE
    if os.path.exists("upload_to_hf.py"):
        try:
            import upload_to_hf; import importlib; importlib.reload(upload_to_hf)
            uploader, UPLOADER_AVAILABLE = upload_to_hf, True; return True
        except: pass
    return False
try_load_uploader()

try:
    import torch; from safetensors.torch import save_file, load_file
    TORCH_AVAILABLE = True
except ImportError: TORCH_AVAILABLE = False

# --- CONFIG ---
QUANT_GROUPS = [
    ["F16", "BF16"], ["Q2_K"], ["Q3_K_S", "Q3_K_M", "Q3_K_L"],
    ["Q4_0", "Q4_K_S", "Q4_K_M"], ["Q5_0", "Q5_K_S", "Q5_K_M"],
    ["Q6_K", "Q8_0"], ["FP8_E5M2", "FP8_E5M2 (All)"], ["FP8_E4M3FN", "FP8_E4M3FN (All)"],
    ["MODEL", "VAE", "CLIP"],
]

if TORCH_AVAILABLE:
    class FP8Quantizer:
        def __init__(self, quant_dtype: str = "float8_e5m2"):
            self.quant_dtype = quant_dtype

        def quantize_weights(self, weight: torch.Tensor) -> torch.Tensor:
            if not weight.is_floating_point(): return weight
            dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            w = weight.to(dev)
            mx = torch.max(torch.abs(w))
            if mx == 0: return torch.zeros_like(w, dtype=getattr(torch, self.quant_dtype))
            scale = torch.max(mx / 127.0, torch.tensor(1e-12, device=dev, dtype=w.dtype))
            q = torch.round(w / scale * 127.0) / 127.0 * scale
            return q.to(dtype=getattr(torch, self.quant_dtype))

        def apply_quantization_to_file(self, src_path, dst_path, unet_only=True, check_stop_func=None):
            state_dict = load_file(src_path) if src_path.endswith(".safetensors") else torch.load(src_path, map_location="cpu")
            quantized_dict = {}
            total = len(state_dict)
            
            # Key model identifiers for Flux, Wan, SD3, and SDXL
            model_keys = ["model.diffusion_model", "transformer.single_blocks", "transformer.double_blocks", "model.transformers"]

            for i, (name, param) in enumerate(state_dict.items()):
                if check_stop_func and check_stop_func(): return False
                
                if i % 10 == 0 or i == total - 1:
                    percent = (i + 1) / total * 100
                    sys.stdout.write(f"\r[FP8 Progress] {percent:3.1f}% | Tensor {i+1}/{total}")
                    sys.stdout.flush()

                is_model_weight = any(k in name for k in model_keys)

                if unet_only and not is_model_weight:
                    quantized_dict[name] = param
                elif isinstance(param, torch.Tensor) and param.is_floating_point():
                    quantized_dict[name] = self.quantize_weights(param)
                else:
                    quantized_dict[name] = param
            
            print("") 
            save_file(quantized_dict, dst_path)
            return True
else:
    class FP8Quantizer:
        pass

# --- GUI UTILS ---
class DualOutput:
    def __init__(self, original_stream, msg_queue, log_file_handle):
        self.original_stream = original_stream
        self.msg_queue = msg_queue
        self.log_file_handle = log_file_handle

    def write(self, message):
        if not message: return
        # 1. Write to the hidden black console (Standard behavior)
        self.original_stream.write(message)
        self.original_stream.flush()

        # 2. Write RAW to the Log File (keeps the disk log clean)
        if self.log_file_handle:
            self.log_file_handle.write(message)
            self.log_file_handle.flush()

        # 3. Send to the GUI Queue
        self.msg_queue.put(("RAW", message))

    def flush(self):
        self.original_stream.flush()

class ProgressPopup(tk.Toplevel):
    def __init__(self, app_instance):
        super().__init__(app_instance.root)
        self.app = app_instance 
        self.title("Job Progress Status")
        self.geometry("700x500")
        self.protocol("WM_DELETE_WINDOW", self.hide_window)
        
        # Get theme shortcut
        t = self.app.current_theme
        self.configure(bg=t["bg"]) # Set window background
        
        # 1. Initialize Canvas with theme background
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, bg=t["bg"])
        
        # 2. Scrollbars (Note: ttk scrollbars have limited theming, but we place them on themed bg)
        self.scroll_y = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scroll_x = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        
        # 3. Inner Frame with theme background
        self.inner = tk.Frame(self.canvas, bg=t["bg"])
        
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw", tags="inner_win")
        self.canvas.configure(yscrollcommand=self.scroll_y.set, xscrollcommand=self.scroll_x.set)
        
        self.scroll_y.pack(side="right", fill="y")
        self.scroll_x.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.cells = {} 

    def hide_window(self):
        self.withdraw()

    def setup_grid(self, models, steps):
        for w in self.inner.winfo_children(): w.destroy()
        self.cells = {}
        t = self.app.current_theme
        
        # Set the inner frame background explicitly again to be safe
        self.inner.configure(bg=t["bg"])
        
        # Header Row
        tk.Label(self.inner, text="Model Name", font=("Arial", 9, "bold"), 
                 bg=t["btn_bg"], fg=t["fg"], width=30, anchor="w", relief="flat").grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        for i, step in enumerate(steps):
            tk.Label(self.inner, text=step, font=("Arial", 8, "bold"), 
                     bg=t["btn_bg"], fg=t["fg"], width=12, relief="flat").grid(row=0, column=i+1, sticky="nsew", padx=1, pady=1)
        
        # Data Rows
        for r, model in enumerate(models):
            row = r + 1
            tk.Label(self.inner, text=model[:40], anchor="w", bg=t["frame_bg"], fg=t["fg"]).grid(row=row, column=0, sticky="nsew", padx=1, pady=1)
            for c, step in enumerate(steps):
                # Use theme bg for the empty "..." state
                lbl = tk.Label(self.inner, text="...", bg=t["bg"], fg=t["fg"], width=12)
                lbl.grid(row=row, column=c+1, sticky="nsew", padx=1, pady=1)
                self.cells[(model, step)] = lbl

    def update_status(self, model, step, status):
        if (model, step) not in self.cells: return
        lbl = self.cells[(model, step)]
        t = self.app.current_theme
        is_dark = t["bg"] != "#f0f0f0"
        
        if status == "RUNNING":
            lbl.config(bg="#ffff99" if not is_dark else "#5c5c00", text="Running", fg="black" if not is_dark else "white")
        elif status == "DONE":
            lbl.config(bg="#99ff99" if not is_dark else "#004400", text="Done", fg="black" if not is_dark else "white")
        elif status == "ERROR":
            lbl.config(bg="#ff9999" if not is_dark else "#440000", text="Error", fg="black" if not is_dark else "white")
        elif status == "SKIP":
            lbl.config(bg=t["btn_bg"], text="-", fg=t["fg"])
        elif status == "CANCEL":
            lbl.config(bg="#ffcc00" if not is_dark else "#886600", text="Cancel", fg="black" if not is_dark else "white")
        else:
            lbl.config(bg=t["bg"], text="...", fg=t["fg"])

# --- MAIN APP ---
class ConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GGUF & FP8 Manager")
        self.root.geometry("1300x850")
        
        # 1. Initialize Theme first
        self.current_theme = THEMES["Default Light"]
        self.gui_ansi_buffer = ""

        # 2. Initialize ALL Logic Variables (Prevents the AttributeError)
        self.python_path_var = tk.StringVar(value=os.path.normpath(sys.executable))
        self.out_dir_var = tk.StringVar()
        self.out_mode_var = tk.StringVar(value="folder")
        self.upload_mode_var = tk.StringVar(value="global")
        self.do_upload = tk.BooleanVar(value=False)
        self.hf_token = tk.StringVar(value=os.getenv("HUGGING_FACE_HUB_TOKEN",""))
        self.hf_repo_gguf = tk.StringVar()
        self.hf_repo_fp8 = tk.StringVar()
        self.hf_dest_gguf = tk.StringVar()
        self.hf_dest_fp8 = tk.StringVar()
        self.cleanup_mode = tk.StringVar(value="per_model")
        self.shutdown_var = tk.BooleanVar(value=False)
        self.keep_dequant_var = tk.BooleanVar(value=False)
        self.keep_convert_var = tk.BooleanVar(value=False)
        
        # Grid Storage
        self.quant_vars_gen = {}
        self.quant_vars_up = {}
        self.quant_vars_keep = {}

        # Process State
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.settings_file = os.path.join(script_dir, "last_run_settings.json")
        self.msg_queue = queue.Queue()
        self.source_files = []
        self.custom_file_data = {} 
        self.is_running = False
        self.current_process = None
        self.stop_requested = False
        self.progress_window = None
        self.quant_cmd = self.get_quantize_command()

        # 3. Build UI and Load settings
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._setup_ui()
        self._setup_logging()
        DependencyManager.check_and_setup(logging.info)
        self.root.after(100, self.process_queue)
        self.load_settings(self.settings_file, silent=True)

    def get_quantize_command(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        name = "llama-quantize.exe" if platform.system() == "Windows" else "llama-quantize"
        local_path = os.path.join(script_dir, name)
        return local_path if os.path.exists(local_path) else name

    def _setup_logging(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers(): self.logger.handlers.clear()
        
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
        
        # Console Handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # File Handler
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.current_log_path = f"logs/log_{ts}.log"
        fh = logging.FileHandler(self.current_log_path, encoding='utf-8')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # REMOVED TextHandler here because DualOutput now handles the UI directly.

    def apply_theme(self, theme_data=None):
        if theme_data:
            self.current_theme = theme_data
        
        t = self.current_theme
        self.root.configure(bg=t["bg"])
        
        def walk_and_paint(parent):
            for child in parent.winfo_children():
                try:
                    # 1. Containers
                    if isinstance(child, (tk.Frame, tk.LabelFrame, tk.PanedWindow, tk.Canvas)):
                        # If it's a Canvas, we MUST set the background or we get grey gaps
                        bg_color = t["frame_bg"] if isinstance(child, tk.LabelFrame) else t["bg"]
                        child.configure(bg=bg_color)
                        
                        if isinstance(child, tk.LabelFrame):
                            child.configure(fg=t["label_fg"])
                        if isinstance(child, tk.Canvas):
                            child.configure(highlightthickness=0, borderwidth=0)
                        
                        walk_and_paint(child)
                    
                    elif isinstance(child, tk.Label):
                        # GGUF Labels match their container
                        p_bg = child.master.cget("bg")
                        child.configure(bg=p_bg, fg=t["fg"])
                    
                    # 3. Labels
                    elif isinstance(child, tk.Label):
                        p_bg = child.master.cget("bg")
                        child.configure(bg=child.master.cget("bg"), fg=t["fg"])
                    
                    # 4. Inputs
                    elif isinstance(child, tk.Entry):
                        child.configure(bg=t["entry_bg"], fg=t["entry_fg"], insertbackground=t["entry_fg"])
                    elif isinstance(child, tk.Listbox):
                        child.configure(bg=t["entry_bg"], fg=t["entry_fg"], selectbackground=t["label_fg"])
                    
                    # 5. Checks/Radios
                    elif isinstance(child, (tk.Checkbutton, tk.Radiobutton)):
                        p_bg = child.master.cget("bg")
                        child.configure(bg=p_bg, fg=t["fg"], selectcolor=t["entry_bg"], 
                                        activebackground=p_bg, activeforeground=t["fg"])
                    
                    # 6. Text/Logs
                    elif isinstance(child, (tk.Text, scrolledtext.ScrolledText)):
                        child.configure(bg=t["log_bg"], fg=t["log_fg"], insertbackground=t["log_fg"])
                except:
                    pass

        walk_and_paint(self.root)
        
        # FIX FOR POPUP: Re-paint the progress window containers
        if self.progress_window and self.progress_window.winfo_exists():
            self.progress_window.configure(bg=t["bg"])
            # Specifically target the Canvas and Inner frame
            self.progress_window.canvas.configure(bg=t["bg"])
            self.progress_window.inner.configure(bg=t["bg"])
            walk_and_paint(self.progress_window)

    def open_theme_menu(self):
        from tkinter import colorchooser
        top = tk.Toplevel(self.root)
        top.title("Theme Settings")
        top.geometry("350x450")
        
        main_f = tk.Frame(top, padx=20, pady=20)
        main_f.pack(fill="both", expand=True)

        tk.Label(main_f, text="Select Preset:", font="Arial 10 bold").pack(pady=(0, 10))
        
        # Preset Buttons
        for name, data in THEMES.items():
            tk.Button(main_f, text=name, width=25, 
                      command=lambda d=data: self.apply_theme(d)).pack(pady=2)

        tk.Label(main_f, text="Custom Color Pickers:", font="Arial 10 bold").pack(pady=(20, 10))
        
        # Add a way to pick a custom background
        def pick_bg():
            color = colorchooser.askcolor(title="Choose Main BG")[1]
            if color:
                new_t = self.current_theme.copy()
                new_t["bg"] = color
                new_t["frame_bg"] = color
                self.apply_theme(new_t)

        tk.Button(main_f, text="Pick Custom Background", command=pick_bg).pack(fill="x", pady=2)
        
        tk.Button(main_f, text="DONE", bg="#ddffdd", command=top.destroy).pack(pady=20)

    def _setup_ui(self):
        c = self.current_theme
        
        main_pane = tk.PanedWindow(self.root, orient=tk.VERTICAL, sashwidth=5, bg=c["bg"], bd=0)
        main_pane.pack(fill="both", expand=True)
        
        config_frame = tk.Frame(main_pane, bg=c["bg"])
        canvas = Canvas(config_frame, highlightthickness=0, borderwidth=0, bg=c["bg"])
        scrollbar = ttk.Scrollbar(config_frame, orient="vertical", command=canvas.yview)
        self.content_frame = tk.Frame(canvas, bg=c["bg"])

        def update_scrollregion(event):
            canvas.update_idletasks()
            bbox = canvas.bbox("all")
            canvas_height = canvas.winfo_height()
            if bbox[3] < canvas_height:
                canvas.configure(scrollregion=(0, 0, bbox[2], canvas_height))
            else:
                canvas.configure(scrollregion=bbox)

        def sync_width(event):
            canvas.itemconfig("inner", width=event.width)

        self.content_frame.bind("<Configure>", update_scrollregion)
        canvas.bind("<Configure>", sync_width)
        
        def on_mousewheel(event):
            if platform.system() == 'Windows': canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            else: canvas.yview_scroll(int(-1*event.delta), "units")
        
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        canvas.create_window((0, 0), window=self.content_frame, anchor="nw", tags="inner")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        main_pane.add(config_frame, height=750)

        log_container = tk.Frame(main_pane, bg=c["bg"])
        main_pane.add(log_container, minsize=150)
        
        self.log_display = scrolledtext.ScrolledText(
            log_container, height=12, font=("Consolas", 10), wrap='none',
            bg=c["log_bg"], fg=c["log_fg"], insertbackground=c["log_fg"],
            padx=5, pady=5, undo=False
        )
        self.log_display.pack(side="left", fill="both", expand=True)
        
        # Terminal Mirror State
        self.terminal_buffer = [""] * 100 
        self.term_cursor_line = 0 

        btn_clear = tk.Button(log_container, text="CLEAR\nLOGS", bg=c["btn_bg"], fg=c["btn_fg"], command=self.clear_logs)
        btn_clear.pack(side="right", fill="y", padx=2)

        # 1. Environment
        f_env = tk.LabelFrame(self.content_frame, text="1. Environment", padx=5, pady=5, bg=c["frame_bg"], fg=c["label_fg"])
        f_env.pack(fill="x", padx=5, pady=5)
        tk.Entry(f_env, textvariable=self.python_path_var, bg=c["entry_bg"], fg=c["entry_fg"], insertbackground=c["entry_fg"]).pack(side="left", fill="x", expand=True, padx=(0,5))
        tk.Button(f_env, text="Browse...", bg=c["btn_bg"], fg=c["btn_fg"], command=self.browse_python).pack(side="left", padx=2)
        tk.Button(f_env, text="Restart", bg=c["btn_bg"], fg=c["btn_fg"], command=self.restart).pack(side="left", padx=2)

        # 2. Output
        f_mode = tk.LabelFrame(self.content_frame, text="2. Local Output Configuration", padx=5, pady=5, bg=c["frame_bg"], fg=c["label_fg"])
        f_mode.pack(fill="x", padx=5, pady=5)
        tk.Label(f_mode, text="Output Strategy:", bg=c["frame_bg"], fg=c["fg"]).pack(anchor="w")
        modes_frame = tk.Frame(f_mode, bg=c["frame_bg"])
        modes_frame.pack(fill="x")
        for txt, val in [("Folder per Model", "folder"), ("All in One Folder (Flat)", "flat"), ("Custom Output Path", "custom")]:
            tk.Radiobutton(modes_frame, text=txt, variable=self.out_mode_var, value=val, bg=c["frame_bg"], fg=c["fg"], selectcolor=c["entry_bg"], command=self.refresh_file_list_ui).pack(side="left")

        self.global_out_frame = tk.Frame(self.content_frame, bg=c["bg"])
        tk.Label(self.global_out_frame, text="Base Output Dir:", bg=c["bg"], fg=c["fg"]).pack(side="left")
        tk.Entry(self.global_out_frame, textvariable=self.out_dir_var, bg=c["entry_bg"], fg=c["entry_fg"], insertbackground=c["entry_fg"]).pack(side="left", fill="x", expand=True, padx=5)
        tk.Button(self.global_out_frame, text="Browse", bg=c["btn_bg"], fg=c["btn_fg"], command=self.browse_out).pack(side="left")
        self.global_out_frame.pack(fill="x", padx=10, pady=5) 

        # 3. Files
        self.f_files_container = tk.LabelFrame(self.content_frame, text="3. Input Files & Routing Table", padx=5, pady=5, bg=c["frame_bg"], fg=c["label_fg"])
        self.f_files_container.pack(fill="x", padx=5, pady=5)
        btn_box = tk.Frame(self.f_files_container, bg=c["frame_bg"])
        btn_box.pack(fill="x", pady=2)
        tk.Button(btn_box, text="Add Files...", bg=c["btn_bg"], fg=c["btn_fg"], command=self.add_files).pack(side="left", fill="x", expand=True)
        tk.Button(btn_box, text="Remove Selected", bg=c["btn_bg"], fg=c["btn_fg"], command=self.remove_selected_files).pack(side="left", padx=5)
        tk.Button(btn_box, text="Clear List", bg=c["btn_bg"], fg=c["btn_fg"], command=self.clear_files).pack(side="left", padx=5)
        
        self.simple_list_frame = tk.Frame(self.f_files_container, bg=c["frame_bg"])
        self.file_listbox = tk.Listbox(self.simple_list_frame, height=6, selectmode=tk.EXTENDED, bg=c["entry_bg"], fg=c["entry_fg"])
        self.file_listbox.pack(side="left", fill="x", expand=True)
        self.simple_list_frame.pack(fill="x", expand=True) 
        self.local_custom_frame = tk.Frame(self.f_files_container, bg=c["frame_bg"])

        # 4. Quants
        f_quant = tk.LabelFrame(self.content_frame, text="4. Quantization", padx=5, pady=5, bg=c["frame_bg"], fg=c["label_fg"])
        f_quant.pack(fill="x", padx=5, pady=5)
        if not TORCH_AVAILABLE: tk.Label(f_quant, text="⚠️ Torch missing. FP8 disabled.", bg=c["frame_bg"], fg="red").grid(row=0, column=0, columnspan=10)
        for col_idx, group in enumerate(QUANT_GROUPS):
            base_col = col_idx * 5  
            tk.Label(f_quant, text="Type", font="Arial 8 bold", bg=c["frame_bg"], fg=c["fg"]).grid(row=1, column=base_col, sticky="w")
            tk.Label(f_quant, text="G", font="Arial 8 bold", bg=c["frame_bg"], fg=c["fg"], width=2).grid(row=1, column=base_col+1)
            tk.Label(f_quant, text="U", font="Arial 8 bold", bg=c["frame_bg"], fg=c["fg"], width=2).grid(row=1, column=base_col+2)
            tk.Label(f_quant, text="K", font="Arial 8 bold", bg=c["frame_bg"], fg=c["fg"], width=2).grid(row=1, column=base_col+3)
            tk.Label(f_quant, text="|", bg=c["frame_bg"], fg=c["btn_bg"]).grid(row=1, column=base_col+4, rowspan=10, sticky="ns")
            for i, q in enumerate(group):
                row = i + 2
                tk.Label(f_quant, text=q, bg=c["frame_bg"], fg=c["fg"]).grid(row=row, column=base_col, sticky="w")
                vg, vu, vk = tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar()
                self.quant_vars_gen[q], self.quant_vars_up[q], self.quant_vars_keep[q] = vg, vu, vk
                state = "normal"
                if "FP8" in q and not TORCH_AVAILABLE: state = "disabled"
                def sync(g=vg, u=vu, k=vk): 
                    if g.get(): u.set(True); k.set(True)
                tk.Checkbutton(f_quant, variable=vg, command=sync, state=state, bg=c["frame_bg"], selectcolor=c["entry_bg"]).grid(row=row, column=base_col+1)
                tk.Checkbutton(f_quant, variable=vu, state=state, bg=c["frame_bg"], selectcolor=c["entry_bg"]).grid(row=row, column=base_col+2)
                tk.Checkbutton(f_quant, variable=vk, state=state, bg=c["frame_bg"], selectcolor=c["entry_bg"]).grid(row=row, column=base_col+3)

        # --- 5. Global Settings & Upload ---
        f_sets = tk.LabelFrame(self.content_frame, text="5. Global Settings & Upload", padx=10, pady=10, bg=c["frame_bg"], fg=c["label_fg"])
        f_sets.pack(fill="x", padx=5, pady=5)
        f_sets.columnconfigure(2, weight=1) # Allow Token entry to stretch

        self.do_upload = tk.BooleanVar()
        tk.Checkbutton(f_sets, text="Enable Upload", variable=self.do_upload, 
                       bg=c["frame_bg"], fg=c["fg"], selectcolor=c["entry_bg"]).grid(row=0, column=0, sticky="w")
        
        tk.Label(f_sets, text="Token:", bg=c["frame_bg"], fg=c["fg"]).grid(row=0, column=1, sticky="e", padx=(10, 2))
        
        tk.Entry(f_sets, textvariable=self.hf_token, show="*", 
                 bg=c["entry_bg"], fg=c["entry_fg"], insertbackground=c["entry_fg"]).grid(row=0, column=2, columnspan=3, sticky="ew")
        
        ttk.Separator(f_sets, orient="horizontal").grid(row=1, column=0, columnspan=6, sticky="ew", pady=10)
        
        tk.Label(f_sets, text="Upload Strategy:", bg=c["frame_bg"], fg=c["fg"]).grid(row=2, column=0, sticky="e")
        
        strat_frame = tk.Frame(f_sets, bg=c["frame_bg"])
        strat_frame.grid(row=2, column=1, columnspan=4, sticky="w")
        
        tk.Radiobutton(strat_frame, text="Use Global Repos", variable=self.upload_mode_var, value="global", 
                       bg=c["frame_bg"], fg=c["fg"], selectcolor=c["entry_bg"], command=self.refresh_upload_ui).pack(side="left", padx=5)
        tk.Radiobutton(strat_frame, text="Custom per File", variable=self.upload_mode_var, value="custom", 
                       bg=c["frame_bg"], fg=c["fg"], selectcolor=c["entry_bg"], command=self.refresh_upload_ui).pack(side="left", padx=5)
        
        # This container holds either the global grid or the custom table
        self.upload_container = tk.Frame(f_sets, bg=c["frame_bg"])
        self.upload_container.grid(row=3, column=0, columnspan=6, sticky="ew", pady=5)
        self.upload_container.columnconfigure(0, weight=1)

        # 5a. Global Upload Sub-Frame
        self.global_upload_frame = tk.Frame(self.upload_container, bg=c["frame_bg"])
        self.global_upload_frame.grid(row=0, column=0, sticky="ew")
        
        # Grid layout for global entries
        labels = [("GGUF Repo:", self.hf_repo_gguf), ("FP8 Repo:", self.hf_repo_fp8), 
                  ("GGUF Folder:", self.hf_dest_gguf), ("FP8 Folder:", self.hf_dest_fp8)]
        
        for i, (txt, var) in enumerate(labels):
            r, col = divmod(i, 2)
            tk.Label(self.global_upload_frame, text=txt, bg=c["frame_bg"], fg=c["fg"]).grid(row=r, column=col*2, sticky="e", pady=2)
            tk.Entry(self.global_upload_frame, textvariable=var, width=30,
                     bg=c["entry_bg"], fg=c["entry_fg"], insertbackground=c["entry_fg"]).grid(row=r, column=col*2+1, sticky="ew", padx=5, pady=2)
        
        self.global_upload_frame.columnconfigure((1, 3), weight=1)

        # 5b. Custom Upload Sub-Frame
        self.custom_upload_frame = tk.Frame(self.upload_container, bg=c["frame_bg"])
        # (This frame is packed/hidden by refresh_upload_ui)

        # 5c. Footer (Cleanup)
        self.footer_frame = tk.Frame(f_sets, bg=c["frame_bg"])
        self.footer_frame.grid(row=4, column=0, columnspan=6, sticky="ew", pady=(10, 0))
        ttk.Separator(self.footer_frame, orient="horizontal").pack(fill="x", pady=5)
        
        f_c = tk.Frame(self.footer_frame, bg=c["frame_bg"])
        f_c.pack(fill="x")
        tk.Label(f_c, text="Cleanup Strategy:", bg=c["frame_bg"], fg=c["fg"]).pack(side="left")
        tk.Radiobutton(f_c, text="After Each", variable=self.cleanup_mode, value="per_model", bg=c["frame_bg"], fg=c["fg"], selectcolor=c["entry_bg"]).pack(side="left", padx=5)
        tk.Radiobutton(f_c, text="After All", variable=self.cleanup_mode, value="all_end", bg=c["frame_bg"], fg=c["fg"], selectcolor=c["entry_bg"]).pack(side="left", padx=5)
        
        tk.Checkbutton(f_c, text="Keep Dequant", variable=self.keep_dequant_var, bg=c["frame_bg"], fg=c["fg"], selectcolor=c["entry_bg"]).pack(side="left", padx=10)
        tk.Checkbutton(f_c, text="Keep Source", variable=self.keep_convert_var, bg=c["frame_bg"], fg=c["fg"], selectcolor=c["entry_bg"]).pack(side="left")

        # 6. Actions
        f_act = tk.Frame(self.content_frame, bg=c["bg"])
        f_act.pack(fill="x", padx=5, pady=(10, 0))
        tk.Button(f_act, text="THEME", width=10, bg=c["btn_bg"], fg=c["btn_fg"], command=self.open_theme_menu).pack(side="left", padx=5)
        tk.Checkbutton(f_act, text="Shutdown", variable=self.shutdown_var, bg=c["bg"], fg=c["fg"], selectcolor=c["entry_bg"]).pack(side="left")
        tk.Button(f_act, text="SHOW STATUS", bg=c["btn_bg"], fg=c["btn_fg"], command=self.show_progress_popup).pack(side="left", padx=20)
        tk.Button(f_act, text="CANCEL", bg="#662222" if c["bg"] != "#f0f0f0" else "#ffcccc", fg=c["btn_fg"], command=self.cancel_processing).pack(side="right")
        self.btn_run = tk.Button(f_act, text="START PROCESSING", bg=c["action_bg"], 
                                 fg="white" if c["bg"] != "#f0f0f0" else "black", height=2, command=self.start_thread)
        self.btn_run.pack(side="right", fill="x", expand=True, padx=5)
        
        self.root.after(10, self.apply_theme)

    def clear_logs(self):
        self.log_display.configure(state='normal')
        self.log_display.delete("1.0", tk.END)
        self.log_display.configure(state='disabled')

    def refresh_file_list_ui(self):
        out_mode = self.out_mode_var.get()
        if out_mode == "custom": self.global_out_frame.pack_forget()
        else: self.global_out_frame.pack(fill="x", padx=10, pady=5, before=self.f_files_container)
        if out_mode == "custom":
            self.simple_list_frame.pack_forget()
            self.build_local_table()
            self.local_custom_frame.pack(fill="x", expand=True)
        else:
            self.local_custom_frame.pack_forget()
            self.simple_list_frame.pack(fill="x", expand=True)
            self.file_listbox.delete(0, tk.END)
            for f in self.source_files: self.file_listbox.insert(tk.END, os.path.basename(f))

    def refresh_upload_ui(self):
        mode = self.upload_mode_var.get()
        if mode == "custom":
            self.global_upload_frame.grid_remove()
            self.build_upload_table()
            self.custom_upload_frame.grid(row=0, column=0, sticky="ew")
        else:
            self.custom_upload_frame.grid_remove()
            self.global_upload_frame.grid(row=0, column=0, sticky="ew")

    def build_local_table(self):
        for w in self.local_custom_frame.winfo_children(): w.destroy()
        headers = ["File Name", "Output Path", "Remove"]
        for c, h in enumerate(headers): 
            tk.Label(self.local_custom_frame, text=h, font="Arial 8 bold").grid(row=0, column=c, sticky="ew", padx=1)
        self.local_custom_frame.columnconfigure(1, weight=1)
        for i, fpath in enumerate(self.source_files):
            row = i + 1
            self._ensure_file_data(fpath)
            dat = self.custom_file_data[fpath]
            tk.Label(self.local_custom_frame, text=os.path.basename(fpath), width=50, anchor="w").grid(row=row, column=0, sticky="w", padx=2)
            fr = tk.Frame(self.local_custom_frame)
            fr.grid(row=row, column=1, sticky="ew", padx=2)
            tk.Entry(fr, textvariable=dat["out"]).pack(side="left", fill="x", expand=True)
            tk.Button(fr, text="..", width=3, command=lambda d=dat: self.browse_file_out(d)).pack(side="right")
            btn_rem = tk.Button(self.local_custom_frame, text="X", font="Arial 8 bold",
                                command=lambda f=fpath: self.remove_single_file(f))
            btn_rem.grid(row=row, column=2, padx=2, pady=1)

    def build_upload_table(self):
        for w in self.custom_upload_frame.winfo_children(): w.destroy()
        headers = ["File", "GGUF Repo", "GGUF Folder", "FP8 Repo", "FP8 Folder", "Remove"]
        for c, h in enumerate(headers): 
            tk.Label(self.custom_upload_frame, text=h, font="Arial 8 bold").grid(row=0, column=c, sticky="ew", padx=1)
        for c in range(5): self.custom_upload_frame.columnconfigure(c, weight=1)
        for i, fpath in enumerate(self.source_files):
            row = i + 1
            self._ensure_file_data(fpath)
            dat = self.custom_file_data[fpath]
            tk.Label(self.custom_upload_frame, text=os.path.basename(fpath), width=30, anchor="w").grid(row=row, column=0, sticky="w", padx=1)
            tk.Entry(self.custom_upload_frame, textvariable=dat["gguf_r"]).grid(row=row, column=1, sticky="ew", padx=1)
            tk.Entry(self.custom_upload_frame, textvariable=dat["gguf_d"]).grid(row=row, column=2, sticky="ew", padx=1)
            tk.Entry(self.custom_upload_frame, textvariable=dat["fp8_r"]).grid(row=row, column=3, sticky="ew", padx=1)
            tk.Entry(self.custom_upload_frame, textvariable=dat["fp8_d"]).grid(row=row, column=4, sticky="ew", padx=1)
            btn_rem = tk.Button(self.custom_upload_frame, text="X", font="Arial 8 bold",
                                command=lambda f=fpath: self.remove_single_file(f))
            btn_rem.grid(row=row, column=5, padx=2, pady=1)

    def _ensure_file_data(self, fpath):
        if fpath not in self.custom_file_data:
            self.custom_file_data[fpath] = {
                "out": tk.StringVar(value=os.path.dirname(fpath)),
                "gguf_r": tk.StringVar(value=self.hf_repo_gguf.get()),
                "gguf_d": tk.StringVar(value=self.hf_dest_gguf.get()),
                "fp8_r": tk.StringVar(value=self.hf_repo_fp8.get()),
                "fp8_d": tk.StringVar(value=self.hf_dest_fp8.get())
            }

    def browse_file_out(self, dat_dict):
        d = filedialog.askdirectory()
        if d: dat_dict["out"].set(d)

    def add_files(self):
        fs = filedialog.askopenfilenames()
        for f in fs: 
            norm = os.path.normpath(f)
            if norm not in self.source_files:
                self.source_files.append(norm)
        self.refresh_file_list_ui()
        if self.upload_mode_var.get() == "custom": self.refresh_upload_ui()

    def remove_single_file(self, fpath):
        if fpath in self.source_files: self.source_files.remove(fpath)
        if fpath in self.custom_file_data: del self.custom_file_data[fpath]
        self.refresh_file_list_ui()
        if self.upload_mode_var.get() == "custom": self.refresh_upload_ui()

    def remove_selected_files(self):
        selection = self.file_listbox.curselection()
        if not selection: return
        for index in reversed(selection):
            if index < len(self.source_files): del self.source_files[index]
        self.refresh_file_list_ui()
        if self.upload_mode_var.get() == "custom": self.refresh_upload_ui()

    def clear_files(self):
        self.source_files = []
        self.custom_file_data = {}
        self.refresh_file_list_ui()
        if self.upload_mode_var.get() == "custom": self.refresh_upload_ui()

    def browse_out(self): self.out_dir_var.set(filedialog.askdirectory())
    def browse_python(self): 
        if platform.system() == "Windows": ftypes = [("Python Executable", "python.exe"), ("All Files", "*.*")]
        else: ftypes = [("Python Executable", "python3"), ("Python Executable", "python"), ("All Files", "*")]
        f = filedialog.askopenfilename(filetypes=ftypes)
        if f: self.python_path_var.set(os.path.normpath(f))
    
    def restart(self):
        target = self.python_path_var.get()
        if not os.path.exists(target): return messagebox.showerror("Error", "Python not found")
        self.save_settings(self.settings_file)
        subprocess.Popen([target] + sys.argv)
        self.root.destroy()

    def on_close(self):
        self.save_settings(self.settings_file)
        self.root.destroy()

    def show_progress_popup(self):
        if self.progress_window is None or not self.progress_window.winfo_exists():
            # Pass 'self' (the ConverterApp instance) instead of 'self.root'
            self.progress_window = ProgressPopup(self) 
        self.progress_window.deiconify()
        self.progress_window.lift()

    def cancel_processing(self):
        if not self.is_running: return
        if messagebox.askyesno("Cancel", "Stop processing?"):
            self.stop_requested = True
            logging.warning("STOP REQUESTED")
            if self.current_process:
                try: self.current_process.kill()
                except: pass

    def process_queue(self):
        try:
            self.log_display.configure(state='normal')
            updates_made = False
            
            while not self.msg_queue.empty():
                msg = self.msg_queue.get_nowait()
                if msg[0] == "UPDATE_GRID":
                    if self.progress_window and self.progress_window.winfo_exists():
                        self.progress_window.update_status(msg[1], msg[2], msg[3])
                elif msg[0] == "RAW":
                    data = msg[1]
                    updates_made = True
                    
                    i = 0
                    while i < len(data):
                        char = data[i]
                        
                        if char == '\r':
                            # Move cursor to the start of the current line
                            self.log_display.mark_set("insert", "insert linestart")
                        
                        elif char == '\n':
                            # Move to the very end of all text to add a new line
                            self.log_display.mark_set("insert", "end-1c")
                            self.log_display.insert("insert", "\n")
                        
                        elif data[i:i+3] == "\x1b[A":
                            # Move cursor UP one line (Multi-bar tqdm)
                            curr_idx = self.log_display.index("insert")
                            line, col = map(int, curr_idx.split('.'))
                            if line > 1:
                                self.log_display.mark_set("insert", f"{line - 1}.{col}")
                            i += 2 
                        
                        elif char == '\x1b':
                            pass # Skip other escape fragments
                        
                        else:
                            # Overwrite Logic:
                            # If we are NOT at the end of the current line (because of \r or Up),
                            # we delete the character in front of us before inserting the new one.
                            if self.log_display.compare("insert", "<", "insert lineend"):
                                self.log_display.delete("insert")
                            
                            self.log_display.insert("insert", char)
                        i += 1
            
            if updates_made:
                # Always scroll to the insertion point so we follow the action
                self.log_display.see("insert")
                
                # Trim logs only if they get excessively long (5000+ lines)
                total_lines = int(self.log_display.index('end-1c').split('.')[0])
                if total_lines > 5000:
                    self.log_display.delete("1.0", "500.0")

            self.log_display.configure(state='disabled')
        except:
            pass
            
        self.root.after(30, self.process_queue)

    def start_thread(self):
        if self.is_running: return
        if not self.source_files: return messagebox.showerror("Error", "No files")
        
        gen = [q for q, v in self.quant_vars_gen.items() if v.get()]
        up_only = [q for q, v in self.quant_vars_up.items() if v.get()]
        if not gen and not up_only: return messagebox.showerror("Error", "Select at least one Generate or Upload option.")
        
        self.stop_requested = False
        
        # Define Order and Groups
        SORT_ORDER = ["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
                      "Q4_0", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0",
                      "BF16", "F16", "FP8_E4M3FN", "FP8_E4M3FN (All)", "FP8_E5M2", "FP8_E5M2 (All)",
                      "MODEL", "VAE", "CLIP"]
        
        components = ["MODEL", "VAE", "CLIP"]
        fp8_variants = ["FP8_E5M2", "FP8_E5M2 (All)", "FP8_E4M3FN", "FP8_E4M3FN (All)"]
        
        active_quants = list(set(gen + up_only))
        active_quants.sort(key=lambda x: SORT_ORDER.index(x) if x in SORT_ORDER else 999)

        steps = []
        # 1. Extraction Step
        if any(x in gen for x in components):
            steps.append("Extracting")

        # 2. GGUF Prep Step (Only if generating actual GGUF quants)
        gguf_gen_needed = [q for q in gen if q not in fp8_variants and q not in components]
        if gguf_gen_needed: 
            steps.append("GGUF Prep")
        
        # 3. Add all individual quants/components to steps
        for q in active_quants: 
            steps.append(q)
        
        # 4. Final steps
        if self.do_upload.get(): steps.append("Upload")
        steps.append("Cleanup")

        self.show_progress_popup()
        model_names = [os.path.basename(f) for f in self.source_files]
        self.progress_window.setup_grid(model_names, steps)

        self.is_running = True
        self.btn_run.config(state="disabled")
        threading.Thread(target=self.run_main_logic, args=(gen, up_only)).start()

    def run_main_logic(self, gen_list, up_list):
        # Identify the current log file from the logger
        import os
        #os.environ["TQDM_MININTERVAL"] = "2.0" # Only update progress every 2 seconds
        components = ["MODEL", "VAE", "CLIP"]
        fp8_variants = ["FP8_E5M2", "FP8_E5M2 (All)", "FP8_E4M3FN", "FP8_E4M3FN (All)"]
        log_file_handle = None
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                log_file_handle = handler.stream

        # Redirect stdout/stderr using the new DualOutput logic
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = DualOutput(old_stdout, self.msg_queue, log_file_handle)
        sys.stderr = DualOutput(old_stderr, self.msg_queue, log_file_handle)
        
        try:
            strategy = self.cleanup_mode.get()
            keep_list = [q for q, v in self.quant_vars_keep.items() if v.get()]
            keep_dequant = self.keep_dequant_var.get()
            keep_convert = self.keep_convert_var.get()
            out_mode = self.out_mode_var.get()
            up_mode = self.upload_mode_var.get()
            
            if self.do_upload.get() and UPLOADER_AVAILABLE:
                from huggingface_hub import login
                login(token=self.hf_token.get(), add_to_git_credential=False)

            batch_results = []
            for f in self.source_files:
                if self.stop_requested: break
                
                fix_file = "fix_5d_tensors_wan.safetensors"
                if os.path.exists(fix_file):
                    try: os.remove(fix_file)
                    except: pass

                model_base = os.path.basename(f)
                name = re.sub(r'-(f16|F16|BF16|CONVERT|UnFixed|FIXED)$', '', os.path.splitext(model_base)[0], flags=re.IGNORECASE)
                
                if out_mode == "custom":
                    dat = self.custom_file_data.get(f, {})
                    out_dir = dat["out"].get() if "out" in dat else os.path.dirname(f)
                else:
                    base = self.out_dir_var.get() if self.out_dir_var.get() else os.path.dirname(f)
                    out_dir = os.path.join(base, name) if out_mode == "folder" else base
                
                os.makedirs(out_dir, exist_ok=True)
                
                # Clean both possible locations where fix files might linger
                locations_to_clean = [
                    os.path.dirname(os.path.abspath(__file__)),  # GUI script folder
                    out_dir                                     # current model's output folder
                ]
                
                for loc in locations_to_clean:
                    for stale in glob.glob(os.path.join(loc, "fix_5d_tensors_*.safetensors")):
                        try:
                            os.remove(stale)
                            logging.info(f"Cleaned stale fix file from {loc}: {stale}")
                        except Exception as e:
                            logging.debug(f"Could not remove {stale}: {e}")
                
                generated_files = []

                # --- Extraction Logic ---
                do_ext_m = self.quant_vars_gen.get("MODEL", tk.BooleanVar()).get()
                do_ext_v = self.quant_vars_gen.get("VAE", tk.BooleanVar()).get()
                do_ext_c = self.quant_vars_gen.get("CLIP", tk.BooleanVar()).get()

                if (do_ext_m or do_ext_c or do_ext_v) and f.lower().endswith(".safetensors"):
                    self.msg_queue.put(("UPDATE_GRID", model_base, "Extracting", "RUNNING"))
                    ext_cmd = [sys.executable, "extract_components.py", "--src", f, "--dst_dir", out_dir]
                    if do_ext_m: ext_cmd.append("--model")
                    if do_ext_c: ext_cmd.append("--clip")
                    if do_ext_v: ext_cmd.append("--vae")
                    
                    if self.run_cmd(ext_cmd):
                        self.msg_queue.put(("UPDATE_GRID", model_base, "Extracting", "DONE"))
                        # Mark them as generated so they are eligible for U and K logic
                        extracted = glob.glob(os.path.join(out_dir, f"{name}_*.safetensors"))
                        generated_files.extend(extracted)
                    else:
                        self.msg_queue.put(("UPDATE_GRID", model_base, "Extracting", "ERROR"))

                # --- FP8 Logic ---
                fp8_variants = ["FP8_E5M2", "FP8_E5M2 (All)", "FP8_E4M3FN", "FP8_E4M3FN (All)"]
                for q in fp8_variants:
                    if (q in gen_list or q in up_list) and f.lower().endswith(".safetensors"):
                        if self.stop_requested: break
                        self.msg_queue.put(("UPDATE_GRID", model_base, q, "RUNNING"))
                        
                        suffix = "_All" if "All" in q else ""
                        dtype_str = "float8_e5m2" if "E5M2" in q else "float8_e4m3fn"
                        expected_path = os.path.join(out_dir, f"{name}-{q.split()[0]}{suffix}.safetensors")
                        
                        if q in gen_list:
                            try:
                                qzer = FP8Quantizer(dtype_str)
                                ok = qzer.apply_quantization_to_file(f, expected_path, unet_only=("All" not in q), check_stop_func=lambda: self.stop_requested)
                                if ok: 
                                    generated_files.append(expected_path)
                                    self.msg_queue.put(("UPDATE_GRID", model_base, q, "DONE"))
                                else: self.msg_queue.put(("UPDATE_GRID", model_base, q, "CANCEL"))
                            except Exception as e:
                                logging.error(f"FP8 Error: {e}")
                                self.msg_queue.put(("UPDATE_GRID", model_base, q, "ERROR"))
                        elif os.path.exists(expected_path):
                            generated_files.append(expected_path)
                            self.msg_queue.put(("UPDATE_GRID", model_base, q, "DONE"))

                # --- GGUF Logic ---
                raw_combined = gen_list + up_list
                unique_tasks = list(set(raw_combined))
                all_gguf_active = [q for q in unique_tasks if "FP8" not in q and q not in components]
                if all_gguf_active:
                    gguf_gen_needed = [q for q in gen_list if "FP8" not in q and q not in components]
                    gguf_src = None
                    if gguf_gen_needed:
                        if self.stop_requested: break
                        self.msg_queue.put(("UPDATE_GRID", model_base, "GGUF Prep", "RUNNING"))
                        if f.lower().endswith(".safetensors"):
                            curr = f
                            dq = os.path.join(out_dir, f"{name}-dequant.safetensors")
                            if os.path.exists("dequantize_fp8v2.py"):
                                self.run_cmd([sys.executable, "-u", "dequantize_fp8v2.py", "--src", f, "--dst", dq, "--strip-fp8", "--dtype", "fp16"])
                                if os.path.exists(dq): curr = dq; generated_files.append(dq)
                            conv = os.path.join(out_dir, f"{name}-CONVERT.gguf")
                            self.run_cmd([sys.executable, "-u", "convert.py", "--src", curr, "--dst", conv])
                            if os.path.exists(conv): gguf_src = conv; generated_files.append(conv)
                        elif f.lower().endswith(".gguf"): gguf_src = f
                        if gguf_src: self.msg_queue.put(("UPDATE_GRID", model_base, "GGUF Prep", "DONE"))
                        else: self.msg_queue.put(("UPDATE_GRID", model_base, "GGUF Prep", "ERROR"))
                    
                    for q in all_gguf_active:
                        if self.stop_requested: break
                        self.msg_queue.put(("UPDATE_GRID", model_base, q, "RUNNING"))
                        expected_path = os.path.join(out_dir, f"{name}-{q}.gguf")
                        if q in gen_list:
                            if not gguf_src: 
                                self.msg_queue.put(("UPDATE_GRID", model_base, q, "SKIP"))
                                continue
                            if q in ["F16", "BF16"]:
                                try:
                                    shutil.copy(gguf_src, expected_path)
                                    generated_files.append(expected_path)
                                    self.msg_queue.put(("UPDATE_GRID", model_base, q, "DONE"))
                                except: self.msg_queue.put(("UPDATE_GRID", model_base, q, "ERROR"))
                                continue
                            unfixed = os.path.join(out_dir, f"{name}-{q}-UnFixed.gguf")
                            if self.run_cmd([self.quant_cmd, gguf_src, unfixed, q]):
                                final = unfixed
                                fixes = glob.glob(os.path.join(out_dir, "fix_5d_tensors_*.safetensors"))
                                if fixes:
                                    fixed = os.path.join(out_dir, f"{name}-{q}-FIXED.gguf")
                                    self.run_cmd([sys.executable, "-u", "fix_5d_tensors.py", "--src", unfixed, "--dst", fixed, "--fix", fixes[0], "--overwrite"])
                                    if os.path.exists(fixed): final = fixed
                                try: os.rename(final, expected_path); generated_files.append(expected_path)
                                except: generated_files.append(final)
                                if os.path.exists(unfixed) and os.path.abspath(unfixed) != os.path.abspath(expected_path):
                                    try: os.remove(unfixed)
                                    except: pass
                                self.msg_queue.put(("UPDATE_GRID", model_base, q, "DONE"))
                            else: self.msg_queue.put(("UPDATE_GRID", model_base, q, "ERROR"))
                        elif q in up_list:
                            if os.path.exists(expected_path):
                                generated_files.append(expected_path)
                                self.msg_queue.put(("UPDATE_GRID", model_base, q, "DONE"))
                            else: self.msg_queue.put(("UPDATE_GRID", model_base, q, "SKIP"))

                generated_files = list(set(generated_files))
                res_obj = { "name": name, "files": generated_files, "model_display": model_base, "src_path": f }
                batch_results.append(res_obj)
                if strategy == "per_model" and not self.stop_requested:
                    self.handle_upload_cleanup(res_obj, keep_list, up_list, up_mode, out_mode, keep_dequant, keep_convert)

            if strategy == "all_end" and not self.stop_requested:
                for item in batch_results:
                    if self.stop_requested: break
                    self.handle_upload_cleanup(item, keep_list, up_list, up_mode, out_mode, keep_dequant, keep_convert)

            if self.shutdown_var.get() and not self.stop_requested:
                if platform.system() == "Windows": subprocess.run(["shutdown", "/s", "/t", "60"])
                else: subprocess.run(["sudo", "shutdown", "-h", "+1"])
            if not self.stop_requested: messagebox.showinfo("Done", "Finished")
        except Exception as e:
            logging.exception("Error")
            messagebox.showerror("Error", str(e))
        finally:
            # Restore streams when thread finishes
            sys.stdout, sys.stderr = old_stdout, old_stderr
            self.is_running = False
            self.btn_run.config(state="normal")

    def _check_file_match_quant(self, fname, q):
        if q == "MODEL": return fname.endswith("_model.safetensors")
        if q == "VAE":   return fname.endswith("_vae.safetensors")
        if q == "CLIP":  
            suffixes = ["_clip.safetensors", "_clip_l.safetensors", "_clip_g.safetensors", 
                        "_clip_h.safetensors", "_t5xxl.safetensors", "_t5base.safetensors", 
                        "_llama.safetensors", "_gemma.safetensors"]
            return any(fname.endswith(s) for s in suffixes)
        if "FP8" in q:
            base_q = q.split(" ")[0] 
            is_all_q = "(All)" in q
            if is_all_q: return (base_q in fname and "_All" in fname)
            else: return (base_q in fname and "_All" not in fname)
        if q in ["F16", "BF16"]: return f"-{q}.gguf" in fname
        return f"-{q}.gguf" in fname

    def handle_upload_cleanup(self, item, keep_list, up_list, up_mode, out_mode, keep_dequant, keep_convert):
        if self.stop_requested: return
        
        # Identify the current log file handle
        log_file_handle = None
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                log_file_handle = handler.stream

        if not UPLOADER_AVAILABLE:
            try_load_uploader()

        name, files, disp, src = item['name'], item['files'], item['model_display'], item['src_path']
        r_gguf, d_gguf, r_fp8, d_fp8 = self.hf_repo_gguf.get(), self.hf_dest_gguf.get(), self.hf_repo_fp8.get(), self.hf_dest_fp8.get()
        
        if up_mode == "custom":
            dat = self.custom_file_data.get(src, {})
            if "gguf_r" in dat and dat["gguf_r"].get(): r_gguf = dat["gguf_r"].get()
            if "gguf_d" in dat and dat["gguf_d"].get(): d_gguf = dat["gguf_d"].get()
            if "fp8_r" in dat and dat["fp8_r"].get(): r_fp8 = dat["fp8_r"].get()
            if "fp8_d" in dat and dat["fp8_d"].get(): d_fp8 = dat["fp8_d"].get()

        if out_mode == "folder" and up_mode == "global":
            d_gguf = f"{d_gguf}/{name}" if d_gguf else name
            d_fp8 = f"{d_fp8}/{name}" if d_fp8 else name

        if self.do_upload.get():
            if not UPLOADER_AVAILABLE:
                logging.error("Upload requested but upload_to_hf.py is missing.")
                self.msg_queue.put(("UPDATE_GRID", disp, "Upload", "ERROR"))
            else:
                self.msg_queue.put(("UPDATE_GRID", disp, "Upload", "RUNNING"))
                
                # 1. Identify which files belong to which repo
                files_to_upload = []
                for f in files:
                    fname = os.path.basename(f)
                    if any(x in fname for x in ["-CONVERT", "-UnFixed", "-dequant"]): continue
                    
                    for q in up_list: 
                        if self._check_file_match_quant(fname, q):
                            files_to_upload.append(f) # FIXED
                            break
                
                files_to_upload = list(set(files_to_upload)) 

                files_for_fp8_repo = []
                files_for_gguf_repo = []

                # Define the suffixes that identify extracted parts
                extraction_suffixes = ["_model", "_vae", "_clip_l", "_clip_g", "_clip_h", "_t5xxl", "_t5base", "_llama", "_gemma", "_clip"]

                for f_path in files_to_upload:
                    fname = os.path.basename(f_path).lower()
                    is_component = any(suffix in fname for suffix in extraction_suffixes)
                    
                    if fname.endswith(".gguf"):
                        files_for_gguf_repo.append(f_path)
                    elif fname.endswith(".safetensors"):
                        files_for_fp8_repo.append(f_path)
                        # If it's a VAE/CLIP/Model part, put it in GGUF repo too
                        if is_component:
                            files_for_gguf_repo.append(f_path)

                try:
                    success = True
                    python_exe = self.python_path_var.get()
                    token = self.hf_token.get()

                    # 2. Launch Uploader processes
                    if files_for_fp8_repo and r_fp8:
                        logging.info(f"Launching Safetensors/FP8 Upload for {len(files_for_fp8_repo)} files...")
                        cmd = [python_exe, "upload_to_hf.py", "--token", token, "--repo", r_fp8, "--dest", d_fp8, "--yes", "--path"] + files_for_fp8_repo
                        if not self.run_cmd(cmd): success = False
                    
                    if files_for_gguf_repo and r_gguf:
                        logging.info(f"Launching GGUF Upload for {len(files_for_gguf_repo)} files...")
                        cmd = [python_exe, "upload_to_hf.py", "--token", token, "--repo", r_gguf, "--dest", d_gguf, "--yes", "--path"] + files_for_gguf_repo
                        if not self.run_cmd(cmd): success = False

                    if success:
                        self.msg_queue.put(("UPDATE_GRID", disp, "Upload", "DONE"))
                    else:
                        self.msg_queue.put(("UPDATE_GRID", disp, "Upload", "ERROR"))
                except Exception as e:
                    logging.error(f"Upload Process Error: {e}")
                    self.msg_queue.put(("UPDATE_GRID", disp, "Upload", "ERROR"))
        else:
            self.msg_queue.put(("UPDATE_GRID", disp, "Upload", "SKIP"))

        self.msg_queue.put(("UPDATE_GRID", disp, "Cleanup", "RUNNING"))
        for p in files:
            if not os.path.exists(p): continue
            fname = os.path.basename(p)
            should_keep = False
            for q in keep_list:
                if self._check_file_match_quant(fname, q):
                    should_keep = True; break
            if keep_dequant and "-dequant.safetensors" in fname: should_keep = True
            if keep_convert and "-CONVERT.gguf" in fname: should_keep = True
            if not should_keep: 
                try: os.remove(p)
                except: pass
        self.msg_queue.put(("UPDATE_GRID", disp, "Cleanup", "DONE"))

    def run_cmd(self, cmd):
        import os
        logging.info(f"CMD: {' '.join(cmd)}")
        
        log_file = None
        try: log_file = open(self.current_log_path, "a", encoding="utf-8", errors="replace")
        except: pass

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["TERM"] = "xterm-256color"
        env["COLUMNS"] = "130" # Set a fixed width
        env["TQDM_TTY"] = "1"  # CRITICAL: Forces tqdm to use \r updates
        env["PYTHONIOENCODING"] = "utf-8"

        try:
            self.current_process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                bufsize=0, env=env # Binary mode
            )

            while True:
                # Read in small chunks to keep the UI snappy
                chunk = self.current_process.stdout.read(128)
                if not chunk and self.current_process.poll() is not None:
                    break
                
                if chunk:
                    # Mirror to CMD
                    os.write(1, chunk)
                    
                    text_data = chunk.decode('utf-8', errors='replace')
                    
                    # Write to File
                    if log_file:
                        log_file.write(text_data)
                        log_file.flush()
                    
                    # Send to GUI
                    self.msg_queue.put(("RAW", text_data))

            return (self.current_process.wait() == 0)
        except Exception as e:
            logging.error(f"Execution error: {e}")
            return False
        finally:
            if log_file: log_file.close()

    def save_settings(self, f):
        # Prepare custom file data for JSON (convert StringVars to strings)
        serializable_custom_data = {}
        for path, data in self.custom_file_data.items():
            serializable_custom_data[path] = {
                "out": data["out"].get(),
                "gguf_r": data["gguf_r"].get(),
                "gguf_d": data["gguf_d"].get(),
                "fp8_r": data["fp8_r"].get(),
                "fp8_d": data["fp8_d"].get()
            }

        d = {
            "python": self.python_path_var.get(), 
            "out": self.out_dir_var.get(),
            "out_mode": self.out_mode_var.get(), 
            "up_mode": self.upload_mode_var.get(),
            "do_upload": self.do_upload.get(), # Added
            "token": self.hf_token.get(), 
            "r_gguf": self.hf_repo_gguf.get(), 
            "d_gguf": self.hf_dest_gguf.get(),
            "r_fp8": self.hf_repo_fp8.get(), 
            "d_fp8": self.hf_dest_fp8.get(), 
            "clean": self.cleanup_mode.get(),
            "shut": self.shutdown_var.get(), 
            "q_gen": [k for k,v in self.quant_vars_gen.items() if v.get()],
            "q_up": [k for k,v in self.quant_vars_up.items() if v.get()],
            "q_keep": [k for k,v in self.quant_vars_keep.items() if v.get()],
            "k_dequant": self.keep_dequant_var.get(), 
            "k_convert": self.keep_convert_var.get(),
            "geometry": self.root.geometry(),
            "source_files": self.source_files, # Added
            "custom_file_data": serializable_custom_data, # Added
            "theme": self.current_theme
        }
        try: 
            with open(f, 'w') as jf:
                json.dump(d, jf, indent=4)
        except Exception as e: 
            print(f"Save settings error: {e}")

    def load_settings(self, f, silent=False):
        if not os.path.exists(f): return
        try:
            with open(f, 'r') as jf:
                d = json.load(jf)
            
            # 1. Basic Variables
            if "python" in d: self.python_path_var.set(d["python"])
            if "out" in d: self.out_dir_var.set(d["out"])
            if "token" in d: self.hf_token.set(d["token"])
            if "r_gguf" in d: self.hf_repo_gguf.set(d["r_gguf"])
            if "d_gguf" in d: self.hf_dest_gguf.set(d["d_gguf"])
            if "r_fp8" in d: self.hf_repo_fp8.set(d["r_fp8"])
            if "d_fp8" in d: self.hf_dest_fp8.set(d["d_fp8"])
            if "out_mode" in d: self.out_mode_var.set(d["out_mode"])
            if "up_mode" in d: self.upload_mode_var.set(d["up_mode"])
            if "do_upload" in d: self.do_upload.set(d["do_upload"]) # Added
            if "clean" in d: self.cleanup_mode.set(d["clean"])
            if "shut" in d: self.shutdown_var.set(d["shut"])
            if "k_dequant" in d: self.keep_dequant_var.set(d["k_dequant"])
            if "k_convert" in d: self.keep_convert_var.set(d["k_convert"])
            if "geometry" in d: self.root.geometry(d["geometry"])
            if "theme" in d: 
                self.current_theme = d["theme"]
                self.root.after(200, self.apply_theme)
                        
            # 2. Source Files & Custom Data (Crucial for the Routing Tables)
            if "source_files" in d: 
                self.source_files = d["source_files"]
            
            if "custom_file_data" in d:
                for path, data in d["custom_file_data"].items():
                    self.custom_file_data[path] = {
                        "out": tk.StringVar(value=data.get("out", "")),
                        "gguf_r": tk.StringVar(value=data.get("gguf_r", "")),
                        "gguf_d": tk.StringVar(value=data.get("gguf_d", "")),
                        "fp8_r": tk.StringVar(value=data.get("fp8_r", "")),
                        "fp8_d": tk.StringVar(value=data.get("fp8_d", ""))
                    }

            # 3. Quantization Grid
            for v in self.quant_vars_gen.values(): v.set(False)
            for v in self.quant_vars_up.values(): v.set(False)
            for v in self.quant_vars_keep.values(): v.set(False)
            for q in d.get("q_gen", []): 
                if q in self.quant_vars_gen: self.quant_vars_gen[q].set(True)
            for q in d.get("q_up", []): 
                if q in self.quant_vars_up: self.quant_vars_up[q].set(True)
            for q in d.get("q_keep", []): 
                if q in self.quant_vars_keep: self.quant_vars_keep[q].set(True)
            
            # 4. Refresh UI to reflect loaded state
            self.refresh_file_list_ui()
            self.refresh_upload_ui()
            
        except Exception as e: 
            print(f"Load settings error: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ConverterApp(root)
    root.mainloop()