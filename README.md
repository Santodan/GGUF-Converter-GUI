# GGUF-Converter-GUI

This is my attempt of making the convertion of GGUFs easier.

I wen ahead and created this user-friendly graphical interface designed to simplify the process of converting and quantizing AI models into the GGUF format.

This tool eliminates the need for manual command-line execution, providing a visual way to handle model paths, output settings, quantization levels and direct upload to hugginface.

All the scripts were based, or are coming from, https://github.com/city96/ComfyUI-GGUF/tree/main/tools and https://github.com/Kickbub/Dequant-FP8-ComfyUI/blob/main/dequantize_fp8v2.py

**In this repo, I have the `llama-quantize` ready for linux and AMD CPU, and `llama-quantize.exe` ready for Windows and AMD CPU. For Intel CPU, you might need to generate it again, it is pretty simple if you follow the steps on https://github.com/city96/ComfyUI-GGUF/tree/main/tools**

## 🚀 Features
- **No CLI Required**: Fully graphical interface for all conversion steps.
- **HuggingFace Integration**: Easily select downloaded models from your local storage.
- **Custom Quantization**: Support for all standard GGUF quantization types (e.g., Q4_K_M, Q8_0, Q5_K_S, etc.).
- **Output Management**: Custom output naming and directory selection.
- **Progress Tracking**: Real-time console/status output within the GUI to monitor the conversion process.[1]

<img width="1920" height="1080" alt="c86e9333-d276-4214-9dc9-d24f849d3d1c" src="https://github.com/user-attachments/assets/177c35c6-737d-47be-9bca-dba32eada1e9" />


## 🛠️ Prerequisites
- **Python 3.10+**
- **Libraries**:
  - safetensors
  - huggingface_hub
  - tqdm
  - sentencepiece
  - numpy
  - gguf
  - prompt_toolkit
  - requests
 
## 📦 Installation
1. Clone this repository:
   ```
   git clone https://github.com/Santodan/GGUF-Converter-GUI.git
   cd GGUF-Converter-GUI
   ```
2. Install all the requirements
   ```
   pip install -r requirements.txt
   ```
3. Run the `gui_run_conversion.py`
   
Optional
- **Hugginface Token**
  In case you want to autmatically gather the Hugginface token from the OS, you can set the variable `HUGGING_FACE_HUB_TOKEN` that the code will automatically gather it
- If you want to use an existing python environmnet, you will need to activate that environmnet before running the `gui_run_conversion.py`

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.
