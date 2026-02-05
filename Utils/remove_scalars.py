# A brute-force, backward-compatible script to remove scalar tensors from a GGUF file.
# This version copies ONLY the architecture and the valid tensors, skipping all other metadata
# to avoid bugs in older gguf libraries.
import os
import gguf
import argparse
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Fix a GGUF file by removing scalar tensors (brute-force method).")
    parser.add_argument("--src", required=True, help="The source GGUF file that is causing the crash.")
    parser.add_argument("--dst", required=True, help="The path to save the fixed GGUF file.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.src):
        parser.error(f"Invalid source file: '{args.src}'")
    if not args.overwrite and os.path.exists(args.dst):
        parser.error(f"Output file exists. Use --overwrite to replace it: '{args.dst}'")

    return args

if __name__ == "__main__":
    args = get_args()

    print(f"Loading source GGUF file: {args.src}")
    reader = gguf.GGUFReader(args.src)

    # Prepare the new GGUF writer, copying ONLY the architecture.
    arch = str(reader.fields['general.architecture'].parts[-1], encoding='utf-8')
    writer = gguf.GGUFWriter(path=None, arch=arch)

    print("Filtering and copying tensors (metadata will be minimal)...")
    tensors_skipped = 0
    # Iterate through all tensors in the source file
    for tensor in tqdm(reader.tensors, desc="Processing tensors"):
        # The critical check: if a tensor has 0 dimensions, it's a scalar.
        if tensor.data.ndim == 0:
            print(f"Found and skipped scalar tensor: '{tensor.name}'")
            tensors_skipped += 1
            continue # Skip this tensor

        # If it's not a scalar, copy it to the new file
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)

    if tensors_skipped == 0:
        print("\nWarning: No scalar tensors were found. The output file will be stripped of most metadata.")
    else:
        print(f"\nSuccessfully skipped {tensors_skipped} scalar tensor(s).")

    # Write the new, fixed GGUF file to disk
    print(f"Saving fixed GGUF file to: {args.dst}")
    writer.write_header_to_file(path=args.dst)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    print("\nFix complete. You can now use the new file for quantization.")