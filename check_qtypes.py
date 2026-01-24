import gguf
import inspect

print("--- Available quantization types in your gguf library ---")
try:
    for name, member in inspect.getmembers(gguf.GGMLQuantizationType):
        # We only want to print the actual enum members, not internal attributes
        if not name.startswith('_'):
            print(name)
except Exception as e:
    print(f"An error occurred while inspecting the library: {e}")

print("--- End of list ---")