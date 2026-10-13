import sys
import importlib.util
import platform

print("========== ENVIRONMENT CHECK ==========")
print("Python:", sys.version)
print("Platform:", platform.platform())

for package in ["torch", "transformers", "datasets", "evaluate",
                "jiwer", "accelerate", "librosa", "soundfile"]:
    print(f"{package}:",
          "installed" if importlib.util.find_spec(package) else "NOT INSTALLED")

try:
    import torch
    print("\n========== PYTORCH ==========")
    print("Version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU memory (GB):",
              round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))

    if hasattr(torch.backends, "mps"):
        print("Apple MPS available:", torch.backends.mps.is_available())
except Exception as e:
    print("PyTorch check failed:", repr(e))
