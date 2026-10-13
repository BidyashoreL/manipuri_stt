import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
    
    a = torch.randn(1000, 1000, device="cuda")
    b = torch.randn(1000, 1000, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    
    print("Matmul result sample:", c[0, 0].item())
    print("GPU actually works.")
else:
    print("GPU is not accessible to PyTorch.")