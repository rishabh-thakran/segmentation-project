import torch

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU Device Count: {torch.cuda.device_count()}")
print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
