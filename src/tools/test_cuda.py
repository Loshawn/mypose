import torch

num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")

for i in range(num_gpus):
    device = torch.device(f'cuda:{i}')
    is_available = torch.cuda.is_available() and (torch.cuda.current_device() == i)
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}, Is Available: {is_available}")