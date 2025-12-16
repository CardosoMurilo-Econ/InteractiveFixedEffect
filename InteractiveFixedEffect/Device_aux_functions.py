import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_device():
    """Returns the best available device (CUDA if available, otherwise CPU)."""
    if torch.cuda.is_available():
        #print("Using GPU (CUDA).")
        return torch.device("cuda")
    else:
        print("GPU not available. Using CPU.")
        return torch.device("cpu")

def ensure_tensor(data, dtype=torch.float64):
    """Converts data to a PyTorch tensor if it's not already one."""
    if not isinstance(data, torch.Tensor):
        return torch.tensor(data, dtype=dtype)
    
    return data

def move_to_device(tensor, device):
    """Moves tensor to the specified device only if necessary."""
    tensor = ensure_tensor(tensor, dtype=torch.float64)  # Ensure it's a tensor first
    if tensor.device != device:
        return tensor.to(device)
    return tensor

def move_to_cpu(tensor):
    """Moves tensor to CPU only if necessary."""
    if not isinstance(tensor, torch.Tensor):
        return tensor
    
    if tensor.device != 'cpu':
        tensor = tensor.cpu().detach().numpy()
        return tensor

    return tensor

    
