import torch
import platform


def get_device():
    """
    Detect and return the best available device (CUDA GPU, MPS, or CPU).

    Returns:
        torch.device: The device to use for inference
        str: Device name for display
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_name = f"NVIDIA GPU (CUDA) - {torch.cuda.get_device_name(0)}"
    elif torch.backends.mps.is_available() and platform.system() == "Darwin":
        device = torch.device("mps")
        device_name = "Apple Silicon (MPS)"
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    return device, device_name
