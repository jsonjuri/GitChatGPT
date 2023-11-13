import torch.cuda
from common import config
from typing import Any

def get_gpu_memory() -> int:
    """
    Returns the amount of free memory in megabytes for each GPU.
    """
    return int(torch.cuda.mem_get_info()[0] / (1024 ** 2))


def calculate_layer_count() -> None | int | float | Any:
    """
    Calculates the number of layers that can be used on the GPU.
    """
    if not config.get('GPU_ENABLED'):
        return None
    if not torch.cuda.is_available():
        print("You have chosen to use a GPU, but it seems your installation was for CPU. Please update through the installer.")
        exit(1)
    if (model_n_gpu_layers := config.get('MODEL_N_GPU_LAYERS')) is not None:
        return model_n_gpu_layers
    else:
        # This is the size of a single layer in VRAM, and is an approximation.
        layers_size_mb = 120.6
        # The current value is for 7B models. For other models, this value should be adjusted.
        # About 700 MB is needed for the LLM to run, so we reduce the layer count by 6 to be safe.
        layers_to_reduce = 6
        if (get_gpu_memory() // layers_size_mb) - layers_to_reduce > 32:
            return 32
        else:
            return get_gpu_memory() // layers_size_mb - layers_to_reduce
