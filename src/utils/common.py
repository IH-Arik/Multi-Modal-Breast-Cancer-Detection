"""
Common utilities for breast cancer detection research.
"""

import torch
import numpy as np
from typing import Dict, Any


def set_device():
    """Set and return the appropriate device (CUDA/CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def json_default(obj):
    """Convert numpy / torch types into JSON-serializable Python objects."""
    import numpy as np
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist() if obj.numel() != 1 else obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def convert_results_to_native(results: Dict[str, Any]) -> Dict[str, Any]:
    """Convert all numpy types in results to native Python types."""
    import numpy as np
    
    native_results = {}
    for key, value in results.items():
        if isinstance(value, np.generic):
            native_results[key] = value.item()
        elif isinstance(value, np.ndarray):
            native_results[key] = value.tolist()
        else:
            native_results[key] = value
    return native_results


def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_model_info(model, model_name="Model"):
    """Print model information including parameter count."""
    total_params, trainable_params = count_parameters(model)
    print(f"{model_name} Information:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {next(model.parameters()).device}")
