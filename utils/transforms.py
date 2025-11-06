"""
Data transformations and preprocessing utilities.

Provides common transformations that can be reused across different training pipelines.
All transforms follow PyTorch conventions and can be composed using torchvision.transforms.Compose.
"""

import torch
import numpy as np
from typing import Union, Tuple, Optional
import torchvision.transforms as T


def center_crop_transform(tensor: torch.Tensor, crop_size: int) -> torch.Tensor:
    """
    Center crop a tensor to crop_size × crop_size pixels.
    
    Args:
        tensor: Input tensor of shape [B, C, H, W] or [C, H, W]
        crop_size: Target size for both height and width
        
    Returns:
        Cropped tensor of shape [B, C, crop_size, crop_size] or [C, crop_size, crop_size]
    """
    if len(tensor.shape) == 3:
        # Single image: [C, H, W]
        _, h, w = tensor.shape
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2
        return tensor[:, start_h:start_h + crop_size, start_w:start_w + crop_size]
    elif len(tensor.shape) == 4:
        # Batch of images: [B, C, H, W]
        _, _, h, w = tensor.shape
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2
        return tensor[:, :, start_h:start_h + crop_size, start_w:start_w + crop_size]
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tensor.shape}")


def log_scale_transform(tensor: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """
    Apply log-scaling to tensor values.
    
    This is useful for data with large dynamic ranges (like optical intensity data)
    where features at different scales need to be emphasized.
    
    Args:
        tensor: Input tensor (assumed to be in [0, 1] range after normalization)
        epsilon: Small constant added before taking log to avoid log(0)
        
    Returns:
        Log-scaled tensor
    """
    # Apply log(1 + x) transform to compress dynamic range
    # Input should be normalized to [0, 1], output will be roughly [0, log(2)] ≈ [0, 0.69]
    return torch.log1p(tensor)


def normalize_transform(tensor: torch.Tensor, 
                       input_range: str = 'uint16',
                       output_range: Tuple[float, float] = (0.0, 1.0)) -> torch.Tensor:
    """
    Normalize tensor from input range to output range.
    
    Args:
        tensor: Input tensor
        input_range: Type of input data ('uint16', 'uint8', 'float', 'auto')
        output_range: Target range for output values
        
    Returns:
        Normalized tensor
    """
    if input_range == 'uint16':
        # Convert uint16 [0, 65535] to [0, 1]
        normalized = tensor.float() / 65535.0
    elif input_range == 'uint8':
        # Convert uint8 [0, 255] to [0, 1]
        normalized = tensor.float() / 255.0
    elif input_range == 'float':
        # Assume already in [0, 1] range
        normalized = tensor.float()
    elif input_range == 'auto':
        # Auto-detect based on max value
        # Convert to float first to avoid uint16 max() issues
        tensor_float = tensor.float()
        max_val = tensor_float.max()
        if max_val > 256:
            normalized = tensor_float / 65535.0
        elif max_val > 1.0:
            normalized = tensor_float / 255.0
        else:
            normalized = tensor_float
    else:
        raise ValueError(f"Unknown input_range: {input_range}")
    
    # Scale to output range
    if output_range != (0.0, 1.0):
        min_out, max_out = output_range
        normalized = normalized * (max_out - min_out) + min_out
    
    return normalized


class ToTensor:
    """Convert numpy arrays to PyTorch tensors with optional normalization."""
    
    def __init__(self, normalize: bool = True, input_range: str = 'auto'):
        self.normalize = normalize
        self.input_range = input_range
    
    def __call__(self, data: Union[np.ndarray, Tuple[np.ndarray, ...]]) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if isinstance(data, tuple):
            # Handle multiple arrays (e.g., observation, action pairs)
            tensors = []
            for i, array in enumerate(data):
                tensor = torch.from_numpy(array)
                if self.normalize and i == 0:  # Normalize observations only
                    tensor = normalize_transform(tensor, self.input_range)
                elif not self.normalize:
                    tensor = tensor.float()
                tensors.append(tensor)
            return tuple(tensors)
        else:
            # Single array
            tensor = torch.from_numpy(data)
            if self.normalize:
                tensor = normalize_transform(tensor, self.input_range)
            else:
                tensor = tensor.float()
            return tensor


class CenterCrop:
    """Center crop transformation following PyTorch conventions."""
    
    def __init__(self, size: int):
        self.size = size
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return center_crop_transform(tensor, self.size)


class LogScale:
    """Log-scale transformation for data with large dynamic ranges."""
    
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return log_scale_transform(tensor, self.epsilon)


class Compose:
    """Compose multiple transforms together (similar to torchvision.transforms.Compose)."""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data


# Common transform presets
def get_autoencoder_transforms(crop_size: Optional[int] = None, 
                              normalize: bool = True,
                              log_scale: bool = False) -> Compose:
    """Get standard transforms for autoencoder training."""
    transforms = []
    
    if normalize:
        transforms.append(ToTensor(normalize=True))
    else:
        transforms.append(ToTensor(normalize=False))
    
    if crop_size is not None:
        transforms.append(CenterCrop(crop_size))
    
    if log_scale:
        transforms.append(LogScale())
    
    return Compose(transforms)


def get_supervised_transforms(crop_size: Optional[int] = None,
                             normalize_obs: bool = True) -> Compose:
    """Get standard transforms for supervised learning (observation-action pairs)."""
    transforms = []
    
    # Convert to tensors with observation normalization
    transforms.append(ToTensor(normalize=normalize_obs))
    
    # Apply cropping to observations only (handled in dataset __getitem__)
    if crop_size is not None:
        transforms.append(CenterCrop(crop_size))
    
    return Compose(transforms)


class ClipActions:
    """
    Clip action values to a specified range.
    
    Useful for ensuring training targets stay within valid action bounds,
    particularly when dataset contains actions outside [-1, 1] range.
    """
    
    def __init__(self, min_val: float = -1.0, max_val: float = 1.0):
        """
        Args:
            min_val: Minimum action value
            max_val: Maximum action value
        """
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, action: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Clip action values.
        
        Args:
            action: Action array or tensor to clip
            
        Returns:
            Clipped action in same format as input
        """
        if isinstance(action, torch.Tensor):
            return torch.clamp(action, self.min_val, self.max_val)
        else:
            # NumPy array
            return np.clip(action, self.min_val, self.max_val)


# Export commonly used transforms
__all__ = [
    'center_crop_transform',
    'normalize_transform', 
    'ToTensor',
    'CenterCrop',
    'ClipActions',
    'Compose',
    'get_autoencoder_transforms',
    'get_supervised_transforms',
]