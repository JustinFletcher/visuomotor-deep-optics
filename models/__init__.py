"""
Models Package

This package provides reusable neural network architectures and utilities for
model persistence, component extraction, and transfer learning.

Key Components:
- models: Reusable neural network architectures
- model_utils: Utilities for saving, loading, and managing trained models

Main Features:
- Codebase-independent model persistence
- Component extraction and reuse
- Model registry for systematic tracking
- TorchScript export for deployment
- Transfer learning support
"""

# Import main model architectures
from .models import (
    VanillaCNN,
    ResNet18GroupNorm,
    ResNet18Actor,
    BasicBlockGroupNorm,
    AutoEncoderCNN,
    AutoEncoderResNet,
    create_model,
    center_crop_transform
)

# Import model utilities
from .model_utils import (
    save_trained_model,
    load_trained_model,
    create_model_with_pretrained_features,
    ModelRegistry,
    ModelSaver,
    ModelLoader,
    ComponentExtractor
)

__all__ = [
    # Model architectures
    'VanillaCNN',
    'ResNet18GroupNorm',
    'ResNet18Actor',
    'BasicBlockGroupNorm',
    'AutoEncoderCNN',
    'AutoEncoderResNet',
    'create_model',
    'center_crop_transform',
    
    # Model utilities
    'save_trained_model',
    'load_trained_model',
    'create_model_with_pretrained_features',
    'ModelRegistry',
    'ModelSaver',
    'ModelLoader',
    'ComponentExtractor',
]

__version__ = '1.0.0'
