"""
Utils Package

Centralized utilities for data handling, dataset creation, and common functionality
across different training pipelines (autoencoder, supervised learning, behavior cloning).

Key Components:
- datasets: Modular dataset classes for different data types and formats
- data_loading: Common data loading utilities for HDF5, NPZ, JSON formats
- transforms: Reusable data transformations and preprocessing
- replay_buffers: Memory-efficient replay buffer implementations

Design Principles:
- Modularity: Components can be mixed and matched across use cases
- PyTorch idioms: Inherits from standard PyTorch Dataset and DataLoader patterns
- Efficiency: Memory-efficient lazy loading for large datasets
- Flexibility: Support for multiple data formats and transformations
"""

from .datasets import *
from .data_loading import *
from .transforms import *

__all__ = [
    # Dataset classes
    'BaseDataset',
    'AutoencoderDataset', 
    'SupervisedDataset',
    'BehaviorCloningDataset',
    'LazyDataset',
    
    # Data loading utilities
    'DatasetDiscovery',
    'FileLoader',
    'CacheManager',
    
    # Transforms
    'center_crop_transform',
    'normalize_transform',
    'ToTensor',
    'Compose',
]