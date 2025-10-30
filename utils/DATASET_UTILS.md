# Unified Dataset Utilities

This folder provides centralized, modular dataset handling capabilities for all ML training pipelines in the project.

## 🎯 **Design Goals**

- **Modularity**: Components can be mixed and matched across different use cases
- **PyTorch Idioms**: Inherits from standard PyTorch Dataset and DataLoader patterns  
- **Memory Efficiency**: Lazy loading for large datasets (100K+ samples)
- **Format Flexibility**: Support for HDF5, NPZ, and JSON data formats
- **Task Agnostic**: Same utilities work for autoencoder, supervised learning, behavior cloning

## 📁 **Structure**

```
utils/
├── __init__.py           # Package exports
├── datasets.py           # PyTorch Dataset classes
├── data_loading.py       # File format handling and discovery
├── transforms.py         # Data transformations and preprocessing
└── README.md            # This file
```

## 🔧 **Core Components**

### **Dataset Classes** (`datasets.py`)

All inherit from `torch.utils.data.Dataset` and follow PyTorch conventions:

- **`BaseDataset`**: Abstract base with common functionality
- **`AutoencoderDataset`**: For reconstruction tasks (input = target)
- **`SupervisedDataset`**: For observation → action mapping
- **`BehaviorCloningDataset`**: Alias for supervised with better semantics
- **`LazyDataset`**: Memory-efficient on-demand loading for large datasets

### **Data Loading** (`data_loading.py`)

- **`DatasetDiscovery`**: Automatically detect and analyze dataset formats
- **`FileLoader`**: Efficient loading for HDF5, NPZ, JSON files
- **`CacheManager`**: Smart caching for improved performance

### **Transforms** (`transforms.py`)

- **`center_crop_transform()`**: Center crop to NxN pixels
- **`normalize_transform()`**: Normalize from uint16/uint8 to float ranges
- **`ToTensor`**: Convert numpy arrays to PyTorch tensors
- **`Compose`**: Chain multiple transforms together

## 🚀 **Usage Examples**

### **Autoencoder Training**

```python
from utils.datasets import LazyDataset
from torch.utils.data import DataLoader

# Memory-efficient loading for large datasets
dataset = LazyDataset(
    dataset_path="datasets/sa_dataset_100k",
    task_type='autoencoder',
    input_crop_size=256,  # Crop 512x512 → 256x256
    max_examples=None     # Use full dataset
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for inputs, targets in dataloader:
    # inputs == targets for autoencoder
    loss = criterion(model(inputs), targets)
```

### **Supervised Learning**

```python
from utils.datasets import LazyDataset

# Same utilities, different task type
dataset = LazyDataset(
    dataset_path="datasets/sml_dataset",
    task_type='supervised',
    input_crop_size=128
)

for observations, actions in DataLoader(dataset, batch_size=32):
    predictions = model(observations)
    loss = criterion(predictions, actions)
```

### **Small Dataset (In-Memory)**

```python
from utils.datasets import AutoencoderDataset
from utils.data_loading import DatasetDiscovery, FileLoader

# For small datasets, load everything into memory
file_paths, dataset_type, metadata = DatasetDiscovery.discover_files("small_dataset")
observations = []

for file_path in file_paths:
    data = FileLoader.load_hdf5_observations(file_path)
    observations.extend(data['observations'])

dataset = AutoencoderDataset(observations, input_crop_size=256)
```

## 🔄 **Migration from Existing Code**

### **Before** (scattered dataset handling):
```python
# Different data loading for each script
class OptomechDataset(Dataset):  # in train_sml_model.py
class AutoencoderDataset(Dataset):  # in train_autoencoder.py  
class BehaviorCloningDataset(Dataset):  # in train_bc.py

# Duplicated file loading logic
def load_hdf5_files(...):  # repeated across scripts
def load_npz_files(...):   # repeated across scripts
```

### **After** (unified utilities):
```python
from utils.datasets import LazyDataset

# Same interface for all tasks
autoencoder_data = LazyDataset(path, task_type='autoencoder')
supervised_data = LazyDataset(path, task_type='supervised') 
bc_data = LazyDataset(path, task_type='behavior_cloning')
```

## 🎛️ **Configuration Options**

### **Dataset Parameters**

- **`task_type`**: `'autoencoder'`, `'supervised'`, `'behavior_cloning'`
- **`input_crop_size`**: Center crop input images (e.g., 512→256)
- **`max_examples`**: Limit dataset size for debugging
- **`use_cache`**: Enable/disable caching for performance
- **`transform`**: Custom PyTorch transforms

### **Supported Data Formats**

- **HDF5** (`.h5`, `.hdf5`): Preferred for large datasets
  - Keys: `observations`, `actions` (optional)
- **NPZ** (`.npz`): NumPy compressed archives
  - Auto-detects observation/action keys
- **JSON** (`.json`): Episode/batch files
  - Format: `{"observations": [...], "actions": [...]}`

## 🧪 **Testing**

Run the demo script to test all functionality:

```bash
poetry run python demo_dataset_utils.py
```

This will test:
- Dataset discovery and format detection
- Autoencoder dataset loading
- Supervised learning dataset loading  
- Behavior cloning dataset loading
- Memory-efficient lazy loading

## 🔧 **Integration with Existing Scripts**

The new utilities are designed to be drop-in replacements:

1. **`models/train_autoencoder.py`**: Uses `LazyDataset` for memory efficiency
2. **`optomech/supervised_ml/train_sml_model.py`**: Can migrate to `LazyDataset`
3. **`optomech/imitation_learning/train_bc.py`**: Can use `BehaviorCloningDataset`

## 🚀 **Future Extensions**

- **Replay Buffers**: Memory-efficient experience replay for RL
- **Data Augmentation**: Built-in augmentation transforms
- **Multi-Modal**: Support for different observation types
- **Distributed Loading**: Multi-process data loading optimizations

The modular design makes it easy to extend functionality while maintaining backward compatibility.