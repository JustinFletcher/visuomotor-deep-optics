# Imitation Learning Framework

A comprehensive framework for training neural networks to imitate expert behavior, with robust model persistence and component reuse capabilities.

## Features

### 🎯 **Behavior Cloning Training**
- Train models to predict actions from observations
- Support for multiple target types (SA actions, perfect actions, incremental actions)
- Configurable loss functions and optimizers
- Multi-GPU support with DataParallel
- Resume training from checkpoints

### 🏗️ **Modular Architecture System**
- Reusable model components (`VanillaCNN`, `ResNet18GroupNorm`)
- Factory function for easy model creation
- Configurable architectures with consistent interfaces

### 💾 **Codebase-Independent Model Persistence**
- Save models with full metadata and configuration
- Load models without requiring original codebase
- TorchScript export for deployment
- Component extraction for transfer learning

### 🧩 **Component Reuse & Transfer Learning**
- Extract feature extractors from trained models
- Create hybrid models with pretrained components
- Freeze/unfreeze components for fine-tuning
- Model registry for managing trained models

## Quick Start

### 1. Train a Behavior Cloning Model

```bash
# Basic training
python train_behavior_cloning.py \
    --dataset-path datasets/sa_dataset \
    --target-type sa_incremental_action \
    --arch resnet18_gn \
    --num-epochs 50

# Advanced training with custom settings
python train_behavior_cloning.py \
    --dataset-path datasets/sa_dataset \
    --target-type sa_incremental_action \
    --arch vanilla_cnn \
    --input-crop-size 128 \
    --batch-size 64 \
    --learning-rate 1e-3 \
    --num-epochs 100
```

### 2. Load and Use a Trained Model

```python
from model_utils import load_trained_model

# Load a model by ID
model, metadata = load_trained_model("bc_resnet18_gn_sa_incremental_action_1729123456")

# Use the model
import torch
test_input = torch.randn(1, 2, 256, 256)
with torch.no_grad():
    actions = model(test_input)
```

### 3. Create Models with Pretrained Components

```python
from model_utils import create_model_with_pretrained_features

# Create a new model using pretrained feature extractor
model = create_model_with_pretrained_features(
    architecture="vanilla_cnn",
    feature_extractor_model_id="trained_feature_extractor_001",
    model_config={
        'input_channels': 2,
        'action_dim': 30,  # Different output size
        'input_crop_size': 128
    }
)
```

## Model Utilities API

### Core Functions

#### `save_trained_model()`
```python
save_trained_model(
    model=trained_model,
    model_id="my_model_001",
    architecture="resnet18_gn",
    model_config={'input_channels': 2, 'action_dim': 15},
    training_info={'best_val_loss': 0.025},
    task="behavior_cloning",
    example_input=torch.randn(1, 2, 256, 256)
)
```

#### `load_trained_model()`
```python
model, metadata = load_trained_model("my_model_001")
```

#### `ModelRegistry`
```python
from model_utils import ModelRegistry

registry = ModelRegistry("saved_models")
all_models = registry.list_models()
cnn_models = registry.list_models(architecture="vanilla_cnn")
bc_models = registry.list_models(task="behavior_cloning")
```

### Component Extraction

#### Extract Specific Components
```python
from model_utils import ModelLoader

# Load pretrained features into a new model
ModelLoader.load_component(
    model_path="saved_models/pretrained_model.pth",
    component_name="features",
    target_module=new_model.features
)
```

#### Create Hybrid Models
```python
from model_utils import ComponentExtractor

hybrid_model = ComponentExtractor.create_hybrid_model(
    base_architecture="vanilla_cnn",
    pretrained_components={
        'features': "saved_models/feature_extractor.pth",
        'layer1': "saved_models/resnet_model.pth"
    },
    model_config={'input_channels': 2, 'action_dim': 25}
)
```

#### Freeze Components
```python
# Freeze pretrained components for fine-tuning
ComponentExtractor.freeze_components(model, ['features', 'layer1'])
```

## Available Architectures

### `vanilla_cnn`
- Standard CNN with feature extractor and classifier
- Good balance of performance and speed
- Configurable with `channel_scale` and `mlp_scale`

### `resnet18_gn`
- ResNet-18 style architecture with GroupNorm
- Better performance on complex tasks
- Residual connections for deep networks

## Model Persistence Features

### 🏷️ **Full Metadata Preservation**
Every saved model includes:
- Architecture configuration
- Training hyperparameters
- Performance metrics
- Dataset information
- Timestamp and version info

### 🧩 **Component Extraction**
Automatically extracts reusable components:
- `features`: Feature extraction layers
- `classifier`: Classification head
- `layer1`, `layer2`, etc.: Individual ResNet layers
- `conv1`: Initial convolution layer

### ⚡ **TorchScript Export**
- Models automatically exported to TorchScript
- Completely independent of Python/PyTorch source
- Ready for production deployment

### 📚 **Model Registry**
- Centralized tracking of all trained models
- Search and filter by architecture, task, etc.
- Automatic cleanup and management

## Advanced Usage

### Transfer Learning Pipeline

```python
# 1. Train a feature extractor on large dataset
python train_behavior_cloning.py \
    --dataset-path datasets/large_sa_dataset \
    --arch resnet18_gn \
    --model-id feature_extractor_v1

# 2. Create specialized model using pretrained features
model = create_model_with_pretrained_features(
    architecture="vanilla_cnn",
    feature_extractor_model_id="feature_extractor_v1",
    model_config={'action_dim': 45}  # Different task
)

# 3. Fine-tune on specialized dataset
ComponentExtractor.freeze_components(model, ['features'])
# ... train only the classifier ...
```

### Model Deployment

```python
# Load TorchScript version for deployment
from model_utils import ModelLoader

deployed_model = ModelLoader.load_torchscript("saved_models/model_traced.pt")

# This model runs without any source code dependencies
```

### Cross-Project Model Sharing

```python
# Models can be moved between different codebases
# The factory functions and metadata ensure compatibility

# In Project A:
save_trained_model(model, "shared_feature_extractor", ...)

# In Project B (different codebase):
model, metadata = load_trained_model("shared_feature_extractor")
# Works seamlessly!
```

## Target Types

The framework supports four different target types for learning:

### 1. `sa_action` (Direct Action Imitation)
- **What**: Direct SA actions chosen by the optimization algorithm
- **Use Case**: Learn to replicate SA decision-making exactly
- **Learning Objective**: `model(observation) → sa_action`

### 2. `sa_incremental_action` (Default - Incremental SA Actions)  
- **What**: Change in SA actions from previous accepted action
- **Use Case**: Learn incremental adjustments (most common)
- **Learning Objective**: `model(observation) → action_delta`
- **Advantage**: Often more stable and easier to learn

### 3. `perfect_action` (Perfect Correction Imitation)
- **What**: Optimal actions computed from perfect knowledge
- **Use Case**: Learn ideal behavior when available
- **Learning Objective**: `model(observation) → perfect_action`

### 4. `perfect_incremental_action` (Perfect Incremental Actions)
- **What**: Optimal incremental changes
- **Use Case**: Learn perfect incremental adjustments
- **Learning Objective**: `model(observation) → perfect_action_delta`

## File Structure

```
optomech/imitation_learning/
├── train_behavior_cloning.py  # Main training script
├── models.py                  # Model architectures  
├── model_utils.py            # Persistence utilities
├── demo_model_utils.py       # Usage demonstrations
├── test_bc.py                # Unit tests
└── README.md                 # This file

saved_models/                 # Model storage
├── model_registry.json       # Model metadata registry
├── model_001.pth            # Full model with metadata
├── model_001_traced.pt      # TorchScript version
└── ...
```

## Examples

Run the demonstration script to see all features in action:

```bash
python demo_model_utils.py
```

This will show:
- Model saving with full metadata
- Codebase-independent loading
- Component extraction and reuse
- Hybrid model creation
- Registry management
- TorchScript deployment

## Benefits

### 🚀 **Productivity**
- Reuse trained components across projects
- No need to retrain from scratch
- Quick experimentation with hybrid architectures

### 🔒 **Reliability**
- Models work independent of codebase changes
- Full reproducibility with preserved metadata
- Automated component compatibility checking

### 📦 **Deployment Ready**
- TorchScript models for production
- No Python dependencies in deployment
- Consistent interfaces across environments

### 🧪 **Research Friendly**
- Easy model comparison and ablation studies
- Component-level analysis and reuse
- Systematic experiment tracking

## Configuration

### Training Configuration Options

```python
@dataclass
class TrainingConfig:
    dataset_path: str = "datasets/sa_dataset"
    target_type: str = "sa_incremental_action"
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 50
    arch: str = "resnet18_gn"
    input_channels: int = 2
    input_crop_size: int = None
    # ... many more options
```

### Command Line Interface

```bash
python train_behavior_cloning.py --help
```

Shows all available configuration options with descriptions.

## Testing

Run the test suite to verify everything works:

```bash
python test_bc.py
```

This tests:
- Model creation and training
- Saving and loading functionality
- Component extraction
- Dataset handling
- All utility functions
