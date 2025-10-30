# Models Package

A comprehensive package for neural network architectures and model management utilities, designed for codebase-independent persistence and component reuse.

## Overview

This package provides:
- **Reusable Neural Network Architectures**: Well-t### Testing

The package includes comprehensive testing utilities. Run tests from the project root:

```python
# Test model creation
from models import create_model
model = create_model("vanilla_cnn", input_channels=2, action_dim=15)

# Test autoencoder
autoencoder = create_model("autoencoder_cnn", input_channels=2, latent_dim=256)

# Test saving/loading
from models.model_utils import save_trained_model, load_trained_model
# ... test save/load cycle ...
```

### Autoencoder Training

Train autoencoders for unsupervised representation learning:

```bash
# Basic autoencoder training
poetry run python -m models.train_autoencoder \
    --dataset-path datasets/sml_100k_dataset \
    --arch autoencoder_cnn \
    --latent-dim 256 \
    --num-epochs 100

# Advanced autoencoder training
poetry run python -m models.train_autoencoder \
    --dataset-path datasets/sml_100k_dataset \
    --arch autoencoder_resnet \
    --latent-dim 512 \
    --batch-size 64 \
    --learning-rate 1e-3 \
    --loss-function mae \
    --optimizer adamw \
    --input-crop-size 128
```

The autoencoder training provides:
- **Separable Components**: Encoder and decoder can be used independently
- **Multiple Architectures**: CNN and ResNet-based options
- **Flexible Loss Functions**: MSE, MAE, Smooth L1, Huber loss
- **Reconstruction Visualization**: Sample outputs saved during training
- **Full Model Persistence**: Save encoders for later reuse in other models ResNet models
- **Model Persistence Utilities**: Save/load models with full metadata
- **Component Extraction**: Reuse parts of trained models
- **Transfer Learning Support**: Create hybrid models with pretrained components
- **Model Registry**: Systematic tracking and management of trained models
- **Deployment Ready**: TorchScript export for production use

## Quick Start

### Import the Package

```python
# Import specific components
from models import VanillaCNN, ResNet18GroupNorm, create_model
from models.model_utils import save_trained_model, load_trained_model

# Or import everything
import models
```

### Create Models

```python
from models import create_model

# Create a vanilla CNN
model = create_model(
    arch="vanilla_cnn",
    input_channels=2,
    action_dim=15,
    input_crop_size=128
)

# Create a ResNet with GroupNorm
model = create_model(
    arch="resnet18_gn", 
    input_channels=2,
    action_dim=15
)

# Create an autoencoder
autoencoder = create_model(
    arch="autoencoder_cnn",
    input_channels=2,
    latent_dim=256
)

# Use separable encoder/decoder
import torch
test_input = torch.randn(1, 2, 256, 256)
latent = autoencoder.encode(test_input)          # Get latent representation
reconstruction = autoencoder.decode(latent)      # Reconstruct from latent
full_recon, full_latent = autoencoder(test_input) # Full forward pass
```

### Save and Load Models

```python
from models.model_utils import save_trained_model, load_trained_model

# Save a trained model with full metadata
save_trained_model(
    model=trained_model,
    model_id="my_feature_extractor_v1",
    architecture="resnet18_gn",
    model_config={'input_channels': 2, 'action_dim': 15},
    training_info={'best_val_loss': 0.025},
    task="feature_extraction"
)

# Load the model (works even if codebase changes)
model, metadata = load_trained_model("my_feature_extractor_v1")
```

### Transfer Learning

```python
from models.model_utils import create_model_with_pretrained_features

# Create new model with pretrained feature extractor
model = create_model_with_pretrained_features(
    architecture="vanilla_cnn",
    feature_extractor_model_id="my_feature_extractor_v1",
    model_config={'action_dim': 30}  # Different output size
)
```

## Package Structure

```
models/
├── __init__.py           # Package interface and exports
├── models.py             # Neural network architectures
├── model_utils.py        # Persistence and management utilities
├── train_autoencoder.py  # Autoencoder training script
├── demo_model_utils.py   # Demonstration script
└── README.md             # This file
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

### `autoencoder_cnn`
- Convolutional autoencoder for unsupervised representation learning
- Separable encoder and decoder components
- Configurable latent dimension
- Suitable for image reconstruction and feature learning

### `autoencoder_resnet`
- ResNet-based autoencoder with GroupNorm
- More powerful architecture for complex data
- Separable encoder and decoder components
- Better reconstruction quality on challenging datasets

### Factory Function
```python
create_model(arch, input_channels, action_dim, **kwargs)
```
Supported architectures: `"vanilla_cnn"`, `"resnet18_gn"`

## Model Utilities

### Core Classes

- **`ModelSaver`**: Save models with full metadata and component extraction
- **`ModelLoader`**: Load models independently of original codebase
- **`ModelRegistry`**: Track and manage all trained models
- **`ComponentExtractor`**: Create hybrid models with pretrained components

### Convenience Functions

- **`save_trained_model()`**: Easy model saving with metadata
- **`load_trained_model()`**: Load models by ID
- **`create_model_with_pretrained_features()`**: Transfer learning helper

## Features

### 🔒 **Codebase Independence**
- Models saved with factory function info
- Recreate models without original source code
- Version-independent loading

### 🧩 **Component Reuse**
- Extract feature extractors, classifiers, layers
- Mix and match components from different models
- Freeze/unfreeze for fine-tuning

### 📚 **Model Registry**
- Centralized tracking of all models
- Search by architecture, task, performance
- Automatic metadata preservation

### ⚡ **Production Ready**
- TorchScript export for deployment
- No Python dependencies in production
- Optimized inference models

### 🎯 **Transfer Learning**
- Pretrained component integration
- Hybrid model creation
- Progressive fine-tuning support

## Usage Examples

### Basic Model Training and Saving

```python
import torch
from models import create_model
from models.model_utils import save_trained_model

# Create and train model
model = create_model("resnet18_gn", input_channels=2, action_dim=15)
# ... training code ...

# Save with full metadata
save_trained_model(
    model=model,
    model_id="my_model_001", 
    architecture="resnet18_gn",
    model_config={'input_channels': 2, 'action_dim': 15},
    training_info={'epochs': 50, 'best_loss': 0.025},
    task="behavior_cloning"
)
```

### Advanced Transfer Learning

```python
from models.model_utils import ModelRegistry, ComponentExtractor

# List available models
registry = ModelRegistry()
feature_models = registry.list_models(task="feature_extraction")

# Create hybrid model
hybrid = ComponentExtractor.create_hybrid_model(
    base_architecture="vanilla_cnn",
    pretrained_components={
        'features': f"saved_models/{feature_models[0]}.pth"
    },
    model_config={'input_channels': 2, 'action_dim': 25}
)

# Freeze pretrained parts
ComponentExtractor.freeze_components(hybrid, ['features'])
```

### Production Deployment

```python
from models.model_utils import ModelLoader

# Load TorchScript version for deployment
deployed_model = ModelLoader.load_torchscript("saved_models/model_traced.pt")

# This runs without any Python source dependencies
prediction = deployed_model(input_tensor)
```

## Integration with Training Scripts

The package is designed to integrate seamlessly with training scripts:

```python
# In your training script
from models import create_model
from models.model_utils import save_trained_model

# Create model
model = create_model(config.arch, **config.model_params)

# Train model
# ... training loop ...

# Save with metadata
save_trained_model(
    model=model,
    model_id=f"experiment_{timestamp}",
    architecture=config.arch,
    model_config=config.model_params,
    training_info=training_results,
    task=config.task_name
)
```

## Testing

The package includes comprehensive testing utilities. Run tests from the project root:

```python
# Test model creation
from models import create_model
model = create_model("vanilla_cnn", input_channels=2, action_dim=15)

# Test saving/loading
from models.model_utils import save_trained_model, load_trained_model
# ... test save/load cycle ...
```

## Best Practices

### Model Naming
- Use descriptive model IDs: `"feature_extractor_resnet_v2"`
- Include version numbers for iterations
- Use consistent naming schemes across projects

### Metadata Preservation
- Always include training configuration
- Save performance metrics
- Document dataset and preprocessing info

### Component Organization
- Design models with extractable components
- Use meaningful component names
- Test component compatibility

### Transfer Learning
- Start with pretrained features when possible
- Freeze appropriate layers during fine-tuning
- Validate hybrid model performance

This package provides a robust foundation for scalable machine learning projects with reusable, version-independent model management.
