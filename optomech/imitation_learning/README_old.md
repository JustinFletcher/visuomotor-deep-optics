# Imitation Learning Framework

A comprehensive framework for behavior cloning training on SA (Simulated Annealing) datasets. This framework allows training neural networks to imitate different types of actions from the SA optimization process.

## Overview

The framework provides:
- **Behavior Cloning**: Train models to predict actions taken by SA
- **Multiple Target Types**: Support for different learning objectives
- **Flexible Architectures**: Various CNN and ResNet-based models
- **Robust Training**: Based on the proven supervised_ml framework
- **Multi-GPU Support**: DataParallel training on multiple GPUs

## Quick Start

### Basic Usage

```bash
# Train a behavior cloning model on SA incremental actions
python train_behavior_cloning.py \
    --dataset-path datasets/sa_dataset \
    --target-type sa_incremental_action \
    --arch il_resnet_gn \
    --batch-size 32 \
    --num-epochs 50
```

### Test the Framework

```bash
# Run a quick test with limited data
python test_bc.py
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
- **What**: Ideal correcting actions for the current state
- **Use Case**: Learn optimal corrections rather than SA decisions
- **Learning Objective**: `model(observation) → perfect_correction`

### 4. `perfect_incremental_action` (Incremental Perfect Actions)
- **What**: Change in perfect actions from previous state
- **Use Case**: Learn incremental perfect corrections
- **Learning Objective**: `model(observation) → perfect_action_delta`

## Model Architectures

### Available Models

- **`il_cnn`**: Standard CNN with multiple conv layers and adaptive pooling
- **`il_resnet`**: ResNet-like architecture with BatchNorm
- **`il_resnet_gn`**: ResNet with GroupNorm (recommended for small batches)
- **`il_simple`**: Lightweight CNN for quick training
- **`il_vanilla`**: Configurable vanilla conv model with minimal parameters

### Architecture Selection

```bash
# Use ResNet with GroupNorm (recommended)
--arch il_resnet_gn

# Use lightweight model for testing
--arch il_vanilla --channel-scale 16 --mlp-scale 64

# Use standard CNN
--arch il_cnn
```

## Training Configuration

### Basic Settings

```bash
python train_behavior_cloning.py \
    --dataset-path datasets/sa_dataset \
    --target-type sa_incremental_action \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --num-epochs 50 \
    --arch il_resnet_gn
```

### Advanced Settings

```bash
python train_behavior_cloning.py \
    --dataset-path datasets/sa_dataset \
    --target-type sa_incremental_action \
    --arch il_resnet_gn \
    --input-crop-size 128 \
    --batch-size 64 \
    --learning-rate 2e-4 \
    --num-epochs 100 \
    --model-save-path saved_models/my_bc_model.pth \
    --max-examples 50000 \
    --seed 42
```

### Resume Training

```bash
# Resume from checkpoint
python train_behavior_cloning.py \
    --resume-from saved_models/bc_checkpoint_epoch_20.pth \
    --dataset-path datasets/sa_dataset
```

## Dataset Format

The framework expects SA datasets with HDF5 or NPZ format containing:

### Required Fields
- `observations`: uint16 arrays (observations after SA action)
- Target field (one of):
  - `sa_action`: SA actions taken
  - `sa_incremental_action`: SA action increments
  - `perfect_action`: Perfect correcting actions
  - `perfect_incremental_action`: Perfect action increments

### Example HDF5 Structure
```
batch_uuid.h5
├── observations [N, C, H, W] (uint16)
├── sa_actions [N, action_dim] (float32)
├── sa_incremental_actions [N, action_dim] (float32)
├── perfect_actions [N, action_dim] (float32)
├── perfect_incremental_actions [N, action_dim] (float32)
├── rewards [N] (float32)
└── ... (other metadata)
```

## Command Line Arguments

### Dataset Settings
- `--dataset-path`: Path to SA dataset directory
- `--target-type`: Target to predict (sa_action, sa_incremental_action, perfect_action, perfect_incremental_action)
- `--max-examples`: Limit dataset size for debugging

### Training Settings
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--num-epochs`: Number of training epochs (default: 50)
- `--device`: Device to use (auto, cuda, mps, cpu)

### Model Settings
- `--arch`: Model architecture (il_cnn, il_resnet, il_resnet_gn, il_simple, il_vanilla)
- `--input-crop-size`: Center crop size for inputs (None for no cropping)

### Output Settings
- `--model-save-path`: Path to save trained model
- `--no-save`: Don't save model
- `--resume-from`: Resume training from checkpoint

## Programming Interface

### Training Function

```python
from train_behavior_cloning import TrainingConfig, train_behavior_cloning

# Create configuration
config = TrainingConfig(
    dataset_path="datasets/sa_dataset",
    target_type="sa_incremental_action",
    batch_size=32,
    learning_rate=1e-4,
    num_epochs=50,
    arch="il_resnet_gn",
    input_crop_size=128,
    model_save_path="saved_models/bc_model.pth"
)

# Start training
train_behavior_cloning(config)
```

### Model Creation

```python
from models import create_model

# Create model
model = create_model(
    arch="il_resnet_gn",
    input_channels=2,
    action_dim=15,
    input_crop_size=128
)
```

## Output and Monitoring

### TensorBoard Logging
- Training and validation losses
- Learning rate schedules
- Logs saved to `saved_models/logs/`

### Model Checkpoints
- Automatic checkpointing every 10 epochs
- Best model saved based on validation loss
- Final model saved at completion

### Training Curves
- Automatic plotting of loss curves
- Saved as PNG in model directory

## Tips and Best Practices

### Target Type Selection
1. **Start with `sa_incremental_action`** - Often most stable and effective
2. **Use `perfect_action`** for learning optimal policies
3. **Try `sa_action`** for direct SA imitation
4. **Experiment with incremental vs. absolute** based on your use case

### Model Architecture
1. **`il_resnet_gn`** - Good default choice, works well with small batches
2. **`il_vanilla`** - Fast training for prototyping
3. **`il_cnn`** - Standard baseline
4. **Input cropping** - Use 128x128 or 256x256 for memory efficiency

### Training Settings
1. **Batch size**: Start with 32, increase if GPU memory allows
2. **Learning rate**: 1e-4 is a good default, try 2e-4 for faster convergence
3. **Epochs**: 50-100 typically sufficient, monitor validation loss
4. **Data splitting**: Default 70/20/10 train/val/test works well

### Debugging
1. **Use `--max-examples 1000`** for quick testing
2. **Start with `il_vanilla`** architecture for fast iteration
3. **Check data loading first** - ensure target type exists in dataset
4. **Monitor GPU memory** - reduce batch size if needed

## File Structure

```
optomech/imitation_learning/
├── __init__.py                    # Module initialization
├── models.py                      # Neural network architectures
├── train_behavior_cloning.py      # Main training script
├── test_bc.py                     # Test/example script
└── README.md                      # This documentation
```

## Integration with SA Pipeline

1. **Generate SA dataset** using `build_sa_dataset.py`
2. **Train behavior cloning model** using this framework
3. **Deploy trained model** for inference in optomech environment
4. **Evaluate performance** against SA and perfect baselines

## Future Extensions

Planned additions to the framework:
- **Multi-objective learning**: Combine behavior cloning with correction learning
- **Sequence modeling**: LSTM/Transformer models for temporal dependencies
- **Domain adaptation**: Transfer learning between different environments
- **Online learning**: Continual learning during deployment
