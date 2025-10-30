# Behavior Cloning Training with Pre-trained Encoder Support

## Summary of Changes

### 1. Model Architecture (`models/models.py`)

Added `ResNet18Actor` class to the models module for reusability:

**Key Features:**
- ResNet-18 based architecture for behavior cloning
- Support for loading pre-trained encoders from autoencoder models
- Optional encoder freezing for fine-tuning
- Automatic encoder extraction from AutoEncoderCNN or AutoEncoderResNet
- 2-channel input support for complex image data
- Configurable action prediction head with dropout

**Usage:**
```python
from models.models import ResNet18Actor

# From scratch
model = ResNet18Actor(input_channels=2, action_dim=4)

# With pre-trained encoder
model = ResNet18Actor(
    input_channels=2, 
    action_dim=4,
    pretrained_encoder_path="runs/autoencoder_run_*/autoencoder_best.pth",
    freeze_encoder=False  # Set True to freeze encoder weights
)
```

**Methods:**
- `forward(x)`: Standard forward pass through encoder → action head
- `_load_pretrained_encoder(path)`: Load encoder from saved autoencoder
- `_freeze_encoder()`: Freeze encoder weights for fine-tuning
- `unfreeze_encoder()`: Unfreeze encoder weights during training

### 2. Training Script Updates (`optomech/imitation_learning/train_bc_unified.py`)

**Removed:**
- Local `ResNet18Actor` class definition (moved to models module)

**Added:**
- Import from `models.models`
- Pre-trained encoder configuration in `TrainingConfig`:
  - `pretrained_encoder_path`: Path to saved autoencoder model
  - `freeze_encoder`: Boolean flag for freezing encoder weights
- Command-line arguments:
  - `--pretrained-encoder`: Path to pre-trained encoder
  - `--freeze-encoder`: Flag to freeze encoder weights
- Enhanced model creation with encoder loading support
- Parameter counting that distinguishes frozen vs trainable parameters

**Example Commands:**

Train from scratch:
```bash
python optomech/imitation_learning/train_bc_unified.py \
    --dataset-path datasets/sa_dataset_100k \
    --batch-size 32 \
    --num-epochs 50
```

Train with pre-trained encoder (fine-tuning):
```bash
python optomech/imitation_learning/train_bc_unified.py \
    --dataset-path datasets/sa_dataset_100k \
    --pretrained-encoder runs/autoencoder_run_*/autoencoder_best.pth \
    --batch-size 32 \
    --num-epochs 50
```

Train with frozen encoder (action head only):
```bash
python optomech/imitation_learning/train_bc_unified.py \
    --dataset-path datasets/sa_dataset_100k \
    --pretrained-encoder runs/autoencoder_run_*/autoencoder_best.pth \
    --freeze-encoder \
    --learning-rate 1e-3 \
    --batch-size 64 \
    --num-epochs 30
```

### 3. Unified Utilities Updates (`utils/datasets.py`)

**Enhanced `LazyDataset` class:**
- Added `target_action_key` parameter to support custom action keys
- Now properly loads `sa_incremental_action` or any other action type
- Updated all loading methods (`_load_hdf5_item`, `_load_npz_item`, `_load_json_item`)
- Updated index building to use custom action keys

**Usage:**
```python
from utils.datasets import LazyDataset

dataset = LazyDataset(
    dataset_path="datasets/sa_dataset_100k",
    task_type='behavior_cloning',
    input_crop_size=256,
    log_scale=True,
    target_action_key='sa_incremental_action'  # Custom action key
)
```

### 4. Documentation (`optomech/imitation_learning/BC_TRAINING_GUIDE.md`)

**Added comprehensive section on pre-trained encoders:**
- Step-by-step guide for training autoencoder first
- Usage examples for fine-tuning vs frozen encoder
- Two-stage training workflow
- Benefits of pre-trained encoders
- Command-line argument documentation

**Topics covered:**
- When to freeze encoder vs fine-tune
- Transfer learning best practices
- Computational and data efficiency benefits
- Two-stage training approach for optimal results

### 5. Test Script Updates (`test_bc_setup.py`)

**Enhanced testing:**
- Added model creation test using `ResNet18Actor`
- Added forward pass test to verify model output shapes
- Imports from models module
- Tests complete pipeline: data → model → predictions

**Tests performed:**
1. Dataset loading with custom action keys
2. Single item loading and verification
3. DataLoader functionality
4. Log-scaling verification
5. Model creation with correct dimensions
6. Forward pass through model
7. Output shape validation

### 6. Package Exports (`models/__init__.py`)

**Updated exports:**
- Added `ResNet18Actor` to `__all__` list
- Added import from `models.models`
- Now available via `from models import ResNet18Actor`

## Benefits

### 1. **Reusability**
- `ResNet18Actor` can be imported and used in other scripts
- No code duplication across different training paradigms
- Consistent model architecture across experiments

### 2. **Transfer Learning**
- Can leverage pre-trained autoencoder encoders
- Faster convergence with better representations
- Lower data requirements for good performance

### 3. **Flexibility**
- Choose between training from scratch or using pre-trained encoder
- Option to freeze encoder for faster training / less overfitting
- Two-stage training for optimal performance

### 4. **Consistency**
- Same model code used everywhere (no local redefinitions)
- Unified utilities used across all training types
- Standardized pre-training → fine-tuning workflow

## File Structure

```
visuomotor-deep-optics/
├── models/
│   ├── __init__.py                      # Added ResNet18Actor export
│   └── models.py                        # Added ResNet18Actor class with encoder loading
├── optomech/
│   └── imitation_learning/
│       ├── train_bc_unified.py          # Updated to use models.ResNet18Actor
│       └── BC_TRAINING_GUIDE.md         # Added pre-trained encoder documentation
├── utils/
│   └── datasets.py                      # Added target_action_key parameter
└── test_bc_setup.py                     # Enhanced with model testing
```

## Migration Path

### For existing users:

1. **Update imports:**
   ```python
   # Old (local definition)
   class ResNet18Actor(nn.Module):
       ...
   
   # New (from models module)
   from models.models import ResNet18Actor
   ```

2. **Use pre-trained encoders:**
   ```bash
   # Train autoencoder first
   python models/train_autoencoder.py --dataset-path datasets/sa_dataset_100k --log-scale
   
   # Then use in BC training
   python optomech/imitation_learning/train_bc_unified.py \
       --pretrained-encoder runs/autoencoder_run_*/autoencoder_best.pth
   ```

3. **Custom action keys:**
   ```python
   # Now supports any action key from dataset
   dataset = LazyDataset(
       ...,
       target_action_key='sa_incremental_action'  # or 'perfect_action', etc.
   )
   ```

## Next Steps

1. Test the setup with `python test_bc_setup.py`
2. Train an autoencoder on your dataset
3. Use the pre-trained encoder for behavior cloning
4. Compare performance: scratch vs pre-trained vs frozen
5. Document best practices based on experimental results
