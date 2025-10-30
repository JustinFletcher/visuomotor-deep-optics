# Behavior Cloning Training with Unified Utilities

## Overview

The new behavior cloning training script (`optomech/imitation_learning/train_bc_unified.py`) uses:
- **ResNet-18** based actor model
- **Unified dataset utilities** with log-scaling support
- **256-pixel center cropping** from 512x512 observations
- **TensorBoard logging** for training metrics
- **Environment rollouts** after each validation epoch
- **Incremental control mode** forced for rollouts

## Quick Start

### 1. Test the Setup

First, verify that the dataset and utilities are working correctly:

```bash
python test_bc_setup.py
```

This will:
- Load 100 examples from the dataset
- Verify observation shapes (should be 2x256x256 after cropping)
- Verify action shapes (should match sa_incremental_action dimension)
- Test log-scaling is applied correctly
- Test DataLoader functionality

### 2. Basic Training Run

Train with default settings (recommended for first run):

```bash
python optomech/imitation_learning/train_bc_unified.py \
    --dataset-path datasets/sa_dataset_100k \
    --batch-size 32 \
    --num-epochs 50 \
    --learning-rate 1e-4
```

This will:
- Use ResNet-18 architecture
- Apply log-scaling to 256px center crops
- Train for 50 epochs with Adam optimizer
- Run rollouts with 8 seeds after each validation epoch
- Save results to `runs/bc_run_<timestamp>/`

### 3. Quick Debug Run

For testing/debugging with limited data:

```bash
python optomech/imitation_learning/train_bc_unified.py \
    --dataset-path datasets/sa_dataset_100k \
    --max-examples 1000 \
    --batch-size 16 \
    --num-epochs 5 \
    --no-rollouts
```

This will:
- Only load first 1000 examples
- Train for 5 epochs quickly
- Skip rollouts for faster iteration

### 4. Full Training Run

For production training on full dataset:

```bash
python optomech/imitation_learning/train_bc_unified.py \
    --dataset-path datasets/sa_dataset_100k \
    --batch-size 64 \
    --num-epochs 100 \
    --learning-rate 1e-4 \
    --weight-decay 1e-5 \
    --enable-rollouts \
    --rollout-seeds 16 \
    --rollout-steps 250
```

This will:
- Use full dataset
- Train for 100 epochs
- Run comprehensive rollouts (16 seeds × 250 steps)
- May take several hours depending on hardware

## Key Command-Line Arguments

### Dataset Settings
- `--dataset-path`: Path to SA dataset (default: `datasets/sa_dataset_100k`)
- `--target-action`: Action key to predict (default: `sa_incremental_action`)
- `--max-examples`: Limit dataset size for debugging (optional)

### Model Settings
- `--input-crop-size`: Center crop size in pixels (default: 256)
- `--log-scale` / `--no-log-scale`: Enable/disable log-scaling (default: enabled)
- `--pretrained-encoder`: Path to pre-trained autoencoder model to load encoder from (optional)
- `--freeze-encoder`: Freeze encoder weights for fine-tuning (only train action head)

### Training Settings
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--num-epochs`: Number of training epochs (default: 50)
- `--optimizer`: Optimizer type - `adam` or `adamw` (default: adam)
- `--weight-decay`: Weight decay for regularization (default: 1e-5)
- `--grad-clip`: Gradient clipping threshold (optional)

### Rollout Settings
- `--enable-rollouts` / `--no-rollouts`: Enable/disable rollouts (default: enabled)
- `--rollout-seeds`: Number of rollout seeds (default: 8)
- `--rollout-steps`: Steps per rollout (default: 250)
- `--force-incremental-mode`: Force incremental control mode (default: True)

### Hardware Settings
- `--device`: Device to use - `auto`, `cuda`, `mps`, or `cpu` (default: auto)
- `--num-workers`: Number of data loading workers (default: 4)

### Output Settings
- `--no-save`: Don't save trained model
- `--seed`: Random seed for reproducibility (default: 42)

## Output Files

Training creates a timestamped directory in `runs/bc_run_<timestamp>_<uuid>/`:

```
runs/bc_run_20240101_120000_abc12345/
├── config.json                    # Training configuration
├── bc_model_best.pth             # Best model checkpoint (lowest val loss)
├── bc_model_final.pth            # Final model after all epochs
├── training_curves.png           # Training/validation loss plot
├── training_curves_final.png     # Final plot
├── events.out.tfevents.*         # TensorBoard logs
├── rollouts_epoch_1/             # Rollout results for epoch 1
│   ├── rollout_results/
│   │   ├── reward_curves.png
│   │   ├── reward_statistics.json
│   │   └── ...
├── rollouts_epoch_2/             # Rollout results for epoch 2
└── ...
```

## Monitoring Training

### TensorBoard

Monitor training in real-time:

```bash
tensorboard --logdir runs
```

Then open http://localhost:6006 in your browser.

TensorBoard shows:
- Training and validation loss curves
- Validation MAE statistics (mean, median, std)
- Learning rate schedule
- Rollout episode rewards per epoch
- Rollout success rates

### Training Output

The script prints detailed progress:

```
Epoch   1/50 | Train Loss: 0.024531 | Val Loss: 0.023142 | Val MAE: 0.098234 | Time: 45.2s
  🎯 Running rollout instrumentation...
  ✅ Rollout completed! Episode reward: -12.3456 ± 2.1234
Epoch   2/50 | Train Loss: 0.021234 | Val Loss: 0.020987 | Val MAE: 0.092341 | Time: 44.8s
  🎯 Running rollout instrumentation...
  ✅ Rollout completed! Episode reward: -10.2345 ± 1.8765
...
```

## Advanced Usage

### Using Pre-trained Encoders

You can use encoders from pre-trained autoencoder models to improve training efficiency and performance. This is particularly useful for transfer learning:

#### 1. Train an Autoencoder First

```bash
# Train an autoencoder on your observation dataset
python models/train_autoencoder.py \
    --dataset-path datasets/sa_dataset_100k \
    --arch autoencoder_resnet \
    --input-size 256 \
    --log-scale \
    --num-epochs 100 \
    --batch-size 32
```

This will save the trained autoencoder to `runs/autoencoder_run_*/autoencoder_best.pth`.

#### 2. Use the Encoder in Behavior Cloning

**Fine-tune the entire model (encoder + action head):**

```bash
python optomech/imitation_learning/train_bc_unified.py \
    --dataset-path datasets/sa_dataset_100k \
    --pretrained-encoder runs/autoencoder_run_20241026_120000_abc123/autoencoder_best.pth \
    --batch-size 32 \
    --num-epochs 50
```

**Freeze encoder and only train action head:**

```bash
python optomech/imitation_learning/train_bc_unified.py \
    --dataset-path datasets/sa_dataset_100k \
    --pretrained-encoder runs/autoencoder_run_20241026_120000_abc123/autoencoder_best.pth \
    --freeze-encoder \
    --batch-size 64 \
    --learning-rate 1e-3 \
    --num-epochs 30
```

Freezing the encoder is useful when:
- The encoder is already well-trained on similar data
- You have limited computational resources
- You want faster training (fewer parameters to update)
- You want to prevent overfitting on small datasets

**Benefits of pre-trained encoders:**
- **Faster convergence**: The encoder already knows how to extract useful features
- **Better generalization**: Pre-training on reconstruction helps learn robust representations
- **Lower data requirements**: Less behavior cloning data needed for good performance
- **Transfer learning**: Can use encoders trained on different but related tasks

#### 3. Two-Stage Training

For best results, consider a two-stage approach:

**Stage 1: Pre-train encoder with frozen action head**
```bash
python optomech/imitation_learning/train_bc_unified.py \
    --pretrained-encoder autoencoder_best.pth \
    --freeze-encoder \
    --learning-rate 1e-3 \
    --num-epochs 20 \
    --no-rollouts
```

**Stage 2: Fine-tune everything together**
```bash
python optomech/imitation_learning/train_bc_unified.py \
    --pretrained-encoder runs/bc_run_stage1/bc_model_best.pth \
    --learning-rate 1e-5 \
    --num-epochs 30 \
    --enable-rollouts
```

### Custom Target Actions

To train on different action types:

```bash
# Train on perfect incremental actions
python optomech/imitation_learning/train_bc_unified.py \
    --target-action perfect_incremental_action

# Train on absolute actions
python optomech/imitation_learning/train_bc_unified.py \
    --target-action sa_action
```

### Disable Log-Scaling

If you want to train without log-scaling (not recommended):

```bash
python optomech/imitation_learning/train_bc_unified.py \
    --no-log-scale
```

### Different Crop Sizes

To experiment with different input sizes:

```bash
python optomech/imitation_learning/train_bc_unified.py \
    --input-crop-size 128  # Smaller, faster training

python optomech/imitation_learning/train_bc_unified.py \
    --input-crop-size 512  # Full resolution (no cropping)
```

### Gradient Clipping

For more stable training:

```bash
python optomech/imitation_learning/train_bc_unified.py \
    --grad-clip 1.0
```

## Troubleshooting

### Out of Memory

If you encounter OOM errors:

1. Reduce batch size:
   ```bash
   --batch-size 16
   ```

2. Reduce number of workers:
   ```bash
   --num-workers 2
   ```

3. Use smaller crop size:
   ```bash
   --input-crop-size 128
   ```

### Slow Data Loading

If data loading is slow:

1. Increase workers (if you have CPU cores available):
   ```bash
   --num-workers 8
   ```

2. Disable pin_memory for CPU training:
   ```bash
   --device cpu
   ```

### Rollout Errors

If rollouts fail:

1. Disable rollouts for pure supervised learning:
   ```bash
   --no-rollouts
   ```

2. Check environment configuration in the error message

3. Verify the dataset's environment configuration matches the rollout environment

## Comparison with Old Script

The new script (`train_bc_unified.py`) vs old script (`train_behavior_cloning.py`):

| Feature | Old Script | New Script |
|---------|-----------|------------|
| Dataset Loading | Custom SADataset class | Unified LazyDataset |
| Log-Scaling | Not supported | Automatic with `--log-scale` |
| Model Architecture | Custom models (models.py empty) | ResNet-18 from torchvision |
| Memory Efficiency | Loads all data | Lazy loading on-demand |
| Cropping | Manual implementation | Unified transform pipeline |
| Rollout Integration | Not implemented | Automatic after validation |
| Code Reuse | Duplicated data handlers | Shares utilities with SML/AE |

## Next Steps

After training:

1. **Analyze Results**: Check TensorBoard for training curves and rollout performance
2. **Compare Models**: Load and compare multiple trained models
3. **Full Rollouts**: Run comprehensive rollouts on best model
4. **Deployment**: Use `bc_model_best.pth` for deployment or fine-tuning
