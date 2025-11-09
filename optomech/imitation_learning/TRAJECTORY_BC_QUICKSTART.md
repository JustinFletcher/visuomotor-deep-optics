# Quick Start: Trajectory-Based BC Training

## What Changed?

### New Files
1. **`optomech/imitation_learning/train_trajectory_behavior_cloning.py`** - LSTM-based trajectory training
2. **`optomech/imitation_learning/TRAJECTORY_BC_README.md`** - Full documentation

### Modified Files  
1. **`models/models.py`** - Added `ResNet18LSTMActor` class (line ~865)
2. **`utils/datasets.py`** - Added `TrajectoryDataset` class (line ~670)

### Unchanged
✅ All existing code continues to work
✅ Original `train_behavior_cloning.py` unchanged
✅ Original `ResNet18Actor` unchanged

## Quick Training Command

```bash
# Basic trajectory BC training
python optomech/imitation_learning/train_trajectory_behavior_cloning.py \
    --dataset-path datasets/sa_dataset_100k \
    --target-action sa_incremental_actions \
    --sequence-length 20 \
    --sequence-stride 10 \
    --batch-size 16 \
    --lstm-hidden-dim 256 \
    --lstm-num-layers 1 \
    --grad-clip 1.0 \
    --num-epochs 50
```

## Key Differences from Original BC

| Parameter | Original BC | Trajectory BC |
|-----------|-------------|---------------|
| Model | `ResNet18Actor` | `ResNet18LSTMActor` |
| Input shape | `[B, C, H, W]` | `[B, T, C, H, W]` |
| Output shape | `[B, A]` | `[B, T, A]` |
| Batch size | 32 | 16 (due to sequences) |
| Grad clip | None | 1.0 (important!) |
| New params | - | `--sequence-length`, `--lstm-*`, `--tbptt-*` |

## Architecture Overview

```
Observation Sequence → ResNet Encoder → LSTM → Action Head → Action Sequence
   [B, T, C, H, W]    [B, T, feature]  [B, T, hidden]  [B, T, A]
```

## Important Notes

1. **Gradient Clipping**: Always use `--grad-clip 1.0` for LSTM stability
2. **Batch Size**: Sequences use more memory, reduce from 32 to 16 or lower
3. **Sequence Length**: Start with 20, adjust based on task horizon
4. **TBPTT**: Use `--tbptt-chunk-size 10` if you run out of memory
5. **Pretrained Encoder**: Same as original BC, use `--pretrained-encoder path`

## Testing Your Setup

```bash
# Quick test (2 minutes)
python optomech/imitation_learning/train_trajectory_behavior_cloning.py \
    --dataset-path datasets/sa_dataset_100k \
    --max-examples 1000 \
    --sequence-length 10 \
    --batch-size 4 \
    --num-epochs 2 \
    --no-rollouts
```

Expected output:
```
📚 Loading trajectory dataset from datasets/sa_dataset_100k
📊 TrajectoryDataset: 100 sequences of length 10
✅ Loaded 100 trajectory sequences
🏗️  Creating ResNet-18 + LSTM actor model...
🚀 Starting training...
```

## Common Issues

**Out of Memory?**
```bash
--batch-size 8          # Reduce batch size
--sequence-length 15    # Reduce sequence length  
--tbptt-chunk-size 10   # Enable TBPTT
```

**Training Unstable?**
```bash
--grad-clip 1.0         # Ensure grad clipping enabled
--learning-rate 5e-5    # Try lower learning rate
```

## Next Steps

See `TRAJECTORY_BC_README.md` for:
- Full parameter documentation
- Advanced training configurations
- Balanced sampling setup
- Troubleshooting guide
- Architecture details
