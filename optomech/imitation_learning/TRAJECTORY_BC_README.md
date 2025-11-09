# Trajectory-Based Behavior Cloning with LSTM

## Overview

This directory now contains **two** behavior cloning training systems:

1. **`train_behavior_cloning.py`** - Single-transition BC using ResNet18Actor
2. **`train_trajectory_behavior_cloning.py`** - Trajectory-based BC using ResNet18LSTMActor (NEW)

The trajectory-based system learns from sequences of observations and actions, capturing temporal dependencies through an LSTM network with Truncated Backpropagation Through Time (TBPTT).

## Architecture

### ResNet18LSTMActor

The new model architecture consists of three main components:

```
Input: [batch, seq_len, channels, height, width]
    ↓
Encoder (ResNet-18 or pretrained AutoEncoder)
    → Processes each frame independently
    → Output: [batch, seq_len, feature_dim]
    ↓
LSTM (1 or more layers)
    → Captures temporal dependencies
    → Output: [batch, seq_len, lstm_hidden_dim]
    ↓
Action Head (MLP)
    → Predicts actions for each timestep
    → Output: [batch, seq_len, action_dim]
```

**Key Features:**
- **Encoder reuse**: Can load pretrained encoders from autoencoder models
- **Encoder freezing**: Option to freeze encoder and only train LSTM + action head
- **LSTM configuration**: Configurable hidden dim, num layers, and dropout
- **Hidden state management**: Proper initialization and propagation for TBPTT

## TrajectoryDataset

New dataset class that creates sliding window sequences from episodes:

```python
TrajectoryDataset(
    dataset_path="datasets/sa_dataset_100k",
    sequence_length=20,          # Length of each sequence
    stride=10,                   # Overlap between sequences (stride < length)
    input_crop_size=256,
    log_scale=True,
    target_action_key="sa_incremental_actions"
)
```

**Features:**
- Sliding window extraction from episodes
- Configurable sequence length and stride
- Returns (observations, actions, mask) tuples
- Mask supports variable-length sequences (currently all full-length)

## Truncated Backpropagation Through Time (TBPTT)

For memory-efficient training on long sequences:

```python
# Full sequence (default)
tbptt_chunk_size = None  # Process entire sequence at once

# TBPTT (memory-efficient)
tbptt_chunk_size = 10    # Process sequence in chunks of 10 timesteps
```

**How it works:**
1. Split sequence into chunks of size `tbptt_chunk_size`
2. Forward pass through each chunk sequentially
3. Detach hidden state between chunks (truncates backprop)
4. Average loss across chunks and backpropagate

**When to use:**
- Long sequences (>50 timesteps)
- Limited GPU memory
- Trade-off: Slightly less accurate gradients for lower memory usage

## Training Configuration

### Key Parameters

**LSTM Architecture:**
```bash
--sequence-length 20          # Length of trajectory sequences
--sequence-stride 10          # Stride for sliding window
--lstm-hidden-dim 256         # LSTM hidden dimension
--lstm-num-layers 1           # Number of LSTM layers
--lstm-dropout 0.0            # Dropout between LSTM layers
--tbptt-chunk-size 20         # TBPTT chunk size (None = full sequence)
```

**Training Settings:**
```bash
--batch-size 16               # Smaller batch size for sequences
--learning-rate 1e-4
--grad-clip 1.0               # Important for LSTM stability
--num-epochs 50
```

**Encoder Settings:**
```bash
--pretrained-encoder path/to/encoder.pth
--freeze-encoder              # Freeze encoder weights
```

**Dataset Pruning (inherited from single-transition BC):**
```bash
--prune-dataset
--prune-l2-threshold 0.1
--prune-keep-fraction 0.1     # Keep 10% of below-threshold samples
```

## Usage Examples

### Basic Training

```bash
python optomech/imitation_learning/train_trajectory_behavior_cloning.py \
    --dataset-path datasets/sa_dataset_100k \
    --sequence-length 20 \
    --batch-size 16 \
    --num-epochs 50
```

### With Pretrained Encoder

```bash
python optomech/imitation_learning/train_trajectory_behavior_cloning.py \
    --dataset-path datasets/sa_dataset_100k \
    --pretrained-encoder saved_models/autoencoder_best.pth \
    --freeze-encoder \
    --sequence-length 20 \
    --lstm-hidden-dim 512 \
    --lstm-num-layers 2
```

### Memory-Efficient TBPTT

```bash
python optomech/imitation_learning/train_trajectory_behavior_cloning.py \
    --dataset-path datasets/sa_dataset_100k \
    --sequence-length 50 \
    --tbptt-chunk-size 10 \
    --batch-size 8
```

### With Dataset Pruning

```bash
python optomech/imitation_learning/train_trajectory_behavior_cloning.py \
    --dataset-path datasets/sa_dataset_100k \
    --prune-dataset \
    --prune-l2-threshold 0.1 \
    --prune-keep-fraction 0.1 \
    --sequence-length 20
```

## Differences from Single-Transition BC

| Feature | Single-Transition BC | Trajectory BC |
|---------|---------------------|---------------|
| **Model** | ResNet18Actor | ResNet18LSTMActor |
| **Dataset** | LazyDataset / InMemoryDataset | TrajectoryDataset |
| **Input** | Single frame | Sequence of frames |
| **Output** | Single action | Sequence of actions |
| **Memory** | Lower | Higher (sequences) |
| **Temporal modeling** | None | LSTM with hidden state |
| **Training** | Standard backprop | TBPTT support |
| **Batch size** | 32 (typical) | 16 (typical, due to sequences) |

## Balanced Sampling Support

Both training scripts support balanced sampling by target action L2 norm:

- 100 linear bins from min to max L2
- Inverse frequency weighting with temperature control
- Applied to train, validation, and test sets
- Controlled by `use_balanced_sampling` flag in code (currently True)

**Note:** Balanced sampling code is in both scripts but needs to be enabled/disabled by modifying the `use_balanced_sampling` variable in the training function.

## Rollout Support

**Current status:** Disabled by default for trajectory training

The original rollout instrumentation expects single-frame inputs, not sequences. To enable rollouts with the LSTM model, you would need to:

1. Maintain hidden state across environment steps
2. Either:
   - Use single-frame mode (hidden state carries temporal info)
   - Buffer observations to create sequences on-the-fly

Set `--enable-rollouts` flag to enable (experimental).

## Model Checkpoints

Checkpoints include:
```python
{
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'epoch': ...,
    'train_loss': ...,
    'val_loss': ...,
    'config': ...  # Full TrainingConfig with LSTM parameters
}
```

## Code Organization

**Added Files:**
- `optomech/imitation_learning/train_trajectory_behavior_cloning.py` - Main training script

**Modified Files:**
- `models/models.py` - Added `ResNet18LSTMActor` class
- `utils/datasets.py` - Added `TrajectoryDataset` class

**Unchanged:**
- All existing training scripts continue to work
- Original `ResNet18Actor` unchanged
- All other models unchanged

## Testing

Verify the system works:

```bash
# Small test run
python optomech/imitation_learning/train_trajectory_behavior_cloning.py \
    --dataset-path datasets/sa_dataset_100k \
    --max-examples 1000 \
    --sequence-length 10 \
    --batch-size 4 \
    --num-epochs 2
```

Check for:
- [ ] Dataset loads successfully
- [ ] Sequences have correct shape
- [ ] Model forward pass works
- [ ] Training loop runs
- [ ] Loss decreases
- [ ] Checkpoints save correctly
- [ ] TensorBoard logs properly

## Future Improvements

Potential enhancements:

1. **Variable-length sequences**: Proper padding and masking for different episode lengths
2. **Attention mechanism**: Replace or augment LSTM with self-attention
3. **Bidirectional LSTM**: For offline learning (not suitable for online deployment)
4. **Multi-step prediction**: Predict multiple future actions given sequence
5. **Rollout integration**: Adapt rollout system for sequence-based models
6. **Sequence sampling strategies**: Beyond simple sliding window

## Troubleshooting

**Out of memory:**
- Reduce `--batch-size`
- Reduce `--sequence-length`
- Use `--tbptt-chunk-size` for TBPTT
- Reduce `--lstm-hidden-dim` or `--lstm-num-layers`

**Training unstable:**
- Ensure `--grad-clip 1.0` is set
- Try lower learning rate
- Check for NaN losses (may need smaller learning rate)

**Poor performance:**
- Try longer sequences (`--sequence-length`)
- Increase LSTM capacity (`--lstm-hidden-dim`, `--lstm-num-layers`)
- Use pretrained encoder
- Check if dataset has sufficient temporal variation

## References

- Original BC implementation: `train_behavior_cloning.py`
- Model definitions: `models/models.py`
- Dataset utilities: `utils/datasets.py`
