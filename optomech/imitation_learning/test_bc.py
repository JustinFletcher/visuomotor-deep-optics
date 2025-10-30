#!/usr/bin/env python3
"""
Example script to test behavior cloning training.
This demonstrates how to use the behavior cloning framework.
"""

import sys
from pathlib import Path

# Add the imitation learning module to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from train_behavior_cloning import TrainingConfig, train_behavior_cloning


def main():
    """Run a quick test of behavior cloning training"""
    
    # Configuration for testing
    config = TrainingConfig(
        dataset_path="test_datasets/local_sa",  # Adjust path as needed
        target_type="sa_incremental_action",   # Default target
        batch_size=16,                         # Smaller for testing
        learning_rate=1e-4,
        num_epochs=5,                          # Few epochs for testing
        device="auto",
        arch="il_vanilla",                     # Lightweight model for testing
        channel_scale=16,
        mlp_scale=64,
        input_crop_size=128,                   # Crop to 128x128
        model_save_path="saved_models/test_bc_model.pth",
        save_model=True,
        plot_losses=True,
        max_examples=1000,                     # Limit for testing
        seed=42,
        
        # Training settings
        loss_type="mse",
        optimizer="adam",
        weight_decay=1e-5,
        num_workers=2,
        pin_memory=True
    )
    
    print("🧪 Running Behavior Cloning Test")
    print("=" * 50)
    
    # Start training
    train_behavior_cloning(config)
    
    print("\n✅ Test completed!")


if __name__ == "__main__":
    main()
