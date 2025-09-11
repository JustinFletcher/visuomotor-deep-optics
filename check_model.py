#!/usr/bin/env python3
import sys
from pathlib import Path
import torch

# Add path for TrainingConfig import
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "optomech"))

try:
    from optomech.supervised_ml.train_sml_model import TrainingConfig
except ImportError:
    from dataclasses import dataclass
    @dataclass
    class TrainingConfig:
        pass

# Load and inspect the model
checkpoint = torch.load('saved_models/sml_model.pth', map_location='cpu', weights_only=False)
print('Type:', type(checkpoint))
if isinstance(checkpoint, dict):
    print('Keys:', list(checkpoint.keys()))
    for key, value in checkpoint.items():
        print(f"  {key}: {type(value)}")
        if hasattr(value, 'keys'):
            print(f"    {key} keys: {list(value.keys())}")
else:
    print('Not a dict, value:', checkpoint)
