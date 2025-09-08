# Utils Directory

This directory contains utility scripts for testing and validating the visuomotor deep optics pipeline.

## Files

### `test_training.py`
**Primary ML Pipeline Validation Script**
- Tests complete dataset loading and partitioning workflow
- Validates episode-level train/val/test splits (70/15/15)
- Extracts individual transitions for ML training
- Tests PyTorch DataLoader integration and batching
- Validates training steps with perfect_action and best_action targets
- **Usage**: Run from project root: `poetry run python utils/test_training.py`

### `test_dataset_structure.py` 
**Dataset Structure Validation Script**
- Validates episode file structure and field presence
- Checks for perfect_actions and best_actions fields added by SA script
- Tests dataset metadata consistency
- **Usage**: Run from project root: `poetry run python utils/test_dataset_structure.py`

### `test_job_launch.py`
**Job Manager Testing Script**
- Tests the enhanced job manager functionality
- Validates CLI argument parsing and dataset naming
- Tests parallel job execution capabilities
- **Usage**: Run from project root: `poetry run python utils/test_job_launch.py`

## Key Features
- All scripts use the enhanced dataset manager with perfect/best action support
- Scripts validate the complete pipeline from SA generation to ML training
- Episode-level partitioning preserves episode boundaries while extracting transitions
- Full PyTorch integration with proper tensor batching

## Dependencies
- All scripts require the poetry environment: `poetry install`
- Depends on the enhanced `optomech/` package with dataset manager updates
- Requires existing dataset in `datasets/checkout_dataset/` for validation
