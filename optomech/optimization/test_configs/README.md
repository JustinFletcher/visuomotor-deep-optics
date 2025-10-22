# SA Dataset Test Configurations

This directory contains pre-configured job configs for different SA dataset generation test scenarios.

## Available Test Configs

### `base_test_config.json`
- **Purpose**: Basic SA dataset generation with minimal settings
- **Image Size**: 256x256 pixels
- **Control**: Secondary mirror segments only
- **Reward**: Strehl ratio
- **Atmosphere**: No atmospheric layers
- **Use Case**: Quick tests and baseline comparisons

### `test_512px_config.json`
- **Purpose**: High-resolution image testing
- **Image Size**: 512x512 pixels
- **Control**: Secondary mirror segments only
- **Reward**: Strehl ratio
- **Atmosphere**: No atmospheric layers
- **Use Case**: Testing larger observation sizes and memory usage

### `test_tip_tilt_config.json`
- **Purpose**: Test tip/tilt control in addition to secondaries
- **Image Size**: 256x256 pixels
- **Control**: Secondary segments + tip/tilt
- **Reward**: Alignment-based reward
- **Atmosphere**: No atmospheric layers
- **Use Case**: Testing expanded action spaces

### `test_dm_config.json`
- **Purpose**: Test deformable mirror control with atmosphere
- **Image Size**: 256x256 pixels
- **Control**: Secondary segments + deformable mirror
- **Reward**: Strehl ratio
- **Atmosphere**: 1 atmospheric layer
- **Use Case**: Testing complex control scenarios

## Usage

To use a test config with the SA dataset builder:

```bash
poetry run python optomech/optimization/build_sa_dataset.py \\
    --num_samples 50 \\
    --dataset_name my_test \\
    --dataset_save_path ./test_datasets/ \\
    --write_frequency 10 \\
    --job_config_file optomech/optimization/test_configs/test_512px_config.json
```

Or bypass the job config entirely for custom parameters:

```bash
poetry run python optomech/optimization/build_sa_dataset.py \\
    --num_samples 50 \\
    --dataset_name my_test \\
    --dataset_save_path ./test_datasets/ \\
    --write_frequency 10 \\
    --focal_plane_image_size_pixels 512 \\
    --command_secondaries
```

## Configuration Guidelines

- **Image Size**: 256px for quick tests, 512px for high-resolution
- **Memory**: Increase `estimated_memory_per_job_gb` for larger images
- **Control Modes**: Start with secondaries only, add tip/tilt or DM as needed
- **Atmosphere**: Use 0 layers for deterministic tests, 1+ for realistic scenarios
- **Reward Function**: `strehl` for image quality, `align` for positioning

## Creating New Configs

1. Copy an existing config as a template
2. Modify the `job_command` array parameters
3. Adjust `resource_limits` based on expected computational load
4. Test with small `num_samples` first
5. Document the purpose and use case
