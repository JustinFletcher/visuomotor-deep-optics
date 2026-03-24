"""
Environment configuration defaults for optomech environments.

This module provides default environment parameters that can be used
across different training and evaluation scripts, ensuring consistency
and reducing duplication.
"""

from dataclasses import dataclass
from typing import Optional
from argparse import Namespace


@dataclass
class OptomechEnvConfig:
    """Default environment configuration for optomech environments."""
    
    # Basic environment settings
    env_id: str = "optomech-v1"
    """The environment id"""
    max_episode_steps: int = 100
    """The maximum number of steps per episode"""
    
    # GPU and rendering settings
    gpu_list: str = "0"
    """The list of GPUs to use."""
    render: bool = False
    """Whether to render the environment."""
    report_time: bool = False
    """Whether to report time statistics."""
    render_frequency: int = 1
    """The frequency of rendering."""
    render_dpi: float = 500.0
    """The DPI for rendering."""
    
    # Action and object configuration
    action_type: str = "none"
    """The type of action to use."""
    object_type: str = "binary"
    """The type of object to use."""
    aperture_type: str = "elf"
    """The type of aperture to use."""
    
    # Control settings
    discrete_control: bool = False
    """Toggle to enable discrete control."""
    discrete_control_steps: int = 128
    """The number of discrete control steps."""
    incremental_control: bool = False
    """Toggle to enable incremental control."""
    command_tensioners: bool = False
    """Toggle to enable agent control of tensioners."""
    command_secondaries: bool = False
    """Toggle to enable agent control of secondaries."""
    command_tip_tilt: bool = False
    """Toggle to enable agent control of tip/tilt for large mirrors."""
    command_dm: bool = False
    """Toggle to enable agent control of dm."""
    
    # Observation settings
    observation_mode: str = "image_only"
    """The type of observation to model 'image_only' or 'image_action'."""
    focal_plane_image_size_pixels: int = 256
    """The size of the focal plane image in pixels."""
    observation_window_size: int = 2**1
    """The size of the observation window."""

    # Dark hole settings
    dark_hole: bool = False
    """Whether to enable dark hole in target image."""
    dark_hole_angular_location_degrees: float = 45
    """Angular location of dark hole center in degrees, measured counterclockwise from positive x-axis."""
    dark_hole_location_radius_fraction: float = 0.3
    """Radius of dark hole location in units of maximum focal grid radius."""
    dark_hole_size_radius: float = 0.05
    """Radius of dark hole size in units of maximum focal grid radius."""
    
    # Adaptive optics settings
    ao_loop_active: bool = False
    """Whether the AO loop is active."""
    ao_interval_ms: float = 1.0
    """The interval between AO updates."""
    
    # Episode and environment settings
    num_episodes: int = 1
    """The number of episodes to run."""
    num_atmosphere_layers: int = 0
    """The number of atmosphere layers."""
    reward_threshold: float = 25.0
    """The reward threshold to reach."""
    num_steps: int = 16
    """The number of steps to take."""
    silence: bool = False
    """Whether to silence the output."""
    optomech_version: str = "test"
    """The version of optomech to use."""
    reward_function: str = "strehl"
    """The reward function to use."""
    
    # Timing intervals
    control_interval_ms: float = 2.0
    """The interval between control updates."""
    frame_interval_ms: float = 4.0
    """The interval between frames."""
    decision_interval_ms: float = 8.0
    """The interval between decisions."""
    
    # Motion modeling
    init_differential_motion: bool = False
    """Whether to initialize differential motion."""
    simulate_differential_motion: bool = False
    """Whether to simulate differential motion."""
    model_wind_diff_motion: bool = False
    """Whether to model wind differential motion."""
    model_gravity_diff_motion: bool = False
    """Whether to model gravity differential motion."""
    model_temp_diff_motion: bool = False
    """Whether to model temperature differential motion."""
    
    # State recording
    record_env_state_info: bool = False
    """Whether to record environment state information."""
    write_env_state_info: bool = False
    """Whether to write environment state information."""
    state_info_save_dir: str = "./tmp/"
    """The directory to save state information."""
    
    # Hardware configuration
    randomize_dm: bool = False
    """Whether to randomize the DM."""
    num_tensioners: int = 16
    """The number of tensioners."""
    
    # Extended object settings
    extended_object_image_file: str = ".\\resources\\sample_image.png"
    """The file for the extended object image."""
    extended_object_distance: Optional[str] = None
    """The distance to the extended object."""
    extended_object_extent: Optional[str] = None
    """The extent of the extended object."""


def create_env_args_from_config(config: OptomechEnvConfig = None, **overrides) -> Namespace:
    """
    Create an argparse.Namespace object from environment configuration.
    
    Args:
        config: OptomechEnvConfig instance. If None, uses default values.
        **overrides: Additional keyword arguments to override config values.
        
    Returns:
        argparse.Namespace object with environment configuration.
    """
    from argparse import Namespace
    
    if config is None:
        config = OptomechEnvConfig()
    
    # Convert dataclass to dict
    env_dict = {k: v for k, v in config.__dict__.items()}
    
    # Apply any overrides
    env_dict.update(overrides)
    
    return Namespace(**env_dict)


def parse_environment_flags(flags_list: list) -> dict:
    """
    Parse a list of environment flags into a dictionary.
    
    Args:
        flags_list: List of strings like ["--flag_name=value", "--other_flag"]
        
    Returns:
        Dictionary with flag names as keys and values
    """
    parsed = {}
    
    for flag in flags_list:
        if flag.startswith('--'):
            flag = flag[2:]  # Remove '--' prefix
            
            if '=' in flag:
                key, value = flag.split('=', 1)
                
                # Try to convert to appropriate type
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.replace('.', '').replace('-', '').isdigit():
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                        
                parsed[key] = value
            else:
                # Flag without value (boolean true)
                parsed[flag] = True
                
    return parsed


def merge_config_with_flags(config: OptomechEnvConfig = None, 
                           flags_list: list = None,
                           **additional_overrides) -> Namespace:
    """
    Create environment args by merging default config with environment flags.
    
    Args:
        config: Base configuration. If None, uses defaults.
        flags_list: List of environment flags to parse and apply.
        **additional_overrides: Additional overrides to apply.
        
    Returns:
        argparse.Namespace with merged configuration.
    """
    if config is None:
        config = OptomechEnvConfig()
    
    # Parse flags if provided
    flag_overrides = {}
    if flags_list:
        flag_overrides = parse_environment_flags(flags_list)
    
    # Merge all overrides
    all_overrides = {**flag_overrides, **additional_overrides}
    
    return create_env_args_from_config(config, **all_overrides)
