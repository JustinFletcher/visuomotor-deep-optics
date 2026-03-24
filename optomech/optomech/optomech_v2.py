"""
Optomech V2 -- Modular Gymnasium environment for distributed-aperture
telescope control.

This is a functionally-identical refactor of optomech.py (v1) with:
  - All hardcoded physics / sim parameters externalized as kwargs with
    documented defaults collected in DEFAULT_CONFIG.
  - Clearly decomposed helper methods for aperture construction, action-
    space building, reward computation, AO correction, and detector
    modeling.
  - Thorough inline commentary aimed at developers building RL or
    classical control actors.

Author: Justin Fletcher (original), refactored for v2.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import copy
import math
import time
import uuid
import pickle
import datetime
import glob

import numpy as np
import gymnasium as gym

from collections import deque
from pathlib import Path

from scipy import signal
from anytree import Node, RenderTree
import scipy.ndimage as ndimage

from PIL import Image, ImageSequence
from gymnasium import spaces, logger
from gymnasium.utils import seeding

from matplotlib import pyplot as plt
from matplotlib import image
from matplotlib.figure import Figure

import astropy.units as u
import hcipy


# ===================================================================
# DEFAULT CONFIGURATION
# ===================================================================
# Every externalisable parameter is listed here with its default value
# and a short description.  Users override any subset of these when
# instantiating OptomechEnv (or the lower-level OpticalSystem).
# Aperture-type switch-case geometry (segment counts, focal lengths,
# etc.) is intentionally left inside the aperture builder because it
# is tightly coupled to the HCIPy aperture API.
# ===================================================================

DEFAULT_CONFIG = {

    # --- Run / debug -------------------------------------------------
    "report_time": False,           # Print wall-clock timing for each sub-step
    "render": False,                # Whether to render the environment
    "render_frequency": 1,          # Render every N steps
    "render_dpi": 500.0,            # DPI for rendered figures
    "silence": False,               # Suppress all print output

    # --- Aperture ----------------------------------------------------
    "aperture_type": "elf",         # One of: elf, circular, nanoelf, nanoelfplus
    "num_tensioners": 16,           # Number of tensioner actuators

    # --- Object plane ------------------------------------------------
    "object_type": "binary",        # One of: single, binary, usaf1951, flat
    "object_plane_extent_meters": 1.0,  # Spatial extent of the object plane (m)
    "object_plane_distance_meters": 1.0,  # Distance to the object plane (m)

    # --- Focal plane / grid ------------------------------------------
    "focal_plane_image_size_pixels": 256,  # Pixels along one side of the focal image

    # --- Optical simulation ------------------------------------------
    "wavelength": 1000e-9,          # Centre wavelength (m)
    "oversampling_factor": 8,       # Aperture super-sampling factor
    "bandwidth_nanometers": 200.0,  # Spectral bandwidth for polychromatic sim (nm)
    "bandwidth_sampling": 2,        # Number of wavelength samples across the band

    # --- Atmosphere --------------------------------------------------
    "num_atmosphere_layers": 0,     # Number of Kolmogorov turbulence layers
    "seeing_arcsec": 0.5,           # Fried seeing at 500 nm (arcsec)
    "outer_scale_meters": 40.0,     # Von-Karman outer scale (m)
    "tau0_seconds": 10.0,           # Greenwood time constant (s)

    # --- Structural differential motion ------------------------------
    "init_differential_motion": False,      # Apply initial PTT perturbation
    "simulate_differential_motion": False,  # Evolve structural motion each step
    "model_wind_diff_motion": False,        # Include wind-driven motion
    "model_gravity_diff_motion": False,     # Include gravity-driven motion
    "model_temp_diff_motion": False,        # Include temperature-driven motion
    # Wind parameters
    "initial_ground_wind_speed_mps": 3.0,   # Mean ground wind speed (m/s)
    "ground_wind_speed_std_fraction": 0.08, # Wind std as fraction of speed per ms
    "max_ground_wind_speed_mps": 20.0,      # Wind speed clamp (m/s)
    # Temperature parameters
    "initial_ground_temp_degcel": 20.0,     # Mean ground temperature (C)
    "ground_temp_ms_sampled_std": 0.0,      # Temperature std per ms (C)
    # Gravity parameters
    "initial_gravity_normal_deg": 45.0,     # Gravity normal angle (deg)
    "gravity_normal_ms_sampled_std": 0.0,   # Gravity normal std per ms (deg)
    # Init wind displacement parameters
    "init_wind_piston_micron_std": 1.0,     # Initial piston std (um)
    "init_wind_piston_clip_m": 4e-6,        # Clip initial piston displacement (m)
    "init_wind_tip_arcsec_std_tt": 0.05,    # Tip std when tip-tilt is commanded (as)
    "init_wind_tilt_arcsec_std_tt": 0.05,   # Tilt std when tip-tilt is commanded (as)
    # Runtime wind displacement parameters
    "runtime_wind_piston_micron_factor": 1.0 / 8.0,  # Piston std = wind * factor
    "runtime_wind_tip_arcsec_factor": 1.0 / 32.0,    # Tip std = wind * factor
    "runtime_wind_tilt_arcsec_factor": 1.0 / 32.0,   # Tilt std = wind * factor
    "runtime_wind_incremental_factor": 0.01,          # Scale factor for incremental PTT
    # Init gravity displacement parameters
    "init_gravity_piston_micron_std": 300.0,  # Initial gravity piston std (um)
    "init_gravity_tip_arcsec_std_tt": 15.0,   # Gravity tip std when TT commanded (as)
    "init_gravity_tilt_arcsec_std_tt": 15.0,  # Gravity tilt std when TT commanded (as)

    # --- Control hierarchy -------------------------------------------
    "control_interval_ms": 2.0,     # Interval between actuator commands (ms)
    "frame_interval_ms": 4.0,       # Interval between science frames (ms)
    "decision_interval_ms": 8.0,    # Interval between agent decisions (ms)
    "ao_interval_ms": 1.0,          # AO loop cadence (ms)
    "max_episode_steps": 100,       # Maximum steps per episode

    # --- Agent action toggles ----------------------------------------
    "command_secondaries": False,    # Agent controls secondary mirror PTT
    "command_tensioners": False,     # Agent controls tensioner forces
    "command_dm": False,             # Agent controls deformable mirror
    "command_tip_tilt": False,       # Agent controls tip/tilt on large mirrors

    # --- Action parameterization -------------------------------------
    "discrete_control": False,       # Use discrete (MultiDiscrete) actions
    "discrete_control_steps": 128,   # Number of discrete increments per DOF
    "incremental_control": False,    # Incremental (relative) commands
    "action_type": "none",           # Additional action type descriptor
    "actuator_noise": True,          # Add Gaussian noise to actuator commands
    "actuator_noise_fraction": 1e-4, # Noise std as fraction of correction range

    # --- Observation -------------------------------------------------
    "observation_mode": "image_only",  # 'image_only' or 'image_action'
    "observation_window_size": 2,      # Observation window (unused placeholder)

    # --- Adaptive optics ---------------------------------------------
    "ao_loop_active": False,         # Run closed-loop AO each step
    "randomize_dm": False,           # Randomize DM actuators on reset
    "dm_gain": 0.6,                  # AO loop integrator gain
    "dm_leakage": 0.01,             # AO loop leaky-integrator leakage
    "dm_model_type": "gaussian_influence",  # DM model ('gaussian_influence' or 'disk_harmonic_basis')
    "dm_num_actuators_across": 35,   # Actuators across the pupil
    "dm_num_modes": 500,             # Number of modes (disk-harmonic only)
    "shwfs_f_number": 50,            # SHWFS f-number
    "shwfs_num_lenslets": 40,        # SHWFS lenslets across diameter
    "shwfs_diameter_m": 5e-3,        # SHWFS physical diameter (m)
    "dm_interaction_rcond": 1e-3,    # Tikhonov regularization for DM calibration
    "dm_probe_amp_fraction": 0.01,   # Probe amplitude as fraction of wavelength
    "dm_flux_limit_fraction": 0.5,   # Flux fraction for subaperture selection
    "dm_cache_dir": "./tmp/cache/",  # Directory for caching interaction matrices

    # --- DM / actuator physics ---------------------------------------
    "microns_opd_per_actuator_bit": 0.00015,  # OPD per bit (um)
    "stroke_count_limit": 20000,               # Max stroke counts

    # --- Secondary mirror corrections --------------------------------
    "max_piston_correction_micron": 10.0,   # Max piston correction (um)
    "max_tip_correction_arcsec": 20.0,      # Max tip correction (arcsec)
    "max_tilt_correction_arcsec": 20.0,     # Max tilt correction (arcsec)
    "get_disp_corr_max_piston_micron": 3.0, # Max piston for displacement correction (um)
    "get_disp_corr_max_tip_arcsec": 20.0,   # Max tip for displacement correction (as)
    "get_disp_corr_max_tilt_arcsec": 20.0,  # Max tilt for displacement correction (as)

    # --- Reward ------------------------------------------------------
    "reward_function": "composite",  # One of: composite, strehl, align, dark_hole,
                                     #         image_mse, negastrehl, negaexpstrehl,
                                     #         strehl_closed, ao_rms_slope,
                                     #         norm_ao_rms_slope, ao_closed
    "reward_weight_strehl": 1.0,     # Weight for Strehl component in composite reward
    "reward_weight_dark_hole": 0.0,  # Weight for dark-hole component in composite reward
    "reward_weight_image_quality": 0.0,  # Weight for image-quality component
    "reward_threshold": 25.0,        # Target reward threshold (informational)
    "align_radius": 32,              # Pixel radius for align reward center mask
    "align_radius_max_expand": 64,   # Expanded radius after mse threshold
    "align_mse_expand_threshold": -1.25,  # MSE threshold to expand radius
    "ao_closed_inv_slope_threshold": 2e6, # Inverse slope RMS threshold for ao_closed
    "dark_hole_alpha": 0.0,          # Blending factor: dark_hole vs mse in dark_hole reward
    "action_penalty": True,          # Penalise large actions in reward
    "action_penalty_weight": 0.03,   # L2 action penalty weight
    "oob_penalty": True,
    "oob_penalty_weight": 0.5,

    # --- Dark hole ---------------------------------------------------
    "dark_hole": False,                         # Enable dark hole in target image
    "dark_hole_angular_location_degrees": 0.0,  # Angular location (degrees CCW from +x)
    "dark_hole_location_radius_fraction": 0.0,  # Location radius (fraction of grid)
    "dark_hole_size_radius": 0.0,               # Dark hole size (fraction of grid)

    # --- Detector model ----------------------------------------------
    "detector_power_watts": 1e-12,      # Integrated power (W)
    "detector_wavelength_meters": 500e-9,  # Detector reference wavelength (m)
    "detector_quantum_efficiency": 0.8, # Quantum efficiency
    "detector_gain_e_per_dn": 0.5,      # System gain (electrons per DN)
    "detector_max_dn": 65535,           # Maximum digital number (saturation)

    # --- State recording ---------------------------------------------
    "record_env_state_info": False,  # Store full environment state each step
    "write_env_state_info": False,   # Write state info to disk
    "state_info_save_dir": "./tmp/", # Save directory for state info

    # --- Episode -----------------------------------------------------
    "num_episodes": 1,               # Number of episodes (for batch runners)
    "num_steps": 16,                 # Number of steps (for batch runners)

    # --- Optomech version tag ----------------------------------------
    "optomech_version": "v2",        # Version tag
}


# ===================================================================
# V2 Logging
# ===================================================================

_V2_TAG = "\033[96m[optomech-v2]\033[0m"   # Cyan tag for terminal colour
_V2_TAG_PLAIN = "[optomech-v2]"             # Fallback for non-ANSI terminals

# Box-drawing pieces for section banners
_TOP    = "\u2554" + "\u2550" * 58 + "\u2557"   # top
_BOT    = "\u255A" + "\u2550" * 58 + "\u255D"   # bottom
_SIDE   = "\u2551"                                # vertical bar
_MID    = "\u2560" + "\u2550" * 58 + "\u2563"   # mid separator
_THIN   = "\u2502"                                # thin vertical
_HTHIN  = "\u2500" * 58                           # thin horizontal


def _v2(msg, indent=0):
    """Print with the v2 prefix tag and optional indent."""
    pad = "  " * indent
    print(f"{_V2_TAG} {pad}{msg}")


def _v2_section(title):
    """Print a box-drawn section header."""
    inner = f" {title} ".center(58)
    print(f"{_V2_TAG} {_TOP}")
    print(f"{_V2_TAG} {_SIDE}{inner}{_SIDE}")
    print(f"{_V2_TAG} {_BOT}")


def _v2_subsection(title):
    """Print a lighter sub-section separator."""
    inner = f" {title} ".center(58, "\u2500")
    print(f"{_V2_TAG} {inner}")


def _v2_kv(key, value, indent=1):
    """Print a key-value pair."""
    pad = "  " * indent
    print(f"{_V2_TAG} {pad}{key:<42s} {value}")


def _v2_timer(label, elapsed, indent=1):
    """Print a timing measurement."""
    pad = "  " * indent
    bar_len = min(int(elapsed * 200), 30)  # rough bar, 5ms = 1 char
    bar = "\u2588" * bar_len
    print(f"{_V2_TAG} {pad}\u23f1  {label:<34s} {elapsed:8.4f}s {bar}")


def _print_config_banner(cfg, title="Optomech V2 Configuration"):
    """Pretty-print all configuration key-value pairs inside a box."""
    _v2_section(title)
    max_key_len = max(len(k) for k in cfg)
    # Group keys by their section prefix for readability.
    prev_section = None
    for k in sorted(cfg.keys()):
        section = k.split("_")[0]
        if section != prev_section and prev_section is not None:
            print(f"{_V2_TAG}   {'':>{max_key_len}}   {'':>10}")
        prev_section = section
        print(f"{_V2_TAG}   {k:<{max_key_len}}  =  {cfg[k]}")
    print(f"{_V2_TAG} {_BOT}")


# ===================================================================
# Utility functions
# ===================================================================


def cosine_similarity(u, v):
    """Cosine similarity between two flat arrays."""
    u = u.flatten()
    v = v.flatten()
    return np.dot(v, u) / (np.linalg.norm(u) * np.linalg.norm(v))


def gaussian_kernel(n, std, normalised=False):
    """Generate an n x n Gaussian kernel with given std."""
    gaussian1D = signal.windows.gaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    if normalised:
        gaussian2D /= (2 * np.pi * (std ** 2))
    return gaussian2D


def generalized_radial_profile(shape, alpha=10.0, beta=2.0,
                                invert=False, normalize=False,
                                epsilon=1e-8):
    """2-D radial profile: exp(-(d/alpha)^beta) from centre of *shape*."""
    rows, cols = shape
    center_row = (rows - 1) / 2.0
    center_col = (cols - 1) / 2.0
    y_indices, x_indices = np.indices((rows, cols))
    dists = np.sqrt((x_indices - center_col) ** 2 +
                    (y_indices - center_row) ** 2)
    profile = np.exp(-(dists / alpha) ** beta)
    if invert:
        profile = 1.0 / (profile + epsilon)
    if normalize:
        profile = (profile - np.min(profile)) / (np.max(profile) - np.min(profile))
    return profile


def offset_gaussian(n, mu_x, mu_y, std, kernel_extent, normalised=False):
    """Place a Gaussian kernel at (mu_x, mu_y) in an n x n array."""
    scene_array = np.zeros((n, n))
    source_array = gaussian_kernel(kernel_extent, std, normalised=normalised)
    x_start = int(mu_x - kernel_extent // 2)
    x_end = int(mu_x + kernel_extent // 2)
    y_start = int(mu_y - kernel_extent // 2)
    y_end = int(mu_y + kernel_extent // 2)
    scene_array[x_start:x_end, y_start:y_end] = source_array
    return scene_array


def one_hot_array(n, x, y, value=1.0):
    """n x n array with a single non-zero entry at (x, y)."""
    scene_array = np.zeros((n, n))
    scene_array[x, y] = value
    return scene_array


# ===================================================================
# Wavelength helper
# ===================================================================

def _centered_wavelengths(center_wl, bandwidth_nm, n_samples):
    """Compute centered wavelength samples across a bandwidth."""
    bw_m = bandwidth_nm / 1e9
    if n_samples <= 1:
        return [center_wl]
    bin_width = bw_m / n_samples
    first = center_wl - bw_m / 2.0 + bin_width / 2.0
    return [first + i * bin_width for i in range(n_samples)]


# ===================================================================
# ObjectPlane
# ===================================================================

class ObjectPlane(object):
    """Generates the 2-D intensity distribution for the object scene.

    Supported object_type values:
        'single'   -- point source at the centre
        'binary'   -- two Gaussian blobs separated by ~0.6 arcsec
        'usaf1951' -- USAF-1951 resolution target image
        'flat'     -- uniform unit-intensity field
    """

    def __init__(self,
                 object_type="binary",
                 object_plane_extent_pixels=128,
                 object_plane_extent_meters=1.0,
                 object_plane_distance_meters=1000000,
                 randomize=False,
                 **kwargs):

        self.extent_pixels = object_plane_extent_pixels
        self.extent_meters = object_plane_extent_meters
        self.distance_meters = object_plane_distance_meters

        if object_type == "single":
            if randomize:
                kwargs['source_vmag'] = np.random.uniform(0.0, 25.0)
                kwargs['source_position'] = [
                    np.random.uniform(0.0, self.extent_pixels),
                    np.random.uniform(0.0, self.extent_pixels)
                ]
            self.array = self._make_single_object(**kwargs)

        elif object_type == "binary":
            self.array = self._make_binary_object(**kwargs)

        elif object_type == "usaf1951":
            self.array = self._load_usaf1951(object_plane_extent_pixels)

        elif object_type == "flat":
            self.array = np.ones((object_plane_extent_pixels,
                                  object_plane_extent_pixels))
        else:
            raise NotImplementedError(
                "ObjectPlane object_type was '%s', but only 'single', "
                "'binary', 'usaf1951' and 'flat' are implemented." % object_type)

    # ---- private helpers -------------------------------------------

    def _make_binary_object(self, **kwargs):
        """Two Gaussian blobs separated by ~0.6 arcsec (binary star)."""
        std = 1.0
        kernel_extent = 8 * std
        # ifov (arcsec/pixel): 0.0165012
        ifov = 0.0165012
        separation_pixels = int(0.6 / ifov)
        mu_x = self.extent_pixels // 2

        # Primary source
        mu_y_primary = (self.extent_pixels // 2) - (separation_pixels // 2)
        primary = offset_gaussian(self.extent_pixels, mu_x, mu_y_primary,
                                  std, kernel_extent, normalised=True)

        # Secondary source
        mu_y_secondary = (self.extent_pixels // 2) + (separation_pixels // 2)
        secondary = offset_gaussian(self.extent_pixels, mu_x, mu_y_secondary,
                                    std, kernel_extent, normalised=True)
        return primary + secondary

    def _make_single_object(self, **kwargs):
        """Point source at the centre of the field."""
        x = self.extent_pixels // 2
        y = self.extent_pixels // 2
        array_value = 1.02e-8
        return one_hot_array(self.extent_pixels, x, y, value=array_value)

    def _load_usaf1951(self, size):
        """Load and normalise USAF-1951 resolution target."""
        valid_sizes = {512, 256, 128}
        if size not in valid_sizes:
            raise NotImplementedError(
                "ObjectPlane extent was %s, but only %s are implemented "
                "for usaf1951." % (size, valid_sizes))
        raw = self._rgb2gray(image.imread('usaf1951_%d.jpg' % size))
        raw = raw / np.max(raw)
        # Invert so chart lines are bright
        raw = -(raw - np.max(raw))
        return raw

    @staticmethod
    def _rgb2gray(rgb):
        """Convert RGB image to greyscale using standard luminance weights."""
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


# ===================================================================
# OpticalSystem
# ===================================================================

class OpticalSystem(object):
    """End-to-end optical simulation: aperture -> atmosphere ->
    segmented mirror -> DM -> focal plane.

    All configurable physics parameters are accepted as **kwargs and
    fall back to DEFAULT_CONFIG values when not provided.
    """

    def __init__(self, **kwargs):

        # Merge with defaults so every key is guaranteed present.
        cfg = {**DEFAULT_CONFIG, **kwargs}

        # --- Store key flags -----------------------------------------
        self.model_wind_diff_motion = cfg["model_wind_diff_motion"]
        self.model_gravity_diff_motion = cfg["model_gravity_diff_motion"]
        self.model_temp_diff_motion = cfg["model_temp_diff_motion"]
        self.incremental_control = cfg["incremental_control"]
        self.command_tip_tilt = cfg["command_tip_tilt"]
        self.report_time = cfg["report_time"]
        self.num_tensioners = cfg["num_tensioners"]
        self.model_ao = cfg.get('model_ao', False)
        self.simulate_differential_motion = cfg['simulate_differential_motion']
        self.init_differential_motion = cfg['init_differential_motion']
        self.discrete_control = cfg['discrete_control']
        self.discrete_control_steps = cfg['discrete_control_steps']
        self._actuator_noise = cfg['actuator_noise']
        self._actuator_noise_fraction = cfg['actuator_noise_fraction']

        # --- DM physics constants ------------------------------------
        self.microns_opd_per_actuator_bit = cfg["microns_opd_per_actuator_bit"]
        self.stroke_count_limit = cfg["stroke_count_limit"]

        # --- Store full config for later reference -------------------
        self._cfg = cfg

        # --- Wind / temperature / gravity state ----------------------
        self.ground_wind_speed_mps = cfg["initial_ground_wind_speed_mps"]
        self.ground_wind_speed_ms_sampled_std_mps = (
            cfg["ground_wind_speed_std_fraction"] * self.ground_wind_speed_mps
        )
        self.ground_temp_degcel = cfg["initial_ground_temp_degcel"]
        self.ground_temp_ms_sampled_std_mps = cfg["ground_temp_ms_sampled_std"]
        if self.ground_temp_ms_sampled_std_mps != 0.0:
            raise NotImplementedError(
                "Non-zero ground_temp_ms_sampled_std is not supported.")
        self.gravity_normal_deg = cfg["initial_gravity_normal_deg"]
        self.gravity_normal_ms_sampled_std_mps = cfg["gravity_normal_ms_sampled_std"]
        if self.gravity_normal_ms_sampled_std_mps != 0.0:
            raise NotImplementedError(
                "Non-zero gravity_normal_ms_sampled_std is not supported.")

        # === Build aperture ==========================================
        aperture_type = cfg['aperture_type']
        _v2_subsection("Aperture: %s" % aperture_type)
        (aperture_func, segments_func,
         focal_length, pupil_diameter,
         focal_plane_image_size_meters) = self._build_aperture(
             aperture_type, cfg)

        # --- Optical grid parameters ---------------------------------
        num_px = cfg['focal_plane_image_size_pixels']
        self.wavelength = cfg['wavelength']
        oversampling_factor = cfg['oversampling_factor']

        # --- Atmosphere parameters -----------------------------------
        seeing = cfg['seeing_arcsec']
        outer_scale = cfg['outer_scale_meters']
        tau0 = cfg['tau0_seconds']

        fried_parameter = hcipy.seeing_to_fried_parameter(seeing)
        _v2_kv("fried_parameter", fried_parameter)
        Cn_squared = hcipy.Cn_squared_from_fried_parameter(
            fried_parameter, wavelength=self.wavelength)
        velocity = 0.314 * fried_parameter / tau0

        # --- Focal plane geometry ------------------------------------
        focal_plane_extent_metres = focal_plane_image_size_meters
        airy_extent_radians = 1.22 * self.wavelength / pupil_diameter
        airy_extent_meters = airy_extent_radians * focal_length
        focal_plane_pixel_extent_meters = focal_plane_extent_metres / num_px
        sampling = airy_extent_meters / focal_plane_pixel_extent_meters
        focal_plane_resolution_element = (
            self.wavelength * focal_length / pupil_diameter)
        focal_plane_pixels_per_meter = num_px / focal_plane_extent_metres
        self.ifov = (206265 / focal_length) * focal_plane_pixel_extent_meters
        fov = self.ifov * num_px
        num_airy = num_px / (2 * sampling)

        # Log computed focal-plane values
        _v2_subsection("Focal Plane Geometry")
        _v2_kv("grid pixels",            "%d" % num_px)
        _v2_kv("extent (m)",             "%.6e" % focal_plane_extent_metres)
        _v2_kv("pixel extent (m)",       "%.6e" % focal_plane_pixel_extent_meters)
        _v2_kv("resolution element (m)", "%.6e" % focal_plane_resolution_element)
        _v2_kv("pixels per meter",       "%.2f" % focal_plane_pixels_per_meter)
        _v2_kv("num airy (res-el radii)","%.4f" % num_airy)
        _v2_kv("sampling (px/res-el)",   "%.4f" % sampling)
        _v2_kv("iFOV (arcsec/px)",       "%.7f" % self.ifov)
        _v2_kv("FOV (arcsec)",           "%.4f" % fov)
        _v2_kv("incremental_control",    str(self.incremental_control))

        # --- Object plane --------------------------------------------
        _v2("Building object plane...")
        self.object_plane = ObjectPlane(
            object_type=cfg['object_type'],
            object_plane_extent_pixels=num_px,
            object_plane_extent_meters=cfg['object_plane_extent_meters'],
            object_plane_distance_meters=cfg['object_plane_distance_meters'],
        )

        # --- Pupil grid ----------------------------------------------
        _v2("Building pupil grid...")
        self.pupil_grid = hcipy.make_pupil_grid(
            dims=num_px, diameter=pupil_diameter)

        # --- Atmosphere layers ---------------------------------------
        self.atmosphere_layers = []
        _v2("Building %d atmosphere layer(s)..." % cfg['num_atmosphere_layers'])
        for _ in range(cfg['num_atmosphere_layers']):
            layer = hcipy.InfiniteAtmosphericLayer(
                self.pupil_grid, Cn_squared, outer_scale, velocity)
            self.atmosphere_layers.append(layer)

        # --- Focal grid & propagator ---------------------------------
        focal_grid = hcipy.make_pupil_grid(
            dims=num_px, diameter=focal_plane_extent_metres)
        focal_grid = focal_grid.shifted(focal_grid.delta / 2)

        _v2("Building Fraunhofer propagator...")
        self.pupil_to_focal_propagator = hcipy.FraunhoferPropagator(
            self.pupil_grid, focal_grid, focal_length)

        # --- Evaluate aperture / segments ----------------------------
        aperture_field = hcipy.evaluate_supersampled(
            aperture_func, self.pupil_grid, oversampling_factor)
        segments_field = hcipy.evaluate_supersampled(
            segments_func, self.pupil_grid, oversampling_factor)

        self.segmented_mirror = hcipy.SegmentedDeformableMirror(segments_field)
        self.aperture = aperture_field

        # --- Perfect (reference) image -------------------------------
        # Polychromatic perfect PSF: average across all sampled wavelengths
        _perfect_wavelengths = _centered_wavelengths(
            self.wavelength,
            cfg['bandwidth_nanometers'],
            cfg['bandwidth_sampling'])
        _v2("Computing polychromatic perfect PSF (%d wavelength(s): %s nm)"
            % (len(_perfect_wavelengths),
               ", ".join("%.1f" % (wl * 1e9) for wl in _perfect_wavelengths)))
        perfect_accum = None
        for _pwl in _perfect_wavelengths:
            wf = hcipy.Wavefront(self.aperture, _pwl)
            img_wf = self.pupil_to_focal_propagator(
                self.segmented_mirror(wf))
            if perfect_accum is None:
                perfect_accum = np.array(img_wf.intensity, dtype=np.float64)
            else:
                perfect_accum += np.array(img_wf.intensity, dtype=np.float64)
        perfect_accum /= len(_perfect_wavelengths)
        self.perfect_image = perfect_accum
        self.perfect_image_max = float(np.max(self.perfect_image))
        side = int(np.sqrt(self.perfect_image.size))
        self.target_image = self.perfect_image.reshape((side, side))

        # --- Dark hole (optional) ------------------------------------
        if cfg.get('dark_hole', False):
            self._apply_dark_hole(cfg, num_px)

        # --- AO subsystem (DM + SHWFS) ------------------------------
        if self.model_ao:
            self._build_ao_subsystem(cfg, pupil_diameter, focal_grid)

        # --- Initialize differential motion --------------------------
        if self.init_differential_motion:
            _v2("Applying initial differential motion...")
            self._init_natural_diff_motion()

        # Store baseline segment positions for relative commands.
        self._store_baseline_segment_displacements()

        # --- Science camera ------------------------------------------
        _v2("Building science camera...")
        self.camera = hcipy.NoiselessDetector(focal_grid)

        _v2_subsection("Optical System Ready")

    # ================================================================
    # Aperture construction
    # ================================================================

    def _build_aperture(self, aperture_type, cfg):
        """Build the selected aperture and return (aperture, segments,
        focal_length, pupil_diameter, focal_plane_image_size_meters).

        Aperture geometry constants are intentionally hardcoded here
        because they are tightly coupled to the HCIPy segment layout.
        """
        if aperture_type == "elf":
            self.num_apertures = 15
            focal_plane_image_size_meters = 8.192e-4
            focal_length = 32.5          # m
            pupil_diameter = 3.6         # m
            segment_diameter = 0.5       # m
            elf_segment_centroid_diameter = 2.7  # m

            # Random optomech interaction placeholders
            self.optomech_interaction_matrix = None
            interaction_size = 1
            self._optomech_encoder = np.random.rand(
                self.num_tensioners, interaction_size)
            self._optomech_decoder = np.random.rand(
                interaction_size, self.num_apertures * 3)

            aperture, segments = self._make_ring_aperture(
                pupil_diameter=elf_segment_centroid_diameter,
                num_apertures=self.num_apertures,
                segment_diameter=segment_diameter)

        elif aperture_type == "circular":
            focal_length = 200.0
            pupil_diameter = 3.6
            elf_segment_centroid_diameter = 2.5
            focal_plane_image_size_meters = 8.192e-4

            aper_coords = hcipy.SeparatedCoords(
                (np.array([0.0]), np.array([0.0])))
            segment_centers = hcipy.PolarGrid(aper_coords)
            circ_aperture = hcipy.make_circular_aperture(
                elf_segment_centroid_diameter)
            aperture, segments = hcipy.make_segmented_aperture(
                circ_aperture, segment_centers, return_segments=True)

        elif aperture_type == "nanoelf":
            self.num_apertures = 2
            focal_length = 1.018
            pupil_diameter = 0.1408
            focal_plane_image_size_meters = 8.192e-5

            aperture, segments = self._make_ring_aperture(
                pupil_diameter=pupil_diameter / 2.0,
                num_apertures=self.num_apertures,
                segment_diameter=0.0254 * 2)

        elif aperture_type == "nanoelfplus":
            self.num_apertures = 3
            focal_length = 1.018
            pupil_diameter = 0.1408
            focal_plane_image_size_meters = 8.192e-5

            aperture, segments = self._make_ring_aperture(
                pupil_diameter=pupil_diameter / 2.0,
                num_apertures=self.num_apertures,
                segment_diameter=0.0254 * 2)

        else:
            raise NotImplementedError(
                "aperture_type was '%s', but only 'elf', 'circular', "
                "'nanoelf', and 'nanoelfplus' are implemented." % aperture_type)

        return (aperture, segments, focal_length,
                pupil_diameter, focal_plane_image_size_meters)

    @staticmethod
    def _make_ring_aperture(pupil_diameter, num_apertures,
                            segment_diameter, return_segments=True):
        """Create a ring of circular sub-apertures (ELF / nanoELF)."""
        pupil_radius = pupil_diameter / 2
        segment_angles = np.linspace(0, 2 * np.pi, num_apertures + 1)[:-1]
        aper_coords = hcipy.SeparatedCoords(
            (np.array([pupil_radius]), segment_angles))
        segment_centers = hcipy.PolarGrid(aper_coords).as_('cartesian')
        circ = hcipy.make_circular_aperture(segment_diameter)
        aperture, segments = hcipy.make_segmented_aperture(
            circ, segment_centers, return_segments=return_segments)
        return aperture, segments

    def _apply_dark_hole(self, cfg, num_px):
        """Zero-out a circular region in the target image for dark-hole
        coronagraphy reward."""
        dh_angle_deg = cfg['dark_hole_angular_location_degrees']
        dh_loc_frac = cfg['dark_hole_location_radius_fraction']
        dh_size = cfg['dark_hole_size_radius']
        dh_size_px = int(dh_size * num_px / 2)
        dh_angle_rad = np.deg2rad(dh_angle_deg)
        dh_loc_px = int(dh_loc_frac * num_px / 2)
        cx = int(num_px / 2 + dh_loc_px * np.cos(dh_angle_rad))
        cy = int(num_px / 2 + dh_loc_px * np.sin(dh_angle_rad))
        y, x = np.ogrid[:num_px, :num_px]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= dh_size_px ** 2
        self.target_image[mask] = 0.0

    def _build_ao_subsystem(self, cfg, pupil_diameter, focal_grid):
        """Build the SHWFS, DM, and associated structures."""
        f_number = cfg['shwfs_f_number']
        num_lenslets = cfg['shwfs_num_lenslets']
        sh_diameter = cfg['shwfs_diameter_m']

        magnification = sh_diameter / pupil_diameter
        self.magnifier = hcipy.Magnifier(magnification)

        _v2("Building SHWFS (f/%.0f, %d lenslets)..." % (f_number, num_lenslets))
        self.shwfs = hcipy.SquareShackHartmannWavefrontSensorOptics(
            self.pupil_grid.scaled(magnification),
            f_number, num_lenslets, sh_diameter)
        self.shwfse = hcipy.ShackHartmannWavefrontSensorEstimator(
            self.shwfs.mla_grid,
            self.shwfs.micro_lens_array.mla_index)

        self.shwfs_camera = hcipy.NoiselessDetector(focal_grid)

        _v2("Building DM (model=%s)..." % dm_model_type)
        dm_model_type = cfg['dm_model_type']
        if dm_model_type == "disk_harmonic_basis":
            num_modes = cfg['dm_num_modes']
            dm_modes = hcipy.make_disk_harmonic_basis(
                self.pupil_grid, num_modes, pupil_diameter, 'neumann')
            dm_modes = hcipy.ModeBasis(
                [mode / np.ptp(mode) for mode in dm_modes],
                self.pupil_grid)
            self.dm = hcipy.DeformableMirror(dm_modes)
        elif dm_model_type == "gaussian_influence":
            n_act = cfg['dm_num_actuators_across']
            self.dm_influence_functions = hcipy.make_gaussian_influence_functions(
                self.pupil_grid,
                num_actuators_across_pupil=n_act,
                actuator_spacing=pupil_diameter / n_act)
            self.dm = hcipy.DeformableMirror(self.dm_influence_functions)
        else:
            raise ValueError("Unknown dm_model_type: %s" % dm_model_type)

    # ================================================================
    # Field helpers
    # ================================================================

    def make_object_field(self, array, center=None):
        """Return an HCIPy Field generator for the given 2-D array."""
        def func(grid):
            f = array.ravel()
            return hcipy.Field(f.astype('float'), grid)
        return func

    # ================================================================
    # Atmosphere
    # ================================================================

    def evolve_atmosphere_to(self, episode_time_ms):
        """Advance all atmosphere layers to the given episode time (ms)."""
        episode_time_seconds = episode_time_ms / 1000.0
        for layer in self.atmosphere_layers:
            layer.evolve_until(episode_time_seconds)

    # ================================================================
    # DM commands
    # ================================================================

    def command_dm(self, dm_command):
        """Apply a normalised [-1, 1] DM command vector."""
        dm_stroke_meters = (
            self.stroke_count_limit * self.microns_opd_per_actuator_bit * 1e-6)
        command_vector = np.array([x[0] for x in dm_command])
        dm_command_meters = (dm_stroke_meters / 2.0) * command_vector
        if self._actuator_noise:
            dm_command_meters += np.random.normal(
                0.0, self._actuator_noise_fraction * dm_stroke_meters,
                size=dm_command_meters.shape)
        self.dm.actuators = dm_command_meters

    # ================================================================
    # Optical simulation
    # ================================================================

    def simulate(self, wavelength):
        """Propagate a wavefront through the full optical train."""
        # 1) Create pupil wavefront from aperture
        t0 = time.time()
        self.object_wavefront = hcipy.Wavefront(self.aperture, wavelength)
        self.pre_atmosphere_object_wavefront = self.object_wavefront
        if self.report_time:
            _v2_timer("Object wavefront", time.time() - t0, indent=2)

        # 2) Propagate through atmosphere layers
        t0 = time.time()
        wf = self.pre_atmosphere_object_wavefront
        for atm_layer in self.atmosphere_layers:
            wf = atm_layer.forward(wf)
        self.post_atmosphere_wavefront = wf
        if self.report_time:
            _v2_timer("Atmosphere forward", time.time() - t0, indent=2)

        # 3) Structural differential motion
        if self.simulate_differential_motion:
            t0 = time.time()
            self._simulate_natural_diff_motion()
            if self.report_time:
                _v2_timer("Diff-motion step", time.time() - t0, indent=2)

        # 4) Segmented mirror
        t0 = time.time()
        self.pupil_wavefront = self.segmented_mirror(
            self.post_atmosphere_wavefront)
        if self.report_time:
            _v2_timer("Segments forward", time.time() - t0, indent=2)

        # 5) Deformable mirror (if modeled)
        if self.model_ao:
            t0 = time.time()
            self.post_dm_wavefront = self.dm.forward(self.pupil_wavefront)
            if self.report_time:
                _v2_timer("DM forward", time.time() - t0, indent=2)
        else:
            self.post_dm_wavefront = self.pupil_wavefront

        # 6) Propagate to focal plane
        t0 = time.time()
        self.focal_plane_wavefront = self.pupil_to_focal_propagator(
            self.post_dm_wavefront)
        if self.report_time:
            _v2_timer("Pupil-to-focal prop", time.time() - t0, indent=2)

    # ================================================================
    # SHWFS
    # ================================================================

    def get_shwfs_frame(self, integration_seconds=1.0):
        """Read a Shack-Hartmann WFS frame."""
        self.shwfs_camera.integrate(
            self.shwfs(self.magnifier(self.post_dm_wavefront)),
            integration_seconds)
        return self.shwfs_camera.read_out()

    # ================================================================
    # DM interaction-matrix calibration
    # ================================================================

    def calibrate_dm_interaction_matrix(self, env_uuid):
        """Calibrate the DM interaction matrix (with disk caching)."""
        cfg = self._cfg
        probe_amp = cfg['dm_probe_amp_fraction'] * self.wavelength

        # Take a reference image from a flat wavefront.
        wf = hcipy.Wavefront(self.aperture, self.wavelength)
        wf.total_power = 1
        self.shwfs_camera.integrate(self.shwfs(self.magnifier(wf)), 1)
        reference_image = self.shwfs_camera.read_out()

        # Refine subaperture selection based on flux.
        fluxes = ndimage.measurements.sum(
            reference_image,
            self.shwfse.mla_index,
            self.shwfse.estimation_subapertures)
        flux_limit = fluxes.max() * cfg['dm_flux_limit_fraction']
        estimation_subapertures = self.shwfs.mla_grid.zeros(dtype='bool')
        estimation_subapertures[
            self.shwfse.estimation_subapertures[fluxes > flux_limit]] = True
        self.shwfse = hcipy.ShackHartmannWavefrontSensorEstimator(
            self.shwfs.mla_grid,
            self.shwfs.micro_lens_array.mla_index,
            estimation_subapertures)

        # Compute reference slopes.
        self.reference_slopes = self.shwfse.estimate([reference_image])

        # Check for cached interaction matrix.
        dm_cache_path = os.path.join(cfg['dm_cache_dir'], str(env_uuid))
        if os.path.exists(dm_cache_path):
            _v2("Found cached interaction matrix at %s" % dm_cache_path)
            with open(os.path.join(dm_cache_path,
                      'dm_interaction_matrix.pkl'), 'rb') as f:
                self.interaction_matrix = pickle.load(f)
                return

        # Calibrate by poking each actuator.
        n_act = len(self.dm.actuators)
        _v2_subsection("DM Calibration (%d actuators)" % n_act)
        response_matrix = []
        for i in range(n_act):
            if i % 50 == 0 or i == n_act - 1:
                _v2("  actuator %d / %d" % (i + 1, n_act))
            slope = 0
            amps = [-probe_amp, probe_amp]
            for amp in amps:
                self.dm.flatten()
                self.dm.actuators[i] = amp
                dm_wf = self.dm.forward(wf)
                dm_wf.total_power = 1
                wfs_wf = self.shwfs(self.magnifier(dm_wf))
                self.shwfs_camera.integrate(wfs_wf, 1)
                img = self.shwfs_camera.read_out()
                slopes = self.shwfse.estimate([img])
                slope += amp * slopes / np.var(amps)
            response_matrix.append(slope.ravel())

        self.interaction_matrix = hcipy.ModeBasis(response_matrix)

        # Cache to disk.
        Path(dm_cache_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(dm_cache_path,
                  "dm_interaction_matrix.pkl"), 'wb') as f:
            pickle.dump(self.interaction_matrix, f)

    # ================================================================
    # Science camera readout
    # ================================================================

    def get_science_frame(self, integration_seconds=1.0):
        """Read the science camera, applying geometric-optics convolution."""
        t0 = time.time()
        self.camera.integrate(self.focal_plane_wavefront, integration_seconds)
        if self.report_time:
            _v2_timer("Camera integrate", time.time() - t0, indent=2)

        t0 = time.time()
        # Read out effective PSF (noiseless camera -> units of charge/pixel).
        effective_psf = self.camera.read_out()
        side = int(np.sqrt(effective_psf.size))
        effective_psf = effective_psf.reshape((side, side))
        self.instantaneous_psf = effective_psf
        if self.report_time:
            _v2_timer("Camera readout", time.time() - t0, indent=2)

        # Geometric-optics image formation via Fourier-domain convolution.
        t0 = time.time()
        effective_otf = np.fft.fft2(effective_psf)
        object_spectrum = np.fft.fft2(self.object_plane.array)
        image_spectrum = object_spectrum * effective_otf
        self.readout_image = np.abs(
            np.fft.fftshift(np.fft.ifft2(image_spectrum)))
        if self.report_time:
            _v2_timer("FFT convolution", time.time() - t0, indent=2)

        return self.readout_image

    # ================================================================
    # Optomechanical interaction
    # ================================================================

    def _optomechanical_interaction(self, tension_forces):
        """Map tensioner forces to PTT displacements via placeholder MLP."""
        tension_forces = np.transpose(np.array(tension_forces))
        optomech_embedding = tension_forces.dot(self._optomech_encoder)
        optomech_ptt = optomech_embedding.dot(self._optomech_decoder)
        optomech_ptt = optomech_ptt.reshape((self.num_apertures, 3))
        # Currently zeroed (interaction model not yet implemented)
        optomech_ptt = np.zeros((self.num_apertures, 3))
        # Convert to physical units (um -> m, arcsec -> rad)
        optomech_ptt[:, 0] *= 1e-6
        optomech_ptt[:, 1] *= np.pi / (180 * 3600)
        optomech_ptt[:, 2] *= np.pi / (180 * 3600)
        self._apply_ptt_displacements(ptt_displacements=optomech_ptt)

    # ================================================================
    # Structural differential motion
    # ================================================================

    def _simulate_natural_diff_motion(self):
        """Evolve structural differential motion for one time step."""
        cfg = self._cfg

        if self.model_wind_diff_motion:
            # Update wind speed via random walk.
            self.ground_wind_speed_mps += (
                np.random.randn() * self.ground_wind_speed_ms_sampled_std_mps)
            self.ground_wind_speed_mps = np.clip(
                self.ground_wind_speed_mps, 0.0, cfg["max_ground_wind_speed_mps"])

            # Compute wind-driven PTT displacements.
            ws = self.ground_wind_speed_mps
            piston_std = ws * cfg["runtime_wind_piston_micron_factor"]
            tip_std = ws * cfg["runtime_wind_tip_arcsec_factor"]
            tilt_std = ws * cfg["runtime_wind_tilt_arcsec_factor"]
            wind_ptt = np.random.randn(self.num_apertures, 3)
            wind_ptt[:, 0] *= piston_std * 1e-6  # m
            wind_ptt[:, 1] *= tip_std * np.pi / (180 * 3600)  # rad
            wind_ptt[:, 2] *= tilt_std * np.pi / (180 * 3600)  # rad
            self._apply_ptt_displacements(
                wind_ptt, incremental=True,
                incremental_factor=cfg["runtime_wind_incremental_factor"])

        if self.model_temp_diff_motion:
            # Placeholder: temperature motion not yet parameterised.
            _ = np.random.randn(self.num_apertures, 3)

        if self.model_gravity_diff_motion:
            # Placeholder: gravity motion not yet parameterised.
            _ = np.random.randn(self.num_apertures, 3)

    def _init_natural_diff_motion(self):
        """Apply initial PTT perturbations to simulate structural
        misalignment at episode start."""
        cfg = self._cfg

        if (not self.model_wind_diff_motion and
                not self.model_temp_diff_motion and
                not self.model_gravity_diff_motion):
            raise ValueError(
                "Initialized differential motion but no motion types selected.")

        if self.model_wind_diff_motion:
            piston_std = cfg["init_wind_piston_micron_std"]
            if self.command_tip_tilt:
                tip_std = cfg["init_wind_tip_arcsec_std_tt"]
                tilt_std = cfg["init_wind_tilt_arcsec_std_tt"]
            else:
                tip_std = 0.0
                tilt_std = 0.0
            wind_ptt = np.random.randn(self.num_apertures, 3)
            wind_ptt[:, 0] *= piston_std * 1e-6
            clip_m = cfg["init_wind_piston_clip_m"]
            wind_ptt[:, 0] = np.clip(wind_ptt[:, 0], -clip_m, clip_m)
            wind_ptt[:, 1] *= tip_std * np.pi / (180 * 3600)
            wind_ptt[:, 2] *= tilt_std * np.pi / (180 * 3600)
            self._apply_ptt_displacements(wind_ptt)

        if self.model_temp_diff_motion:
            # All stds currently 0.
            temp_ptt = np.random.randn(self.num_apertures, 3)
            temp_ptt[:, 0] *= 0.0  # piston
            if self.command_tip_tilt:
                temp_ptt[:, 1] *= 0.0
                temp_ptt[:, 2] *= 0.0
            else:
                temp_ptt[:, 1] *= 0.0
                temp_ptt[:, 2] *= 0.0
            self._apply_ptt_displacements(temp_ptt)

        if self.model_gravity_diff_motion:
            grav_piston_std = cfg["init_gravity_piston_micron_std"]
            if self.command_tip_tilt:
                grav_tip_std = cfg["init_gravity_tip_arcsec_std_tt"]
                grav_tilt_std = cfg["init_gravity_tilt_arcsec_std_tt"]
            else:
                grav_tip_std = 0.0
                grav_tilt_std = 0.0
            grav_ptt = np.random.randn(self.num_apertures, 3)
            grav_ptt[:, 0] *= grav_piston_std * 1e-6
            grav_ptt[:, 1] *= grav_tip_std * np.pi / (180 * 3600)
            grav_ptt[:, 2] *= grav_tilt_std * np.pi / (180 * 3600)
            self._apply_ptt_displacements(grav_ptt)

    # ================================================================
    # PTT displacement helpers
    # ================================================================

    def _apply_ptt_displacements(self, ptt_displacements,
                                  incremental=False,
                                  incremental_factor=1.0):
        """Apply piston/tip/tilt displacements to the segmented mirror.

        ptt_displacements : (num_apertures, 3) -- piston in meters, tip/tilt in radians.
        incremental       : if True, add to current state; else replace.
        incremental_factor: scale applied before adding in incremental mode.
        """
        for seg_id in range(self.num_apertures):
            if incremental:
                (seg_p, seg_t, seg_tl) = self.segmented_mirror.get_segment_actuators(seg_id)
                p = seg_p + incremental_factor * ptt_displacements[seg_id, 0]
                t = seg_t + incremental_factor * ptt_displacements[seg_id, 1]
                tl = seg_tl + incremental_factor * ptt_displacements[seg_id, 2]
            else:
                p = ptt_displacements[seg_id, 0]
                t = ptt_displacements[seg_id, 1]
                tl = ptt_displacements[seg_id, 2]
            self.segmented_mirror.set_segment_actuators(seg_id, p, t, tl)

    def get_ptt_state(self):
        """Return list of (piston, tip, tilt) tuples for all segments."""
        return [
            self.segmented_mirror.get_segment_actuators(i)
            for i in range(self.num_apertures)
        ]

    def get_displacement_correction(self):
        """Normalised [-1,1] corrections to return segments to baseline."""
        cfg = self._cfg
        ptt = self.get_ptt_state()

        max_p_m = cfg["get_disp_corr_max_piston_micron"] * 1e-6
        max_t_r = cfg["get_disp_corr_max_tip_arcsec"] * np.pi / (180 * 3600)
        max_tl_r = cfg["get_disp_corr_max_tilt_arcsec"] * np.pi / (180 * 3600)

        corrections = []
        for seg_id in range(self.num_apertures):
            cp, ct, ctl = ptt[seg_id]
            corrections.append((
                np.clip(-cp / max_p_m, -1.0, 1.0),
                np.clip(-ct / max_t_r, -1.0, 1.0),
                np.clip(-ctl / max_tl_r, -1.0, 1.0),
            ))
        return corrections

    def get_displacement_from_baseline(self):
        """Physical displacement from baseline for each segment."""
        ptt = self.get_ptt_state()
        displacements = []
        for seg_id in range(self.num_apertures):
            cp, ct, ctl = ptt[seg_id]
            bp = self.segment_baseline_dict[seg_id]["piston"]
            bt = self.segment_baseline_dict[seg_id]["tip"]
            btl = self.segment_baseline_dict[seg_id]["tilt"]
            displacements.append((cp - bp, ct - bt, ctl - btl))
        return displacements

    def _store_baseline_segment_displacements(self):
        """Snapshot current PTT state as the baseline reference."""
        self.segment_baseline_dict = {}
        for seg_id in range(self.num_apertures):
            (p, t, tl) = self.segmented_mirror.get_segment_actuators(seg_id)
            self.segment_baseline_dict[seg_id] = {
                "piston": p, "tip": t, "tilt": tl}

    # ================================================================
    # Tensioner & secondary commands
    # ================================================================

    def command_tensioners(self, tensioner_commands):
        """Apply tensioner commands via the optomechanical interaction."""
        self._optomechanical_interaction(tensioner_commands)

    def command_secondaries(self, secondaries_commands):
        """Command secondary mirror PTT for all segments.

        secondaries_commands: list of tuples, one per segment.
        Each tuple has 1 element (piston only) or 3 (piston, tip, tilt).
        Values are normalised [-1, 1] and scaled to physical limits here.
        """
        cfg = self._cfg
        max_p_m = cfg["max_piston_correction_micron"] * 1e-6
        max_t_r = cfg["max_tip_correction_arcsec"] * np.pi / (180 * 3600)
        max_tl_r = cfg["max_tilt_correction_arcsec"] * np.pi / (180 * 3600)

        self.max_piston_correction = max_p_m
        self.max_tip_correction = max_t_r
        self.max_tilt_correction = max_tl_r

        _noisy = self._actuator_noise
        _noise_frac = self._actuator_noise_fraction
        _baseline = self.segment_baseline_dict
        _seg_mirror = self.segmented_mirror

        for seg_id in range(self.num_apertures):
            seg_piston_cmd = secondaries_commands[seg_id][0]
            if len(secondaries_commands[seg_id]) == 3:
                seg_tip_cmd = secondaries_commands[seg_id][1]
                seg_tilt_cmd = secondaries_commands[seg_id][2]
            else:
                seg_tip_cmd = 0.0
                seg_tilt_cmd = 0.0

            if self.discrete_control:
                (seg_p, seg_t, seg_tl) = self.segmented_mirror.get_segment_actuators(seg_id)

                inc_p = seg_piston_cmd[0]
                dec_p = seg_piston_cmd[1]
                inc_t = seg_tip_cmd[0]
                dec_t = seg_tip_cmd[1]
                inc_tl = seg_tilt_cmd[0]
                dec_tl = seg_tilt_cmd[1]

                piston_state = seg_p + inc_p * (max_p_m / self.discrete_control_steps)
                tip_state = seg_t + inc_t * (max_t_r / self.discrete_control_steps)
                tilt_state = seg_tl + inc_tl * (max_tl_r / self.discrete_control_steps)

                piston_state = seg_p - dec_p * (max_p_m / self.discrete_control_steps)
                tip_state = seg_t - dec_t * (max_t_r / self.discrete_control_steps)
                tilt_state = seg_tl - dec_tl * (max_tl_r / self.discrete_control_steps)

                # Clip to baseline +/- max correction
                bl = self.segment_baseline_dict[seg_id]
                pre_clip = np.array([piston_state, tip_state, tilt_state])
                piston_state = np.clip(piston_state, -max_p_m + bl["piston"], max_p_m + bl["piston"])
                tip_state = np.clip(tip_state, -max_t_r + bl["tip"], max_t_r + bl["tip"])
                tilt_state = np.clip(tilt_state, -max_tl_r + bl["tilt"], max_tl_r + bl["tilt"])
                post_clip = np.array([piston_state, tip_state, tilt_state])
                self._clipped_dof_count += int(np.sum(pre_clip != post_clip))
                self._total_dof_count += 3

            elif self.incremental_control:
                p_cmd_m = seg_piston_cmd * max_p_m
                t_cmd_r = seg_tip_cmd * max_t_r
                tl_cmd_r = seg_tilt_cmd * max_tl_r

                (seg_p, seg_t, seg_tl) = self.segmented_mirror.get_segment_actuators(seg_id)
                piston_state = seg_p + p_cmd_m
                tip_state = seg_t + t_cmd_r
                tilt_state = seg_tl + tl_cmd_r

                bl = self.segment_baseline_dict[seg_id]
                pre_clip = np.array([piston_state, tip_state, tilt_state])
                piston_state = np.clip(piston_state, -max_p_m + bl["piston"], max_p_m + bl["piston"])
                tip_state = np.clip(tip_state, -max_t_r + bl["tip"], max_t_r + bl["tip"])
                tilt_state = np.clip(tilt_state, -max_tl_r + bl["tilt"], max_tl_r + bl["tilt"])
                post_clip = np.array([piston_state, tip_state, tilt_state])
                self._clipped_dof_count += int(np.sum(pre_clip != post_clip))
                self._total_dof_count += 3

            else:
                # Absolute control: baseline + scaled command.
                p_cmd_m = seg_piston_cmd * max_p_m
                t_cmd_r = seg_tip_cmd * max_t_r
                tl_cmd_r = seg_tilt_cmd * max_tl_r
                bl = self.segment_baseline_dict[seg_id]
                piston_state = bl["piston"] + p_cmd_m
                tip_state = bl["tip"] + t_cmd_r
                tilt_state = bl["tilt"] + tl_cmd_r
                pre_clip = np.array([piston_state, tip_state, tilt_state])
                piston_state = np.clip(piston_state, -max_p_m + bl["piston"], max_p_m + bl["piston"])
                tip_state = np.clip(tip_state, -max_t_r + bl["tip"], max_t_r + bl["tip"])
                tilt_state = np.clip(tilt_state, -max_tl_r + bl["tilt"], max_tl_r + bl["tilt"])
                post_clip = np.array([piston_state, tip_state, tilt_state])
                self._clipped_dof_count += int(np.sum(pre_clip != post_clip))
                self._total_dof_count += 3

            # Actuator repeatability noise: Gaussian perturbation
            # proportional to correction range (models real hardware).
            if _noisy:
                piston_state += np.random.normal(0.0, _noise_frac * 2.0 * max_p_m)
                tip_state += np.random.normal(0.0, _noise_frac * 2.0 * max_t_r)
                tilt_state += np.random.normal(0.0, _noise_frac * 2.0 * max_tl_r)
                # Re-clip after noise to stay within physical limits
                bl = _baseline[seg_id]
                piston_state = np.clip(piston_state, -max_p_m + bl["piston"], max_p_m + bl["piston"])
                tip_state = np.clip(tip_state, -max_t_r + bl["tip"], max_t_r + bl["tip"])
                tilt_state = np.clip(tilt_state, -max_tl_r + bl["tilt"], max_tl_r + bl["tilt"])

            _seg_mirror.set_segment_actuators(
                seg_id, piston_state, tip_state, tilt_state)


# ===================================================================
# OptomechEnv -- Gymnasium environment
# ===================================================================

class OptomechEnv(gym.Env):
    """Gymnasium environment for distributed-aperture telescope control.

    The environment step comprises one decision interval, during which
    multiple control commands are issued and multiple science frames are
    captured.  The multi-rate hierarchy is:

        decision_interval >= frame_interval >= control_interval >= ao_interval

    Action and observation spaces are determined at construction time
    from the enabled command surfaces (secondaries, tensioners, DM) and
    the focal-plane image size.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50,
    }

    def __init__(self, **kwargs):

        # --- Merge caller kwargs with defaults -----------------------
        self.cfg = {**DEFAULT_CONFIG, **kwargs}
        cfg = self.cfg

        # Print full configuration banner.
        _print_config_banner(cfg)

        # --- Seed & identity -----------------------------------------
        self.seed()
        self.kwargs = kwargs   # Keep raw kwargs for rebuild on reset
        self.uuid = uuid.uuid4()

        # --- Run configuration ---------------------------------------
        self.report_time = cfg['report_time']
        self.render_dpi = cfg['render_dpi']
        self.record_env_state_info = cfg['record_env_state_info']

        # --- Timing --------------------------------------------------
        self.render_frequency = cfg['render_frequency']
        self.control_interval_ms = cfg['control_interval_ms']
        self.frame_interval_ms = cfg['frame_interval_ms']
        self.decision_interval_ms = cfg['decision_interval_ms']
        self.ao_interval_ms = cfg['ao_interval_ms']

        # --- Agent action flags --------------------------------------
        self.command_tip_tilt = cfg['command_tip_tilt']
        self.command_tensioners = cfg['command_tensioners']
        self.command_secondaries = cfg['command_secondaries']
        self.command_dm = cfg['command_dm']

        # --- Action mode flags ---------------------------------------
        self.discrete_control = cfg['discrete_control']
        self.discrete_control_steps = cfg['discrete_control_steps']
        self.randomize_dm = cfg['randomize_dm']

        # --- Actuator noise (stochastic repeatability) ---------------
        self._actuator_noise = cfg['actuator_noise']
        self._actuator_noise_fraction = cfg['actuator_noise_fraction']

        # --- Action penalty ------------------------------------------
        self._action_penalty = cfg['action_penalty']
        self._action_penalty_weight = cfg['action_penalty_weight']

        # --- OOB penalty ---------------------------------------------
        self._oob_penalty = cfg['oob_penalty']
        self._oob_penalty_weight = cfg['oob_penalty_weight']

        self.reward_function = cfg['reward_function']

        # --- Composite reward weights --------------------------------
        self._reward_weight_strehl = cfg['reward_weight_strehl']
        self._reward_weight_dark_hole = cfg['reward_weight_dark_hole']
        self._reward_weight_image_quality = cfg['reward_weight_image_quality']

        self.ao_loop_active = cfg['ao_loop_active']
        self.observation_mode = cfg['observation_mode']

        # --- AO model flag -------------------------------------------
        # Determine whether we need to instantiate the AO subsystem.
        if self.command_dm or self.ao_loop_active:
            cfg['model_ao'] = True
        else:
            cfg['model_ao'] = False

        # --- DM physics (also stored at env level for AO loop) -------
        self.microns_opd_per_actuator_bit = cfg['microns_opd_per_actuator_bit']
        self.stroke_count_limit = cfg['stroke_count_limit']
        self.dm_gain = cfg['dm_gain']
        self.dm_leakage = cfg['dm_leakage']

        _v2_section("Initializing OptomechEnv")

        # --- Internal state ------------------------------------------
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self._init_state_storage()

        # --- Compute timing ratios -----------------------------------
        self._compute_timing_ratios()

        # --- Print dark hole settings --------------------------------
        if cfg.get('dark_hole', False):
            _v2_subsection("Dark Hole")
            _v2_kv("angular location (deg)", cfg.get('dark_hole_angular_location_degrees', 'N/A'))
            _v2_kv("location radius frac",   cfg.get('dark_hole_location_radius_fraction', 'N/A'))
            _v2_kv("size radius",            cfg.get('dark_hole_size_radius', 'N/A'))
        else:
            _v2("Dark hole: disabled")

        _v2_subsection("Actuator Noise")
        if self._actuator_noise:
            _v2_kv("enabled", "True")
            _v2_kv("noise fraction", "%.2e" % self._actuator_noise_fraction)
        else:
            _v2("Actuator noise: disabled")

        _v2_subsection("Action Penalty")
        if self._action_penalty:
            _v2_kv("enabled", "True")
            _v2_kv("weight", "%.4f" % self._action_penalty_weight)
        else:
            _v2("Action penalty: disabled")

        _v2_subsection("OOB Penalty")
        if self._oob_penalty:
            _v2_kv("enabled", "True")
            _v2_kv("weight", "%.4f" % self._oob_penalty_weight)
        else:
            _v2("OOB penalty: disabled")

        _v2_subsection("Reward")
        _v2_kv("function", self.reward_function)
        if self.reward_function == "composite":
            _v2_kv("weight_strehl", "%.3f" % self._reward_weight_strehl)
            _v2_kv("weight_dark_hole", "%.3f" % self._reward_weight_dark_hole)
            _v2_kv("weight_image_quality", "%.3f" % self._reward_weight_image_quality)

        # --- Build optical system ------------------------------------
        self._build_optical_system()

        # --- Episode clock -------------------------------------------
        self.episode_time_ms = 0.0

        # --- Build action spaces -------------------------------------
        self._build_action_spaces()

        # --- Build observation space ---------------------------------
        self._build_observation_space()

        # --- Cache reward images -------------------------------------
        self._cache_reward_images()

        # --- Reward dispatch -----------------------------------------
        self._reward_dispatch = {
            "composite": self._reward_composite,
            "strehl": self._reward_strehl,
            "align": self._reward_align,
            "dark_hole": self._reward_dark_hole,
            "image_mse": self._reward_image_mse,
            "negastrehl": self._reward_negastrehl,
            "negaexpstrehl": self._reward_negaexpstrehl,
            "strehl_closed": self._reward_strehl_closed,
            "ao_rms_slope": self._reward_ao_rms_slope,
            "norm_ao_rms_slope": self._reward_norm_ao_rms_slope,
            "ao_closed": self._reward_ao_closed,
        }
        if self.reward_function not in self._reward_dispatch:
            raise ValueError("Unknown reward_function: '%s'" % self.reward_function)
        self._reward_fn = self._reward_dispatch[self.reward_function]

        # --- Environment-level state storage -------------------------
        self.state_content["wavelength"] = self.optical_system.wavelength

        _v2_section("Environment Ready")

    # ================================================================
    # Timing
    # ================================================================

    def _compute_timing_ratios(self):
        """Derive the multi-rate control hierarchy from intervals."""
        cfg = self.cfg
        _v2_subsection("Timing Hierarchy")
        _v2_kv("control  interval", "%.2f ms" % self.control_interval_ms)
        _v2_kv("frame    interval", "%.2f ms" % self.frame_interval_ms)
        _v2_kv("decision interval", "%.2f ms" % self.decision_interval_ms)
        _v2_kv("AO       interval", "%.2f ms" % self.ao_interval_ms)

        self.commands_per_decision = math.ceil(
            self.decision_interval_ms / self.control_interval_ms)
        self.commands_per_frame = math.ceil(
            self.frame_interval_ms / self.control_interval_ms)
        self.frames_per_decision = math.ceil(
            self.decision_interval_ms / self.frame_interval_ms)
        self.ao_steps_per_command = math.ceil(
            self.control_interval_ms / self.ao_interval_ms)
        self.ao_steps_per_frame = (
            self.ao_steps_per_command * self.commands_per_frame)

        # Expose in metadata for downstream consumers.
        self.metadata['commands_per_decision'] = self.commands_per_decision
        self.metadata['commands_per_frame'] = self.commands_per_frame
        self.metadata['frames_per_decision'] = self.frames_per_decision
        self.metadata['ao_steps_per_command'] = self.ao_steps_per_command
        self.metadata['ao_steps_per_frame'] = self.ao_steps_per_frame

        _v2_subsection("Derived Rates")
        _v2_kv("commands / decision", "%d" % self.commands_per_decision)
        _v2_kv("commands / frame",    "%d" % self.commands_per_frame)
        _v2_kv("AO steps / frame",    "%d" % self.ao_steps_per_frame)
        _v2_kv("frames / decision",   "%d" % self.frames_per_decision)

    # ================================================================
    # Cache helpers
    # ================================================================

    def _cache_reward_images(self):
        """Pre-compute normalized perfect image and target for rewards."""
        pi = self.optical_system.perfect_image
        pi_max = np.max(pi)
        self._norm_perfect_image = pi / pi_max if pi_max != 0 else pi

        ti = self.optical_system.target_image
        ti_max = np.max(ti)
        self._norm_target_image = ti / ti_max if ti_max != 0 else ti

        # For dark_hole reward: mask where target is near-zero
        if self.cfg.get('dark_hole', False):
            self._target_zero_mask = self._norm_target_image < 1e-12
        else:
            self._target_zero_mask = None

        # Use raw HCIPy intensity directly (no detector model).
        # The detector model saturates every pixel to max_dn because the raw
        # power values are enormous.  Since MSE now normalizes by each image's
        # own max, we just need the correct PSF *shape*.
        _pi_2d = np.array(pi).reshape(self.image_shape)
        self._perfect_image_dn = _pi_2d
        self._perfect_image_max_dn = float(np.max(self._perfect_image_dn))
        _v2_kv("perfect_image_max_dn", "%.4f" % self._perfect_image_max_dn)

    # ================================================================
    # State storage
    # ================================================================

    def _init_state_storage(self):
        """Initialize the state content dictionary."""
        self.state_content = {
            "dm_surfaces": [],
            "atmos_layer_0_list": [],
            "action_times": [],
            "object_fields": [],
            "pre_atmosphere_object_wavefronts": [],
            "post_atmosphere_wavefronts": [],
            "segmented_mirror_surfaces": [],
            "pupil_wavefronts": [],
            "post_dm_wavefronts": [],
            "focal_plane_wavefronts": [],
            "readout_images": [],
            "instantaneous_psf": [],
        }

    # ================================================================
    # Optical system builder
    # ================================================================

    def _build_optical_system(self):
        """(Re-)create the OpticalSystem from the stored config."""
        self.optical_system = OpticalSystem(**self.cfg)

    def build_optical_system(self, **kwargs):
        """Public interface matching v1 for compatibility."""
        merged = {**self.cfg, **kwargs}
        self.optical_system = OpticalSystem(**merged)

    # ================================================================
    # Action spaces
    # ================================================================

    def _build_action_spaces(self):
        """Construct the hierarchical Tuple action space, then flatten it
        to a Box(-1, 1) for standard RL algorithms."""
        command_space_list = []

        if self.command_secondaries:
            command_space_list.append(
                self._build_secondaries_space())

        if self.command_tensioners:
            command_space_list.append(
                self._build_tensioners_space())

        if self.command_dm:
            command_space_list.append(
                self._build_dm_space())

        single_command_space = spaces.Tuple(tuple(command_space_list))
        self.dict_action_space = spaces.Tuple(
            [single_command_space] * self.commands_per_decision)

        # Build an anytree for address translation.
        self.action_tree = self._build_tree_from_action_space(
            self.dict_action_space)

        # Flatten to a simple Box or MultiDiscrete.
        if self.discrete_control:
            flat = spaces.MultiDiscrete(
                [1] * self._get_vector_action_size(self.dict_action_space))
            self.action_space = flat
        else:
            self.action_space = self._flatten(
                self.dict_action_space, flat_space_low=-1.0, flat_space_high=1.0)

        # Build the zero-action space (for reset steps).
        zero_cmd_list = []
        if self.command_secondaries:
            zero_cmd_list.append(self._build_secondaries_space(zero=True))
        if self.command_tensioners:
            zero_cmd_list.append(self._build_tensioners_space(zero=True))
        if self.command_dm:
            zero_cmd_list.append(self._build_dm_space(zero=True))

        zero_single = spaces.Tuple(tuple(zero_cmd_list))
        self.zero_dict_action_space = spaces.Tuple(
            [zero_single] * self.commands_per_decision)
        self.zero_action_space = self._flatten(
            self.zero_dict_action_space, flat_space_low=0.0, flat_space_high=0.0)

    def _build_secondaries_space(self, zero=False):
        """Build the secondaries Tuple(Tuple(Box|Discrete)) space."""
        lo = 0.0 if zero else -1.0
        hi = 0.0 if zero else 1.0

        if self.discrete_control and not zero:
            ptt_list = [spaces.Tuple((spaces.Discrete(1), spaces.Discrete(1)))]
            if self.command_tip_tilt:
                ptt_list.append(spaces.Tuple((spaces.Discrete(1), spaces.Discrete(1))))
                ptt_list.append(spaces.Tuple((spaces.Discrete(1), spaces.Discrete(1))))
        else:
            ptt_list = [spaces.Box(low=lo, high=hi, shape=(1,), dtype=np.float32)]
            if self.command_tip_tilt:
                ptt_list.append(spaces.Box(low=lo, high=hi, shape=(1,), dtype=np.float32))
                ptt_list.append(spaces.Box(low=lo, high=hi, shape=(1,), dtype=np.float32))

        ptt_space = spaces.Tuple(tuple(ptt_list))
        return spaces.Tuple(
            tuple([ptt_space] * self.optical_system.num_apertures))

    def _build_tensioners_space(self, zero=False):
        """Build the tensioners Tuple(Box) space."""
        lo = 0.0 if zero else -1.0
        hi = 0.0 if zero else 1.0
        t_space = spaces.Box(low=lo, high=hi, shape=(1,), dtype=np.float32)
        return spaces.Tuple(
            tuple([t_space] * self.optical_system.num_tensioners))

    def _build_dm_space(self, zero=False):
        """Build the DM Tuple(Box) space."""
        lo = 0.0 if zero else -1.0
        hi = 0.0 if zero else 1.0
        if zero:
            dm_space = spaces.Box(
                low=lo, high=hi,
                shape=(len(self.optical_system.dm.actuators),),
                dtype=np.float32)
            return spaces.Tuple((dm_space,))
        else:
            act_space = spaces.Box(low=lo, high=hi, shape=(1,), dtype=np.float32)
            return spaces.Tuple(
                tuple([act_space] * len(self.optical_system.dm.actuators)))

    # ================================================================
    # Observation space
    # ================================================================

    def _build_observation_space(self):
        """Define the observation space (image stack +/- prior action)."""
        cfg = self.cfg
        self.image_shape = (cfg['focal_plane_image_size_pixels'],
                            cfg['focal_plane_image_size_pixels'])
        image_stack_shape = (self.frames_per_decision,
                             self.image_shape[0], self.image_shape[1])
        self.image_space = spaces.Box(
            low=0.0, high=1.0, shape=image_stack_shape, dtype=np.float32)

        if self.observation_mode == "image_only":
            self.observation_space = self.image_space
        elif self.observation_mode == "image_action":
            self.observation_space = spaces.Dict(
                {"image": self.image_space,
                 "prior_action": self.action_space},
                seed=42)
        else:
            raise ValueError(
                "Invalid observation_mode: '%s'. Use 'image_only' or "
                "'image_action'." % self.observation_mode)

    # ================================================================
    # Action space helpers
    # ================================================================

    def _flatten(self, dict_space, flat_space_high=1.0, flat_space_low=0.0):
        """Flatten a hierarchical Tuple space to a Box."""
        return spaces.Box(
            low=flat_space_low, high=flat_space_high,
            shape=(self._get_vector_action_size(dict_space),),
            dtype=np.float32)

    def _flat_to_dict(self, flat_action, dict_space):
        """Convert a flat action vector to the hierarchical Tuple."""
        return self._encode_action_from_vector(dict_space, flat_action)

    @staticmethod
    def _tuple_to_list(tup):
        if isinstance(tup, tuple):
            return [OptomechEnv._tuple_to_list(i) for i in tup]
        return tup

    @staticmethod
    def _list_to_tuple(lst):
        if isinstance(lst, list):
            return tuple(OptomechEnv._list_to_tuple(i) for i in lst)
        return lst

    def _build_tree_from_action_space(self, action_space):
        """Build an anytree Node tree from a hierarchical action space.

        Each leaf node stores its linear address and its tree address
        string (e.g. '0_0_2_1').  The root node stores the total leaf
        count and a dict mapping linear_address -> tree_address.
        """
        root = Node("action", content="")
        linear_address = 0
        linear_to_tree = {}

        for step_num, step in enumerate(action_space):
            step_node = Node(f"step_{step_num}", parent=root, content="")
            for stage_num, stage in enumerate(step):
                stage_node = Node(f"stage_{stage_num}", parent=step_node, content="")
                for comp_num, comp in enumerate(stage):
                    comp_node = Node(f"component_{comp_num}", parent=stage_node, content="")
                    if hasattr(comp, '__iter__'):
                        for cmd_num, cmd in enumerate(comp):
                            if hasattr(cmd, '__iter__'):
                                for elem_num, elem in enumerate(cmd):
                                    addr = f"{step_num}_{stage_num}_{comp_num}_{cmd_num}_{elem_num}"
                                    linear_to_tree[linear_address] = addr
                                    linear_address += 1
                                    Node(f"element_{elem_num}",
                                         parent=comp_node, content=elem,
                                         action_space_address=addr,
                                         linear_address=linear_address)
                            else:
                                addr = f"{step_num}_{stage_num}_{comp_num}_{cmd_num}"
                                linear_to_tree[linear_address] = addr
                                Node(f"command_{cmd_num}",
                                     parent=comp_node, content=cmd,
                                     action_space_address=addr,
                                     linear_address=linear_address)
                                linear_address += 1
                    else:
                        addr = f"{step_num}_{stage_num}_{comp_num}_{0}"
                        Node(f"command_0",
                             parent=comp_node, content=comp,
                             action_space_address=addr,
                             linear_address=linear_address)
                        linear_to_tree[linear_address] = addr
                        linear_address += 1

        root.num_leaf_nodes = linear_address
        root.linear_to_tree_dict = linear_to_tree
        return root

    def _encode_action_from_vector(self, action_space, action_vector):
        """Map a flat action vector into the hierarchical Tuple structure."""
        try:
            tree = self.action_tree
        except AttributeError:
            raise Warning("No action tree found. Building from space.")
            tree = self._build_tree_from_action_space(action_space)

        action_list = self._tuple_to_list(action_space.sample())
        for n, val in enumerate(action_vector):
            indices = list(map(int, tree.linear_to_tree_dict[n].split('_')))
            sub = action_list
            for idx in indices[:-1]:
                sub = sub[idx]
            sub[indices[-1]] = val
        return self._list_to_tuple(action_list)

    def _get_vector_action_size(self, hierarchical_space):
        """Return the total number of leaf nodes (flat action size)."""
        try:
            tree = self.action_tree
        except AttributeError:
            tree = self._build_tree_from_action_space(hierarchical_space)
        return tree.num_leaf_nodes

    # ================================================================
    # Seed
    # ================================================================

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # ================================================================
    # Reset
    # ================================================================

    def reset(self, seed=None, options=None):
        _v2_section("Episode Reset")
        _v2("Rebuilding optical system...")
        self._build_optical_system()

        if self.command_dm or self.ao_loop_active:
            self.optical_system.calibrate_dm_interaction_matrix(self.uuid)
            rcond = self.cfg['dm_interaction_rcond']
            self.reconstruction_matrix = hcipy.inverse_tikhonov(
                self.optical_system.interaction_matrix.transformation_matrix,
                rcond=rcond)
            self.episode_time_ms = 0.0

        _v2("Seeding initial action...")

        # Re-apply initial differential motion if configured.
        if self.cfg['init_differential_motion']:
            _v2("Applying initial differential motion...")
            self.optical_system._init_natural_diff_motion()
        self.optical_system._store_baseline_segment_displacements()

        # Warm up with initial steps.
        _v2("Warm-up: %d initial frame(s)..." % self.frames_per_decision)
        for _ in range(self.frames_per_decision):
            (initial_state, _, _, _, info) = self.step(
                action=self.action_space.sample(),
                noisy_command=False,
                reset=True)

        self.state = initial_state
        self.steps_beyond_done = None
        _v2_section("Reset Complete")
        return self.state, info

    # ================================================================
    # Save state
    # ================================================================

    def save_state(self):
        """Deepcopy and store all optical-system state variables."""
        if self.report_time:
            t0 = time.time()

        if self.optical_system.model_ao:
            self.state_content["dm_surfaces"].append(
                copy.deepcopy(self.optical_system.dm.surface))

        if len(self.optical_system.atmosphere_layers) > 0:
            self.state_content["atmos_layer_0_list"].append(
                copy.deepcopy(self.optical_system.atmosphere_layers[0]))
        else:
            self.state_content["atmos_layer_0_list"].append(None)

        self.state_content["object_fields"].append(
            copy.deepcopy(self.optical_system.object_plane))
        self.state_content["pre_atmosphere_object_wavefronts"].append(
            copy.deepcopy(self.optical_system.pre_atmosphere_object_wavefront))
        self.state_content["post_atmosphere_wavefronts"].append(
            copy.deepcopy(self.optical_system.post_atmosphere_wavefront))
        self.state_content["segmented_mirror_surfaces"].append(
            copy.deepcopy(self.optical_system.segmented_mirror.surface))
        self.state_content["pupil_wavefronts"].append(
            copy.deepcopy(self.optical_system.pupil_wavefront))
        self.state_content["post_dm_wavefronts"].append(
            copy.deepcopy(self.optical_system.post_dm_wavefront))
        self.state_content["focal_plane_wavefronts"].append(
            copy.deepcopy(self.optical_system.focal_plane_wavefront))
        self.state_content["instantaneous_psf"].append(
            copy.deepcopy(self.optical_system.instantaneous_psf))
        self.state_content["readout_images"].append(
            copy.deepcopy(self.science_readout_raster))

        if self.report_time:
            _v2_timer("State deepcopy", time.time() - t0)

    # ================================================================
    # Step
    # ================================================================

    def step(self, action, noisy_command=False, reset=False):
        """Execute one decision interval of the environment.

        Returns (observation, reward, terminated, truncated, info).
        """
        if self.report_time:
            step_t0 = time.time()

        # Validate the action.
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))

        # Clear state recording buffers.
        if self.record_env_state_info:
            for key in self.state_content:
                if isinstance(self.state_content[key], list):
                    self.state_content[key] = []
            self.state_content["shwfs_slopes"] = []

        # Decode flat action -> hierarchical Tuple.
        self.action = self._flat_to_dict(action, self.dict_action_space)

        self.focal_plane_images = []
        self.shwfs_slopes_list = []
        self.optical_system._clipped_dof_count = 0
        self.optical_system._total_dof_count = 0

        # --- Main simulation loop ------------------------------------
        for frame_num in range(self.frames_per_decision):
            frame = np.zeros(self.image_shape, dtype=np.float32)

            for command_num in range(self.commands_per_frame):
                self.episode_time_ms += self.control_interval_ms

                # Evolve atmosphere.
                t0 = time.time()
                self.optical_system.evolve_atmosphere_to(self.episode_time_ms)
                if self.report_time:
                    _v2_timer("Atmosphere evolve", time.time() - t0)

                # Apply commands.
                command = self.action[command_num]
                self._apply_commands(command)

                # Integration time per AO sub-step.
                frame_sec = self.frame_interval_ms / 1000.0
                integration_sec = frame_sec / self.ao_steps_per_frame

                # AO loop iterations.
                for ao_step in range(self.ao_steps_per_command):
                    wavelengths = self._get_wavelengths()
                    for wl in wavelengths:
                        if self.report_time:
                            sim_t0 = time.time()

                        # Full optical simulation at this wavelength.
                        self.optical_system.simulate(wl)
                        if self.report_time:
                            _v2_timer("Optical sim", time.time() - sim_t0, indent=2)

                        # Closed-loop AO correction.
                        if self.report_time:
                            ao_t0 = time.time()
                        if self.ao_loop_active and not reset:
                            self._run_ao_correction(integration_sec)
                        if self.report_time:
                            _v2_timer("AO correction", time.time() - ao_t0, indent=2)

                        # Read science frame.
                        science = self.optical_system.get_science_frame(
                            integration_seconds=integration_sec)
                        self.science_readout_raster = np.reshape(
                            science, self.image_shape)

                        # Accumulate (manual integration).
                        frame += self.science_readout_raster / float(len(wavelengths))

                    # Save state if recording.
                    if self.record_env_state_info and not reset:
                        self.save_state()

            # --- Detector model: power -> photons -> electrons -> DN --
            frame = self._apply_detector_model(frame)
            self.focal_plane_images.append(frame)

        # --- Encode observation --------------------------------------
        if self.observation_mode == "image_only":
            self.state = np.array(self.focal_plane_images)
        elif self.observation_mode == "image_action":
            self.state = {'image': np.array(self.focal_plane_images),
                          'prior_action': action}
        else:
            raise ValueError("Invalid observation_mode: '%s'" % self.observation_mode)

        # --- Compute reward ------------------------------------------
        reward = self._compute_reward()

        # L2 action penalty: reward -= w * |reward| * mean(a^2).
        # With actions in [-1, 1], mean(a^2) ∈ [0, 1].
        # Default w = 0.03 → 3% worse at full-magnitude action,
        # 0% at zero action. Works for both positive and negative rewards.
        if self._action_penalty:
            action_sq_mean = float(np.mean(np.square(action)))
            reward = reward - self._action_penalty_weight * abs(reward) * action_sq_mean

        # Out-of-bounds penalty: penalize actuator clipping at physical rail.
        # Fraction of DOFs that were clipped by np.clip in command_secondaries().
        oob_frac = 0.0
        if self._oob_penalty and self.optical_system._total_dof_count > 0:
            oob_frac = float(self.optical_system._clipped_dof_count) / float(self.optical_system._total_dof_count)
            reward = reward - self._oob_penalty_weight * oob_frac

        # --- Termination / truncation --------------------------------
        terminated = False
        truncated = False

        # --- Diagnostic metrics (always reported) --------------------
        strehls = [self._absolute_strehl(fpi)
                   for fpi in self.focal_plane_images]
        _max_dn = self._perfect_image_max_dn
        _perfect_flat = self._perfect_image_dn.flatten()
        mses = [float(np.mean((fpi.flatten() - _perfect_flat) ** 2)) / (_max_dn ** 2)
                for fpi in self.focal_plane_images]

        # --- Info dict -----------------------------------------------
        info = {}
        info["strehl"] = float(np.mean(strehls))
        info["mse"] = float(np.mean(mses))
        info["reward_raw"] = float(reward)
        info["oob_frac"] = oob_frac
        if self.record_env_state_info:
            info["state_content"] = self.state_content
            info["state"] = self.state

        if self.report_time:
            _v2_timer("STEP TOTAL", time.time() - step_t0)

        reward = np.float32(reward)
        return self.state, reward, terminated, truncated, info

    # ================================================================
    # Command dispatch
    # ================================================================

    def _apply_commands(self, command):
        """Dispatch the hierarchical command tuple to the optical system.

        The command indices depend on which surfaces are enabled and the
        order they were appended in _build_action_spaces:
            [0] = secondaries (if enabled)
            [next] = tensioners (if enabled)
            [next] = dm (if enabled)

        Note: execution order matches v1: tensioners first, then
        secondaries, then dm.  The *index* into the command tuple is
        determined by construction order (secondaries, tensioners, dm).
        """
        # Pre-compute index for each surface based on construction order.
        idx = 0
        sec_idx = ten_idx = dm_idx = None
        if self.command_secondaries:
            sec_idx = idx
            idx += 1
        if self.command_tensioners:
            ten_idx = idx
            idx += 1
        if self.command_dm:
            dm_idx = idx
            idx += 1

        # Execute in v1 order: tensioners, secondaries, dm.
        if self.command_tensioners:
            self.optical_system.command_tensioners(command[ten_idx])
        if self.command_secondaries:
            self.optical_system.command_secondaries(command[sec_idx])
        if self.command_dm:
            self.optical_system.command_dm(command[dm_idx])

    # ================================================================
    # Wavelength sampling
    # ================================================================

    def _get_wavelengths(self):
        """Return the list of wavelengths for polychromatic simulation."""
        return _centered_wavelengths(
            self.optical_system.wavelength,
            self.cfg['bandwidth_nanometers'],
            self.cfg['bandwidth_sampling'])

    # ================================================================
    # AO loop
    # ================================================================

    def _run_ao_correction(self, integration_seconds):
        """One iteration of closed-loop Shack-Hartmann AO."""
        shwfs_vec = self.optical_system.get_shwfs_frame(
            integration_seconds=integration_seconds)
        slopes = self.optical_system.shwfse.estimate([shwfs_vec + 1e-10])
        slopes -= self.optical_system.reference_slopes
        self.shwfs_slopes = slopes.ravel()
        self.shwfs_slopes_list.append(self.shwfs_slopes)

        # Leaky integrator update.
        self.optical_system.dm.actuators = (
            (1 - self.dm_leakage) * self.optical_system.dm.actuators
            - self.dm_gain * self.reconstruction_matrix.dot(self.shwfs_slopes))

        # Clip to physical stroke limits.
        stroke_limit = (
            self.microns_opd_per_actuator_bit
            * self.stroke_count_limit * 1e-6 / 2)
        self.optical_system.dm.actuators = np.clip(
            self.optical_system.dm.actuators, -stroke_limit, stroke_limit)

    # ================================================================
    # Detector model
    # ================================================================

    def _apply_detector_model(self, frame):
        """Convert power-per-pixel frame to digital numbers (DN).

        Pipeline: power -> energy -> photons -> electrons -> DN.
        """
        cfg = self.cfg
        h = 6.62607015e-34   # Planck constant (J*s)
        c = 2.99792458e8     # Speed of light (m/s)
        qe = cfg['detector_quantum_efficiency']
        gain = cfg['detector_gain_e_per_dn']
        max_dn = cfg['detector_max_dn']
        frame_sec = self.frame_interval_ms / 1000.0

        energy_joules = frame * frame_sec
        photon_energy = h * c / self.optical_system.wavelength
        n_photons = energy_joules / photon_energy
        n_electrons = n_photons * qe
        dn = n_electrons / gain
        return np.clip(dn, 0, max_dn)

    # ================================================================
    # Reward computation
    # ================================================================
    #
    # All Strehl-based rewards use ABSOLUTE Strehl:
    #   Strehl = max(observed_dn) / max(perfect_dn)
    # Both images in detector DN units.  No self-normalization.

    def _compute_reward(self):
        """Dispatch to the configured reward function."""
        return self._reward_fn()

    def _absolute_strehl(self, fpi):
        """Absolute Strehl ratio: peak of observed / peak of perfect (both DN)."""
        return float(np.max(fpi)) / self._perfect_image_max_dn

    def _reward_composite(self):
        """Weighted multi-factor reward: Strehl + dark_hole + image_quality."""
        total = 0.0
        w_s = self._reward_weight_strehl
        w_dh = self._reward_weight_dark_hole
        w_iq = self._reward_weight_image_quality

        if w_s > 0:
            strehls = [self._absolute_strehl(fpi)
                       for fpi in self.focal_plane_images]
            total += w_s * np.mean(strehls, dtype=np.float32)

        if w_dh > 0 and self._target_zero_mask is not None:
            dh_vals = []
            for fpi in self.focal_plane_images:
                dh_intensity = float(np.mean(fpi[self._target_zero_mask]))
                dh_vals.append(-dh_intensity / self._perfect_image_max_dn)
            total += w_dh * np.mean(dh_vals, dtype=np.float32)

        if w_iq > 0:
            perfect_dn_flat = self._perfect_image_dn.flatten()
            max_dn_sq = self._perfect_image_max_dn ** 2
            iq_vals = []
            for fpi in self.focal_plane_images:
                mse = -float(np.mean((fpi.flatten() - perfect_dn_flat) ** 2))
                iq_vals.append(mse / max_dn_sq)
            total += w_iq * np.mean(iq_vals, dtype=np.float32)

        return np.float32(total)

    # --- Individual reward methods -----------------------------------

    def _reward_strehl(self):
        """Absolute Strehl ratio (no self-normalization)."""
        strehls = [self._absolute_strehl(fpi)
                   for fpi in self.focal_plane_images]
        return np.mean(strehls, dtype=np.float32)

    def _reward_negastrehl(self):
        strehls = [self._absolute_strehl(fpi)
                   for fpi in self.focal_plane_images]
        return np.float32(np.mean(strehls) - 1.0)

    def _reward_negaexpstrehl(self):
        strehls = [self._absolute_strehl(fpi)
                   for fpi in self.focal_plane_images]
        return np.float32(np.mean(strehls) ** 10 - 1.0)

    def _reward_strehl_closed(self):
        strehls = [self._absolute_strehl(fpi)
                   for fpi in self.focal_plane_images]
        return 1.0 if np.mean(strehls) >= 0.8 else 0.0

    def _reward_image_mse(self):
        """Negative MSE vs perfect image (both normalised by perfect max)."""
        _max_dn = self._perfect_image_max_dn
        norm_ideal = self._perfect_image_dn.flatten() / _max_dn
        mses = []
        for fpi in self.focal_plane_images:
            norm_img = fpi.flatten() / _max_dn
            mses.append(float(np.mean((norm_img - norm_ideal) ** 2)))
        return -np.mean(mses, dtype=np.float32)

    def _reward_align(self):
        """Log-MSE alignment reward using absolute normalization.

        Note: In v1 the radius is set once before the loop from
        radius_max (which is always None at call time, since it is a
        local).  radius_max may be set inside the loop, but this has no
        effect because radius is not re-read.  We replicate that
        behavior exactly here.
        """
        _max_dn = self._perfect_image_max_dn
        norm_target = self._perfect_image_dn / _max_dn
        radius_max = None

        # Radius is evaluated once before the frame loop (v1 behavior).
        if not radius_max:
            radius = self.cfg['align_radius']
        else:
            radius = radius_max

        # Build center mask (v2 uses Python double-loop, preserved for compatibility)
        center_mask = np.zeros_like(norm_target, dtype=bool)
        ctr = (center_mask.shape[0] // 2, center_mask.shape[1] // 2)
        for i in range(center_mask.shape[0]):
            for j in range(center_mask.shape[1]):
                if (i - ctr[0])**2 + (j - ctr[1])**2 <= radius**2:
                    center_mask[i, j] = True

        rewards = []
        for fpi in self.focal_plane_images:
            norm_img = fpi / _max_dn
            loss_img = np.round(65534.0 * norm_img) + 1
            loss_tgt = np.round(65534.0 * norm_target) + 1

            mse = np.power(
                np.log(loss_img[center_mask]) - np.log(loss_tgt[center_mask]), 2)
            mse = -np.mean(mse.flatten())

            if mse < self.cfg['align_mse_expand_threshold']:
                radius_max = self.cfg['align_radius_max_expand']

            rewards.append(mse)

        return np.mean(rewards, dtype=np.float32)

    def _reward_dark_hole(self):
        """Dark hole reward using absolute normalization."""
        _max_dn = self._perfect_image_max_dn
        norm_target = self._perfect_image_dn / _max_dn
        target_mask = self._target_zero_mask
        radius_max = None
        alpha_hole = self.cfg['dark_hole_alpha']

        # Radius is evaluated once before the frame loop (v1 behavior).
        if not radius_max:
            radius = self.cfg['align_radius']
        else:
            radius = radius_max

        # Build center mask (v2 uses Python double-loop, preserved for compatibility)
        center_mask = np.zeros_like(norm_target, dtype=bool)
        ctr = (center_mask.shape[0] // 2, center_mask.shape[1] // 2)
        for i in range(center_mask.shape[0]):
            for j in range(center_mask.shape[1]):
                if (i - ctr[0])**2 + (j - ctr[1])**2 <= radius**2:
                    center_mask[i, j] = True

        rewards = []
        for fpi in self.focal_plane_images:
            norm_img = fpi / _max_dn
            loss_img = np.round(65534.0 * norm_img) + 1
            loss_tgt = np.round(65534.0 * norm_target) + 1

            mse = -np.mean(np.power(
                np.log(loss_img[center_mask]) - np.log(loss_tgt[center_mask]),
                2).flatten())

            if mse < self.cfg['align_mse_expand_threshold']:
                radius_max = self.cfg['align_radius_max_expand']

            dh_mse = -np.mean(
                np.power(np.log(fpi[target_mask] + 1e-30), 2).flatten())

            reward = alpha_hole * dh_mse + (1.0 - alpha_hole) * mse
            rewards.append(reward)

        return np.mean(rewards, dtype=np.float32)

    def _reward_ao_rms_slope(self):
        reward = 0.0
        if self.ao_loop_active:
            for slopes in self.shwfs_slopes_list:
                reward += 1 / np.sqrt(np.mean(slopes ** 2))
        return reward

    def _reward_norm_ao_rms_slope(self):
        reward = 0.0
        if self.ao_loop_active:
            for slopes in self.shwfs_slopes_list:
                reward += 1 / np.sqrt(np.mean(slopes ** 2))
            reward /= 1e7
        return reward

    def _reward_ao_closed(self):
        threshold = self.cfg['ao_closed_inv_slope_threshold']
        inv_rms = 0.0
        if self.ao_loop_active:
            for slopes in self.shwfs_slopes_list:
                inv_rms += 1 / np.sqrt(np.mean(slopes ** 2))
            return 1.0 if inv_rms >= threshold else 0.0
        return 0.0

    # ================================================================
    # Close
    # ================================================================

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
