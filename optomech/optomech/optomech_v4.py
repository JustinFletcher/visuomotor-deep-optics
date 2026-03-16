"""
Optomech V4 -- GPU-accelerated Gymnasium environment for distributed-aperture
telescope control.

This is a performance-optimized version of optomech_v3.py that bypasses HCIPy
in the hot path by extracting raw arrays at init time and running the optical
simulation as PyTorch tensor operations on the best available device:

  - Metal/MPS on Mac  (~10x speedup)
  - CUDA on NVIDIA GPUs (~30-100x speedup)
  - CPU fallback (~3x speedup from eliminated overhead)

HCIPy is still used for:
  - Aperture construction and segment layout
  - DM interaction matrix calibration (disk-cached)
  - Atmosphere evolution (Markov chain phase screen generation)
  - SHWFS forward propagation (when AO loop is active)

The hot path (simulate + get_science_frame) is replaced with:
  1. Fused optics kernel: aperture * exp(j*(atm + 2k*seg + 2k*dm))
  2. FFT propagation: exact replica of HCIPy's FastFourierTransform
  3. PSF + science image: |E|^2 * weights → FFT convolution with object

All intermediate tensors live on the device. Only one .cpu().numpy()
transfer occurs per frame (at the end, for observation/reward).

Functionally identical to v3 -- same action/observation spaces, same reward
values given identical seeds and configurations.

Author: Justin Fletcher (original), refactored for v2, optimized v3, v4 GPU.
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
import torch
import scipy.sparse

from collections import deque
from pathlib import Path

from scipy import signal
import scipy.ndimage as ndimage
import scipy.fft as sp_fft

from PIL import Image, ImageSequence
from gymnasium import spaces, logger
from gymnasium.utils import seeding

from matplotlib import pyplot as plt
from matplotlib import image
from matplotlib.figure import Figure

import astropy.units as u
import hcipy
from hcipy.fourier.fast_fourier_transform import FastFourierTransform
from hcipy.fourier.matrix_fourier_transform import MatrixFourierTransform


# ===================================================================
# DEFAULT CONFIGURATION
# ===================================================================

DEFAULT_CONFIG = {

    # --- Run / debug -------------------------------------------------
    "report_time": False,
    "render": False,
    "render_frequency": 1,
    "render_dpi": 500.0,
    "silence": False,

    # --- Aperture ----------------------------------------------------
    "aperture_type": "elf",
    "num_tensioners": 16,

    # --- Object plane ------------------------------------------------
    "object_type": "binary",
    "object_plane_extent_meters": 1.0,
    "object_plane_distance_meters": 1.0,

    # --- Focal plane / grid ------------------------------------------
    "focal_plane_image_size_pixels": 256,

    # --- Optical simulation ------------------------------------------
    "wavelength": 1000e-9,
    "oversampling_factor": 8,
    "bandwidth_nanometers": 200.0,
    "bandwidth_sampling": 2,

    # --- Atmosphere --------------------------------------------------
    "num_atmosphere_layers": 0,
    "seeing_arcsec": 0.5,
    "outer_scale_meters": 40.0,
    "tau0_seconds": 10.0,

    # --- Structural differential motion ------------------------------
    "init_differential_motion": False,
    "simulate_differential_motion": False,
    "model_wind_diff_motion": False,
    "model_gravity_diff_motion": False,
    "model_temp_diff_motion": False,
    "initial_ground_wind_speed_mps": 3.0,
    "ground_wind_speed_std_fraction": 0.08,
    "max_ground_wind_speed_mps": 20.0,
    "initial_ground_temp_degcel": 20.0,
    "ground_temp_ms_sampled_std": 0.0,
    "initial_gravity_normal_deg": 45.0,
    "gravity_normal_ms_sampled_std": 0.0,
    "init_wind_piston_micron_std": 1.0,
    "init_wind_piston_clip_m": 4e-6,
    "init_wind_tip_arcsec_std_tt": 0.05,
    "init_wind_tilt_arcsec_std_tt": 0.05,
    "runtime_wind_piston_micron_factor": 1.0 / 8.0,
    "runtime_wind_tip_arcsec_factor": 1.0 / 32.0,
    "runtime_wind_tilt_arcsec_factor": 1.0 / 32.0,
    "runtime_wind_incremental_factor": 0.01,
    "init_gravity_piston_micron_std": 300.0,
    "init_gravity_tip_arcsec_std_tt": 15.0,
    "init_gravity_tilt_arcsec_std_tt": 15.0,

    # --- Control hierarchy -------------------------------------------
    "control_interval_ms": 2.0,
    "frame_interval_ms": 4.0,
    "decision_interval_ms": 8.0,
    "ao_interval_ms": 1.0,
    "max_episode_steps": 100,

    # --- Agent action toggles ----------------------------------------
    "command_secondaries": False,
    "command_tensioners": False,
    "command_dm": False,
    "command_tip_tilt": False,

    # --- Action parameterization -------------------------------------
    "discrete_control": False,
    "discrete_control_steps": 128,
    "incremental_control": False,
    "action_type": "none",
    "actuator_noise": True,
    "actuator_noise_fraction": 1e-4,

    # --- Observation -------------------------------------------------
    "observation_mode": "image_only",
    "observation_window_size": 2,

    # --- Adaptive optics ---------------------------------------------
    "ao_loop_active": False,
    "randomize_dm": False,
    "dm_gain": 0.6,
    "dm_leakage": 0.01,
    "dm_model_type": "gaussian_influence",
    "dm_num_actuators_across": 35,
    "dm_num_modes": 500,
    "shwfs_f_number": 50,
    "shwfs_num_lenslets": 40,
    "shwfs_diameter_m": 5e-3,
    "dm_interaction_rcond": 1e-3,
    "dm_probe_amp_fraction": 0.01,
    "dm_flux_limit_fraction": 0.5,
    "dm_cache_dir": "./tmp/cache/",

    # --- DM / actuator physics ---------------------------------------
    "microns_opd_per_actuator_bit": 0.00015,
    "stroke_count_limit": 20000,

    # --- Secondary mirror corrections --------------------------------
    "max_piston_correction_micron": 10.0,
    "max_tip_correction_arcsec": 20.0,
    "max_tilt_correction_arcsec": 20.0,
    "get_disp_corr_max_piston_micron": 3.0,
    "get_disp_corr_max_tip_arcsec": 20.0,
    "get_disp_corr_max_tilt_arcsec": 20.0,

    # --- Reward ------------------------------------------------------
    "reward_function": "composite",
    "reward_weight_strehl": 1.0,
    "reward_weight_dark_hole": 0.0,
    "reward_weight_image_quality": 0.0,
    "reward_threshold": 25.0,
    "align_radius": 32,
    "align_radius_max_expand": 64,
    "align_mse_expand_threshold": -1.25,
    "ao_closed_inv_slope_threshold": 2e6,
    "dark_hole_alpha": 0.0,
    "action_penalty": True,
    "action_penalty_weight": 0.03,
    "oob_penalty": True,
    "oob_penalty_weight": 0.5,

    # --- Dark hole ---------------------------------------------------
    "dark_hole": False,
    "dark_hole_angular_location_degrees": 0.0,
    "dark_hole_location_radius_fraction": 0.0,
    "dark_hole_size_radius": 0.0,

    # --- Detector model ----------------------------------------------
    "detector_power_watts": 1e-12,
    "detector_wavelength_meters": 500e-9,
    "detector_quantum_efficiency": 0.8,
    "detector_gain_e_per_dn": 0.5,
    "detector_max_dn": 65535,

    # --- State recording ---------------------------------------------
    "record_env_state_info": False,
    "write_env_state_info": False,
    "state_info_save_dir": "./tmp/",

    # --- Episode -----------------------------------------------------
    "num_episodes": 1,
    "num_steps": 16,

    # --- Optomech version tag ----------------------------------------
    "optomech_version": "v4",

    # --- Device (v4-specific) ----------------------------------------
    "device": None,    # None = auto-detect, or "cpu", "mps", "cuda"
}


# ===================================================================
# V4 Logging  (green tag to distinguish from v3 magenta, v2 cyan)
# ===================================================================

_V4_TAG = "\033[92m[optomech-v4]\033[0m"
_V4_TAG_PLAIN = "[optomech-v4]"

_TOP    = "\u2554" + "\u2550" * 58 + "\u2557"
_BOT    = "\u255A" + "\u2550" * 58 + "\u255D"
_SIDE   = "\u2551"
_MID    = "\u2560" + "\u2550" * 58 + "\u2563"
_THIN   = "\u2502"
_HTHIN  = "\u2500" * 58


def _v4(msg, indent=0):
    """Print with the v4 prefix tag and optional indent."""
    pad = "  " * indent
    print(f"{_V4_TAG} {pad}{msg}")


def _v4_section(title):
    """Print a box-drawn section header."""
    inner = f" {title} ".center(58)
    print(f"{_V4_TAG} {_TOP}")
    print(f"{_V4_TAG} {_SIDE}{inner}{_SIDE}")
    print(f"{_V4_TAG} {_BOT}")


def _v4_subsection(title):
    """Print a lighter sub-section separator."""
    inner = f" {title} ".center(58, "\u2500")
    print(f"{_V4_TAG} {inner}")


def _v4_kv(key, value, indent=1):
    """Print a key-value pair."""
    pad = "  " * indent
    print(f"{_V4_TAG} {pad}{key:<42s} {value}")


def _v4_timer(label, elapsed, indent=1):
    """Print a timing measurement."""
    pad = "  " * indent
    bar_len = min(int(elapsed * 200), 30)
    bar = "\u2588" * bar_len
    print(f"{_V4_TAG} {pad}\u23f1  {label:<34s} {elapsed:8.4f}s {bar}")


def _print_config_banner(cfg, title="Optomech V4 Configuration"):
    """Pretty-print all configuration key-value pairs inside a box."""
    _v4_section(title)
    max_key_len = max(len(k) for k in cfg)
    prev_section = None
    for k in sorted(cfg.keys()):
        section = k.split("_")[0]
        if section != prev_section and prev_section is not None:
            print(f"{_V4_TAG}   {'':>{max_key_len}}   {'':>10}")
        prev_section = section
        print(f"{_V4_TAG}   {k:<{max_key_len}}  =  {cfg[k]}")
    print(f"{_V4_TAG} {_BOT}")


# ===================================================================
# Device selection
# ===================================================================

def _select_device(requested=None):
    """Select the best available PyTorch device.

    Priority: explicit request > CUDA > MPS > CPU.
    Matches the RL trainer convention in ddpg_tbptt_lstm.py.
    """
    if requested is not None and requested != "auto":
        dev = torch.device(requested)
        _v4("Device (requested): %s" % dev)
        return dev
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        _v4("Device (auto): CUDA")
        return dev
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        dev = torch.device("mps")
        _v4("Device (auto): MPS (Metal)")
        return dev
    dev = torch.device("cpu")
    _v4("Device (auto): CPU")
    return dev


# ===================================================================
# Utility functions  (identical to v3)
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
# ObjectPlane  (identical to v3)
# ===================================================================

class ObjectPlane(object):
    """Generates the 2-D intensity distribution for the object scene."""

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

    def _make_binary_object(self, **kwargs):
        std = 1.0
        kernel_extent = 8 * std
        ifov = 0.0165012
        separation_pixels = int(0.6 / ifov)
        mu_x = self.extent_pixels // 2
        mu_y_primary = (self.extent_pixels // 2) - (separation_pixels // 2)
        primary = offset_gaussian(self.extent_pixels, mu_x, mu_y_primary,
                                  std, kernel_extent, normalised=True)
        mu_y_secondary = (self.extent_pixels // 2) + (separation_pixels // 2)
        secondary = offset_gaussian(self.extent_pixels, mu_x, mu_y_secondary,
                                    std, kernel_extent, normalised=True)
        return primary + secondary

    def _make_single_object(self, **kwargs):
        x = self.extent_pixels // 2
        y = self.extent_pixels // 2
        array_value = 1.02e-8
        return one_hot_array(self.extent_pixels, x, y, value=array_value)

    def _load_usaf1951(self, size):
        valid_sizes = {512, 256, 128}
        if size not in valid_sizes:
            raise NotImplementedError(
                "ObjectPlane extent was %s, but only %s are implemented "
                "for usaf1951." % (size, valid_sizes))
        raw = self._rgb2gray(image.imread('usaf1951_%d.jpg' % size))
        raw = raw / np.max(raw)
        raw = -(raw - np.max(raw))
        return raw

    @staticmethod
    def _rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


# ===================================================================
# Wavelength sampling helper (used by both OpticalSystem and OptomechEnv)
# ===================================================================


def _centered_wavelengths(center_wl, bandwidth_nm, n_samples):
    """Compute centered wavelength samples across a bandwidth.

    For N samples across bandwidth B centered at λ_c, samples are placed
    at the centers of N equal bins:
      N=1  →  [λ_c]
      N=2  →  [λ_c − B/4, λ_c + B/4]
      N=3  →  [λ_c − B/3, λ_c, λ_c + B/3]
    """
    bw_m = bandwidth_nm / 1e9
    if n_samples <= 1:
        return [center_wl]
    bin_width = bw_m / n_samples
    first = center_wl - bw_m / 2.0 + bin_width / 2.0
    return [first + i * bin_width for i in range(n_samples)]


# ===================================================================
# OpticalSystem  (identical to v3 -- HCIPy for construction only)
# ===================================================================

class OpticalSystem(object):
    """End-to-end optical simulation with pre-computed caches for speed.

    Used by OptomechEnv for construction and calibration. The hot-path
    simulation in v4 bypasses this class entirely via PyTorch tensors.
    """

    def __init__(self, **kwargs):
        cfg = {**DEFAULT_CONFIG, **kwargs}

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
        self.microns_opd_per_actuator_bit = cfg["microns_opd_per_actuator_bit"]
        self.stroke_count_limit = cfg["stroke_count_limit"]
        self._dm_stroke_limit_m = (
            self.stroke_count_limit * self.microns_opd_per_actuator_bit * 1e-6 / 2.0)
        self._cfg = cfg

        # Wind / temperature / gravity state
        self.ground_wind_speed_mps = cfg["initial_ground_wind_speed_mps"]
        self.ground_wind_speed_ms_sampled_std_mps = (
            cfg["ground_wind_speed_std_fraction"] * self.ground_wind_speed_mps)
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

        aperture_type = cfg['aperture_type']
        _v4_subsection("Aperture: %s" % aperture_type)
        (aperture_func, segments_func,
         focal_length, pupil_diameter,
         focal_plane_image_size_meters) = self._build_aperture(aperture_type, cfg)

        num_px = cfg['focal_plane_image_size_pixels']
        self.wavelength = cfg['wavelength']
        oversampling_factor = cfg['oversampling_factor']

        seeing = cfg['seeing_arcsec']
        outer_scale = cfg['outer_scale_meters']
        tau0 = cfg['tau0_seconds']
        fried_parameter = hcipy.seeing_to_fried_parameter(seeing)
        _v4_kv("fried_parameter", fried_parameter)
        Cn_squared = hcipy.Cn_squared_from_fried_parameter(
            fried_parameter, wavelength=self.wavelength)
        velocity = 0.314 * fried_parameter / tau0

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

        _v4_subsection("Focal Plane Geometry")
        _v4_kv("grid pixels",            "%d" % num_px)
        _v4_kv("extent (m)",             "%.6e" % focal_plane_extent_metres)
        _v4_kv("pixel extent (m)",       "%.6e" % focal_plane_pixel_extent_meters)
        _v4_kv("resolution element (m)", "%.6e" % focal_plane_resolution_element)
        _v4_kv("pixels per meter",       "%.2f" % focal_plane_pixels_per_meter)
        _v4_kv("num airy (res-el radii)","%.4f" % num_airy)
        _v4_kv("sampling (px/res-el)",   "%.4f" % sampling)
        _v4_kv("iFOV (arcsec/px)",       "%.7f" % self.ifov)
        _v4_kv("FOV (arcsec)",           "%.4f" % fov)
        _v4_kv("incremental_control",    str(self.incremental_control))

        _v4("Building object plane...")
        self.object_plane = ObjectPlane(
            object_type=cfg['object_type'],
            object_plane_extent_pixels=num_px,
            object_plane_extent_meters=cfg['object_plane_extent_meters'],
            object_plane_distance_meters=cfg['object_plane_distance_meters'],
        )
        self._cached_object_spectrum = sp_fft.fft2(self.object_plane.array)

        _v4("Building pupil grid...")
        self.pupil_grid = hcipy.make_pupil_grid(dims=num_px, diameter=pupil_diameter)

        self.atmosphere_layers = []
        _v4("Building %d atmosphere layer(s)..." % cfg['num_atmosphere_layers'])
        for _ in range(cfg['num_atmosphere_layers']):
            layer = hcipy.InfiniteAtmosphericLayer(
                self.pupil_grid, Cn_squared, outer_scale, velocity)
            self.atmosphere_layers.append(layer)

        focal_grid = hcipy.make_pupil_grid(
            dims=num_px, diameter=focal_plane_extent_metres)
        focal_grid = focal_grid.shifted(focal_grid.delta / 2)
        self._focal_grid = focal_grid  # Store for v4 extraction

        _v4("Building Fraunhofer propagator...")
        self.pupil_to_focal_propagator = hcipy.FraunhoferPropagator(
            self.pupil_grid, focal_grid, focal_length)
        self._focal_length = focal_length  # Store for v4 extraction

        aperture_field = hcipy.evaluate_supersampled(
            aperture_func, self.pupil_grid, oversampling_factor)
        segments_field = hcipy.evaluate_supersampled(
            segments_func, self.pupil_grid, oversampling_factor)
        self.segmented_mirror = hcipy.SegmentedDeformableMirror(segments_field)
        self.aperture = aperture_field

        # Polychromatic perfect PSF: average across all sampled wavelengths
        _perfect_wavelengths = _centered_wavelengths(
            self.wavelength,
            cfg['bandwidth_nanometers'],
            cfg['bandwidth_sampling'])
        _v4("Computing polychromatic perfect PSF (%d wavelength(s): %s nm)"
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

        if cfg.get('dark_hole', False):
            self._apply_dark_hole(cfg, num_px)

        if self.model_ao:
            self._build_ao_subsystem(cfg, pupil_diameter, focal_grid)

        if self.init_differential_motion:
            _v4("Applying initial differential motion...")
            self._init_natural_diff_motion()

        self._store_baseline_segment_displacements()

        _v4("Building science camera...")
        self.camera = hcipy.NoiselessDetector(focal_grid)

        self._max_p_m = cfg["max_piston_correction_micron"] * 1e-6
        self._max_t_r = cfg["max_tip_correction_arcsec"] * np.pi / (180 * 3600)
        self._max_tl_r = cfg["max_tilt_correction_arcsec"] * np.pi / (180 * 3600)

        _v4_subsection("Optical System Ready")

    # --- Aperture construction (identical to v3) ---------------------

    def _build_aperture(self, aperture_type, cfg):
        if aperture_type == "elf":
            self.num_apertures = 15
            focal_plane_image_size_meters = 8.192e-4
            focal_length = 32.5
            pupil_diameter = 3.6
            segment_diameter = 0.5
            elf_segment_centroid_diameter = 2.7
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
        f_number = cfg['shwfs_f_number']
        num_lenslets = cfg['shwfs_num_lenslets']
        sh_diameter = cfg['shwfs_diameter_m']
        magnification = sh_diameter / pupil_diameter
        self.magnifier = hcipy.Magnifier(magnification)
        dm_model_type = cfg['dm_model_type']
        _v4("Building SHWFS (f/%.0f, %d lenslets)..." % (f_number, num_lenslets))
        self.shwfs = hcipy.SquareShackHartmannWavefrontSensorOptics(
            self.pupil_grid.scaled(magnification),
            f_number, num_lenslets, sh_diameter)
        self.shwfse = hcipy.ShackHartmannWavefrontSensorEstimator(
            self.shwfs.mla_grid,
            self.shwfs.micro_lens_array.mla_index)
        self.shwfs_camera = hcipy.NoiselessDetector(focal_grid)
        _v4("Building DM (model=%s)..." % dm_model_type)
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

    # --- Field / atmosphere / simulation helpers (for AO + state recording) ---

    def make_object_field(self, array, center=None):
        def func(grid):
            f = array.ravel()
            return hcipy.Field(f.astype('float'), grid)
        return func

    def evolve_atmosphere_to(self, episode_time_ms):
        episode_time_seconds = episode_time_ms / 1000.0
        for layer in self.atmosphere_layers:
            layer.evolve_until(episode_time_seconds)

    def command_dm(self, dm_command):
        dm_stroke_meters = self._dm_stroke_limit_m * 2.0
        command_vector = np.array([x[0] for x in dm_command])
        dm_command_meters = (dm_stroke_meters / 2.0) * command_vector
        if self._actuator_noise:
            dm_command_meters += np.random.normal(
                0.0, self._actuator_noise_fraction * dm_stroke_meters,
                size=dm_command_meters.shape)
        self.dm.actuators = dm_command_meters

    def simulate(self, wavelength):
        """Full HCIPy simulation -- used for AO/SHWFS path only."""
        self.object_wavefront = hcipy.Wavefront(self.aperture, wavelength)
        self.pre_atmosphere_object_wavefront = self.object_wavefront
        wf = self.pre_atmosphere_object_wavefront
        for atm_layer in self.atmosphere_layers:
            wf = atm_layer.forward(wf)
        self.post_atmosphere_wavefront = wf
        if self.simulate_differential_motion:
            self._simulate_natural_diff_motion()
        self.pupil_wavefront = self.segmented_mirror(
            self.post_atmosphere_wavefront)
        if self.model_ao:
            self.post_dm_wavefront = self.dm.forward(self.pupil_wavefront)
        else:
            self.post_dm_wavefront = self.pupil_wavefront
        self.focal_plane_wavefront = self.pupil_to_focal_propagator(
            self.post_dm_wavefront)

    def get_shwfs_frame(self, integration_seconds=1.0):
        self.shwfs_camera.integrate(
            self.shwfs(self.magnifier(self.post_dm_wavefront)),
            integration_seconds)
        return self.shwfs_camera.read_out()

    def get_science_frame(self, integration_seconds=1.0):
        """Full HCIPy science frame -- used for state recording only."""
        self.camera.integrate(self.focal_plane_wavefront, integration_seconds)
        effective_psf = self.camera.read_out()
        side = int(np.sqrt(effective_psf.size))
        effective_psf = effective_psf.reshape((side, side))
        self.instantaneous_psf = effective_psf
        effective_otf = sp_fft.fft2(effective_psf, workers=-1)
        object_spectrum = self._cached_object_spectrum
        image_spectrum = object_spectrum * effective_otf
        self.readout_image = np.abs(
            sp_fft.fftshift(sp_fft.ifft2(image_spectrum, workers=-1)))
        return self.readout_image

    def calibrate_dm_interaction_matrix(self, env_uuid):
        cfg = self._cfg
        probe_amp = cfg['dm_probe_amp_fraction'] * self.wavelength
        wf = hcipy.Wavefront(self.aperture, self.wavelength)
        wf.total_power = 1
        self.shwfs_camera.integrate(self.shwfs(self.magnifier(wf)), 1)
        reference_image = self.shwfs_camera.read_out()
        fluxes = ndimage.measurements.sum(
            reference_image, self.shwfse.mla_index,
            self.shwfse.estimation_subapertures)
        flux_limit = fluxes.max() * cfg['dm_flux_limit_fraction']
        estimation_subapertures = self.shwfs.mla_grid.zeros(dtype='bool')
        estimation_subapertures[
            self.shwfse.estimation_subapertures[fluxes > flux_limit]] = True
        self.shwfse = hcipy.ShackHartmannWavefrontSensorEstimator(
            self.shwfs.mla_grid,
            self.shwfs.micro_lens_array.mla_index,
            estimation_subapertures)
        self.reference_slopes = self.shwfse.estimate([reference_image])
        dm_cache_path = os.path.join(cfg['dm_cache_dir'], str(env_uuid))
        if os.path.exists(dm_cache_path):
            _v4("Found cached interaction matrix at %s" % dm_cache_path)
            with open(os.path.join(dm_cache_path,
                      'dm_interaction_matrix.pkl'), 'rb') as f:
                self.interaction_matrix = pickle.load(f)
                return
        n_act = len(self.dm.actuators)
        _v4_subsection("DM Calibration (%d actuators)" % n_act)
        response_matrix = []
        for i in range(n_act):
            if i % 50 == 0 or i == n_act - 1:
                _v4("  actuator %d / %d" % (i + 1, n_act))
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
        Path(dm_cache_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(dm_cache_path,
                  "dm_interaction_matrix.pkl"), 'wb') as f:
            pickle.dump(self.interaction_matrix, f)

    # --- Differential motion (identical to v3) -----------------------

    def _simulate_natural_diff_motion(self):
        cfg = self._cfg
        if self.model_wind_diff_motion:
            self.ground_wind_speed_mps += (
                np.random.randn() * self.ground_wind_speed_ms_sampled_std_mps)
            self.ground_wind_speed_mps = np.clip(
                self.ground_wind_speed_mps, 0.0, cfg["max_ground_wind_speed_mps"])
            ws = self.ground_wind_speed_mps
            piston_std = ws * cfg["runtime_wind_piston_micron_factor"]
            tip_std = ws * cfg["runtime_wind_tip_arcsec_factor"]
            tilt_std = ws * cfg["runtime_wind_tilt_arcsec_factor"]
            wind_ptt = np.random.randn(self.num_apertures, 3)
            wind_ptt[:, 0] *= piston_std * 1e-6
            wind_ptt[:, 1] *= tip_std * np.pi / (180 * 3600)
            wind_ptt[:, 2] *= tilt_std * np.pi / (180 * 3600)
            self._apply_ptt_displacements(
                wind_ptt, incremental=True,
                incremental_factor=cfg["runtime_wind_incremental_factor"])
        if self.model_temp_diff_motion:
            _ = np.random.randn(self.num_apertures, 3)
        if self.model_gravity_diff_motion:
            _ = np.random.randn(self.num_apertures, 3)

    def _init_natural_diff_motion(self):
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
            temp_ptt = np.random.randn(self.num_apertures, 3)
            temp_ptt[:, 0] *= 0.0
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

    def _apply_ptt_displacements(self, ptt_displacements,
                                  incremental=False, incremental_factor=1.0):
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
        return [self.segmented_mirror.get_segment_actuators(i)
                for i in range(self.num_apertures)]

    def get_displacement_correction(self):
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
        self.segment_baseline_dict = {}
        for seg_id in range(self.num_apertures):
            (p, t, tl) = self.segmented_mirror.get_segment_actuators(seg_id)
            self.segment_baseline_dict[seg_id] = {
                "piston": p, "tip": t, "tilt": tl}

    def command_tensioners(self, tensioner_commands):
        self._optomechanical_interaction(tensioner_commands)

    def _optomechanical_interaction(self, tension_forces):
        tension_forces = np.transpose(np.array(tension_forces))
        optomech_embedding = tension_forces.dot(self._optomech_encoder)
        optomech_ptt = optomech_embedding.dot(self._optomech_decoder)
        optomech_ptt = optomech_ptt.reshape((self.num_apertures, 3))
        optomech_ptt = np.zeros((self.num_apertures, 3))
        optomech_ptt[:, 0] *= 1e-6
        optomech_ptt[:, 1] *= np.pi / (180 * 3600)
        optomech_ptt[:, 2] *= np.pi / (180 * 3600)
        self._apply_ptt_displacements(ptt_displacements=optomech_ptt)

    def command_secondaries(self, secondaries_commands):
        max_p_m = self._max_p_m
        max_t_r = self._max_t_r
        max_tl_r = self._max_tl_r
        self.max_piston_correction = max_p_m
        self.max_tip_correction = max_t_r
        self.max_tilt_correction = max_tl_r
        _seg_mirror = self.segmented_mirror
        _discrete = self.discrete_control
        _incremental = self.incremental_control
        _baseline = self.segment_baseline_dict
        _discrete_steps = self.discrete_control_steps
        _noisy = self._actuator_noise
        _noise_frac = self._actuator_noise_fraction
        for seg_id in range(self.num_apertures):
            seg_piston_cmd = secondaries_commands[seg_id][0]
            if len(secondaries_commands[seg_id]) == 3:
                seg_tip_cmd = secondaries_commands[seg_id][1]
                seg_tilt_cmd = secondaries_commands[seg_id][2]
            else:
                seg_tip_cmd = 0.0
                seg_tilt_cmd = 0.0
            if _discrete:
                (seg_p, seg_t, seg_tl) = _seg_mirror.get_segment_actuators(seg_id)
                inc_p = seg_piston_cmd[0]; dec_p = seg_piston_cmd[1]
                inc_t = seg_tip_cmd[0]; dec_t = seg_tip_cmd[1]
                inc_tl = seg_tilt_cmd[0]; dec_tl = seg_tilt_cmd[1]
                piston_state = seg_p + inc_p * (max_p_m / _discrete_steps)
                tip_state = seg_t + inc_t * (max_t_r / _discrete_steps)
                tilt_state = seg_tl + inc_tl * (max_tl_r / _discrete_steps)
                piston_state = seg_p - dec_p * (max_p_m / _discrete_steps)
                tip_state = seg_t - dec_t * (max_t_r / _discrete_steps)
                tilt_state = seg_tl - dec_tl * (max_tl_r / _discrete_steps)
                bl = _baseline[seg_id]
                pre_clip = np.array([piston_state, tip_state, tilt_state])
                piston_state = np.clip(piston_state, -max_p_m + bl["piston"], max_p_m + bl["piston"])
                tip_state = np.clip(tip_state, -max_t_r + bl["tip"], max_t_r + bl["tip"])
                tilt_state = np.clip(tilt_state, -max_tl_r + bl["tilt"], max_tl_r + bl["tilt"])
                post_clip = np.array([piston_state, tip_state, tilt_state])
                self._clipped_dof_count += int(np.sum(pre_clip != post_clip))
                self._total_dof_count += 3
            elif _incremental:
                p_cmd_m = seg_piston_cmd * max_p_m
                t_cmd_r = seg_tip_cmd * max_t_r
                tl_cmd_r = seg_tilt_cmd * max_tl_r
                (seg_p, seg_t, seg_tl) = _seg_mirror.get_segment_actuators(seg_id)
                piston_state = seg_p + p_cmd_m
                tip_state = seg_t + t_cmd_r
                tilt_state = seg_tl + tl_cmd_r
                bl = _baseline[seg_id]
                pre_clip = np.array([piston_state, tip_state, tilt_state])
                piston_state = np.clip(piston_state, -max_p_m + bl["piston"], max_p_m + bl["piston"])
                tip_state = np.clip(tip_state, -max_t_r + bl["tip"], max_t_r + bl["tip"])
                tilt_state = np.clip(tilt_state, -max_tl_r + bl["tilt"], max_tl_r + bl["tilt"])
                post_clip = np.array([piston_state, tip_state, tilt_state])
                self._clipped_dof_count += int(np.sum(pre_clip != post_clip))
                self._total_dof_count += 3
            else:
                p_cmd_m = seg_piston_cmd * max_p_m
                t_cmd_r = seg_tip_cmd * max_t_r
                tl_cmd_r = seg_tilt_cmd * max_tl_r
                bl = _baseline[seg_id]
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
# OptomechEnv -- Gymnasium environment with PyTorch fast path
# ===================================================================

class OptomechEnv(gym.Env):
    """GPU-accelerated v4 environment for distributed-aperture telescope control.

    Construction and calibration use HCIPy. The step() hot path bypasses
    HCIPy entirely, operating on pre-extracted PyTorch tensors on the
    best available device (MPS / CUDA / CPU).
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50,
    }

    def __init__(self, **kwargs):
        self.cfg = {**DEFAULT_CONFIG, **kwargs}
        cfg = self.cfg
        _print_config_banner(cfg)

        # --- Device selection ----------------------------------------
        self._device = _select_device(cfg.get('device', None))

        # --- Seed & identity -----------------------------------------
        self.seed()
        self.kwargs = kwargs
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
        self.reward_function = cfg['reward_function']

        # --- Composite reward weights --------------------------------
        self._reward_weight_strehl = cfg['reward_weight_strehl']
        self._reward_weight_dark_hole = cfg['reward_weight_dark_hole']
        self._reward_weight_image_quality = cfg['reward_weight_image_quality']

        # --- Actuator noise (stochastic repeatability) ---------------
        self._actuator_noise = cfg['actuator_noise']
        self._actuator_noise_fraction = cfg['actuator_noise_fraction']

        # --- Action penalty ------------------------------------------
        self._action_penalty = cfg['action_penalty']
        self._action_penalty_weight = cfg['action_penalty_weight']
        self._oob_penalty = cfg['oob_penalty']
        self._oob_penalty_weight = cfg['oob_penalty_weight']
        self.ao_loop_active = cfg['ao_loop_active']
        self.observation_mode = cfg['observation_mode']
        self._obs_image_only = (self.observation_mode == "image_only")
        self._obs_image_action = (self.observation_mode == "image_action")

        if self.command_dm or self.ao_loop_active:
            cfg['model_ao'] = True
        else:
            cfg['model_ao'] = False

        self.microns_opd_per_actuator_bit = cfg['microns_opd_per_actuator_bit']
        self.stroke_count_limit = cfg['stroke_count_limit']
        self.dm_gain = cfg['dm_gain']
        self.dm_leakage = cfg['dm_leakage']
        self._dm_stroke_limit_m = (
            self.microns_opd_per_actuator_bit
            * self.stroke_count_limit * 1e-6 / 2.0)

        # Detector constants (pre-computed)
        _h = 6.62607015e-34
        _c = 2.99792458e8
        self._photon_energy = _h * _c / cfg['wavelength']
        self._det_qe = cfg['detector_quantum_efficiency']
        self._det_gain = cfg['detector_gain_e_per_dn']
        self._det_max_dn = cfg['detector_max_dn']
        self._frame_sec = self.frame_interval_ms / 1000.0

        _v4_section("Initializing OptomechEnv")

        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self._init_state_storage()
        self._compute_timing_ratios()

        if cfg.get('dark_hole', False):
            _v4_subsection("Dark Hole")
            _v4_kv("angular location (deg)", cfg.get('dark_hole_angular_location_degrees', 'N/A'))
            _v4_kv("location radius frac",   cfg.get('dark_hole_location_radius_fraction', 'N/A'))
            _v4_kv("size radius",            cfg.get('dark_hole_size_radius', 'N/A'))
        else:
            _v4("Dark hole: disabled")

        _v4_subsection("Actuator Noise")
        if self._actuator_noise:
            _v4_kv("enabled", "True")
            _v4_kv("noise fraction", "%.2e" % self._actuator_noise_fraction)
        else:
            _v4("Actuator noise: disabled")

        _v4_subsection("Action Penalty")
        if self._action_penalty:
            _v4_kv("enabled", "True")
            _v4_kv("weight", "%.4f" % self._action_penalty_weight)
        else:
            _v4("Action penalty: disabled")

        _v4_subsection("OOB Penalty")
        if self._oob_penalty:
            _v4_kv("enabled", "True")
            _v4_kv("weight", "%.4f" % self._oob_penalty_weight)
        else:
            _v4("OOB penalty: disabled")

        _v4_subsection("Reward")
        _v4_kv("function", self.reward_function)
        if self.reward_function == "composite":
            _v4_kv("weight_strehl", "%.3f" % self._reward_weight_strehl)
            _v4_kv("weight_dark_hole", "%.3f" % self._reward_weight_dark_hole)
            _v4_kv("weight_image_quality", "%.3f" % self._reward_weight_image_quality)

        self._build_optical_system()
        self.episode_time_ms = 0.0
        self._build_action_spaces()
        self._build_observation_space()

        # Cache wavelengths
        self._cached_wavelengths = self._compute_wavelengths()

        # Cache reward images
        self._cache_reward_images()
        self._cache_center_masks()

        # Reward dispatch
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

        # [v4] Initialize the fast PyTorch pipeline
        self._init_fast_pipeline()

        self.state_content["wavelength"] = self.optical_system.wavelength
        _v4_section("Environment Ready (device=%s)" % self._device)

    # ================================================================
    # [v4] Fast PyTorch Pipeline
    # ================================================================

    def _init_fast_pipeline(self):
        """Extract all HCIPy arrays into PyTorch tensors on self._device."""
        os_obj = self.optical_system
        dev = self._device

        _v4_subsection("Fast Pipeline Init")

        # 1. Aperture → complex64 tensor
        self._t_aperture = torch.from_numpy(
            np.array(os_obj.aperture, dtype=np.complex64)).to(dev)
        N = self._t_aperture.shape[0]
        _v4_kv("aperture pixels", "%d" % N)

        # 2. Segmented mirror transformation matrix → dense float32
        seg_tm = os_obj.segmented_mirror.influence_functions.transformation_matrix
        if scipy.sparse.issparse(seg_tm):
            seg_dense = np.asarray(seg_tm.todense(), dtype=np.float32)
        else:
            seg_dense = np.array(seg_tm, dtype=np.float32)
        self._t_seg_transform = torch.from_numpy(seg_dense).to(dev)
        _v4_kv("seg transform shape", str(self._t_seg_transform.shape))

        # 3. DM transformation matrix (if AO)
        if os_obj.model_ao:
            dm_tm = os_obj.dm.influence_functions.transformation_matrix
            if scipy.sparse.issparse(dm_tm):
                dm_dense = np.asarray(dm_tm.todense(), dtype=np.float32)
            else:
                dm_dense = np.array(dm_tm, dtype=np.float32)
            self._t_dm_transform = torch.from_numpy(dm_dense).to(dev)
            _v4_kv("dm transform shape", str(self._t_dm_transform.shape))

        # 4. Fraunhofer propagator internals (per wavelength)
        #    HCIPy may select either FastFourierTransform or MatrixFourierTransform
        #    depending on grid sizes.  We detect which and extract accordingly.
        self._t_props = {}
        self._ft_mode = None          # 'fft' or 'mft', set on first wl
        for wl in self._cached_wavelengths:
            # Force HCIPy to populate its internal cache
            test_wf = hcipy.Wavefront(os_obj.aperture, wl)
            os_obj.pupil_to_focal_propagator(os_obj.segmented_mirror(test_wf))
            inst = os_obj.pupil_to_focal_propagator.get_instance_data(
                os_obj.pupil_grid, None, wl)
            ft = inst.fourier_transform

            if isinstance(ft, FastFourierTransform):
                if self._ft_mode is None:
                    self._ft_mode = 'fft'
                shift_in = np.array(ft.shift_input, dtype=np.complex64)
                shift_out = (np.array(ft.shift_output, dtype=np.complex64)
                             if ft.shift_output is not None else None)
                self._t_props[wl] = {
                    'norm_factor': torch.tensor(
                        complex(inst.norm_factor), dtype=torch.complex64,
                        device=dev),
                    'shift_input': torch.from_numpy(shift_in).to(dev),
                    'shift_output': (torch.from_numpy(shift_out).to(dev)
                                     if shift_out is not None else None),
                }

            elif isinstance(ft, MatrixFourierTransform):
                if self._ft_mode is None:
                    self._ft_mode = 'mft'
                # Force matrix computation so we can extract M1 / M2
                ft._compute_matrices(np.complex64)
                # Extract weights_input (scalar or array)
                if np.isscalar(ft.weights_input):
                    w_tensor = torch.tensor(
                        float(ft.weights_input),
                        dtype=torch.float32, device=dev)
                else:
                    w_tensor = torch.from_numpy(
                        np.array(ft.weights_input, dtype=np.float32)).to(dev)
                self._t_props[wl] = {
                    'norm_factor': torch.tensor(
                        complex(inst.norm_factor), dtype=torch.complex64,
                        device=dev),
                    'M1': torch.from_numpy(
                        np.array(ft.M1, dtype=np.complex64)).to(dev),
                    'M2': torch.from_numpy(
                        np.array(ft.M2, dtype=np.complex64)).to(dev),
                    'weights_input': w_tensor,
                }
            else:
                raise TypeError(
                    f"Unsupported Fourier transform type: {type(ft).__name__}")

        # Grid-level params (same for all wavelengths, type-dependent)
        if self._ft_mode == 'fft':
            self._fp_shape_in = tuple(ft.shape_in)
            self._fp_internal_shape = tuple(ft.internal_shape)
            self._fp_cutout_input = ft.cutout_input
            self._fp_cutout_output = ft.cutout_output
            self._fp_emulate_fftshifts = ft.emulate_fftshifts
            _v4_kv("transform mode", "FFT (FastFourierTransform)")
            _v4_kv("FFT shape_in", str(self._fp_shape_in))
            _v4_kv("FFT internal_shape", str(self._fp_internal_shape))
        elif self._ft_mode == 'mft':
            self._mft_shape_in = tuple(ft.shape_input)
            self._mft_shape_out = tuple(ft.shape_output)
            _v4_kv("transform mode", "MFT (MatrixFourierTransform)")
            _v4_kv("MFT shape_in", str(self._mft_shape_in))
            _v4_kv("MFT shape_out", str(self._mft_shape_out))

        # 5. Focal grid weights
        self._t_focal_weights = torch.from_numpy(
            np.array(os_obj._focal_grid.weights, dtype=np.float32)).to(dev)

        # 6. Object spectrum (constant per episode)
        obj_spec = np.fft.fft2(os_obj.object_plane.array).astype(np.complex64)
        self._t_object_spectrum = torch.from_numpy(obj_spec).to(dev)

        # 7. Pre-allocated buffers on device
        self._t_E_pupil = torch.zeros(N, dtype=torch.complex64, device=dev)
        self._t_seg_surface = torch.zeros(N, dtype=torch.float32, device=dev)
        self._t_dm_surface = torch.zeros(N, dtype=torch.float32, device=dev)
        self._t_zero_phase = torch.zeros(N, dtype=torch.float32, device=dev)

        if self._ft_mode == 'fft':
            N_focal = N   # FFT preserves pixel count
            self._t_E_focal = torch.zeros(
                N_focal, dtype=torch.complex64, device=dev)
            self._t_work = torch.zeros(
                self._fp_internal_shape, dtype=torch.complex64, device=dev)
        elif self._ft_mode == 'mft':
            N_focal = int(np.prod(self._mft_shape_out))
            self._t_E_focal = torch.zeros(
                N_focal, dtype=torch.complex64, device=dev)
            # Intermediate buffer for separable 2D MFT: shape (shape_in[0], M2_cols)
            any_wl = self._cached_wavelengths[0]
            M2_cols = self._t_props[any_wl]['M2'].shape[1]
            self._t_mft_intermediate = torch.zeros(
                (self._mft_shape_in[0], M2_cols),
                dtype=torch.complex64, device=dev)

        # Flag for atmosphere
        self._has_atm = len(os_obj.atmosphere_layers) > 0

        # 8. Pre-cached wavenumbers per wavelength (avoid repeated 2*pi/wl)
        self._t_wavenumbers = {}
        for wl in self._cached_wavelengths:
            self._t_wavenumbers[wl] = 2.0 * math.pi / wl

        # 9. Reusable atmosphere phase buffer on CPU (avoid alloc per iter)
        self._atm_buf_np = np.zeros(N, dtype=np.float32)
        # Reusable atmosphere achromatic screen tensor on device
        self._t_atm_buf = torch.zeros(N, dtype=torch.float32, device=dev)
        # Reusable phase accumulator on device (avoid alloc in _fuse_optics)
        self._t_phase_buf = torch.zeros(N, dtype=torch.float32, device=dev)
        # Reusable segment actuator tensor on device
        n_seg_acts = self._t_seg_transform.shape[1]
        self._t_seg_acts_buf = torch.zeros(
            n_seg_acts, dtype=torch.float32, device=dev)
        # Reusable DM actuator tensor on device
        if os_obj.model_ao:
            n_dm_acts = self._t_dm_transform.shape[1]
            self._t_dm_acts_buf = torch.zeros(
                n_dm_acts, dtype=torch.float32, device=dev)

        _v4_kv("focal plane pixels", str(self._t_E_focal.shape[0]))
        _v4_kv("fast pipeline", "ready (device=%s)" % dev)

    # ================================================================
    # [v4] PyTorch Optical Kernels
    # ================================================================

    def _fuse_optics(self, atm_phase_t, wavenumber, use_dm):
        """Fused optics: E = aperture * exp(j*(atm + 2k*seg + 2k*dm)).

        Single GPU kernel replaces 4 separate HCIPy operations.
        All operations are in-place on pre-allocated buffers.
        """
        two_k = 2.0 * wavenumber
        # phase = atm + 2k*seg  (in-place into pre-allocated buffer)
        torch.mul(self._t_seg_surface, two_k, out=self._t_phase_buf)
        self._t_phase_buf.add_(atm_phase_t)
        if use_dm:
            # phase += 2k * dm_surface
            self._t_phase_buf.add_(self._t_dm_surface, alpha=two_k)
        self._t_E_pupil[:] = self._t_aperture * torch.exp(
            1j * self._t_phase_buf.to(torch.complex64))

    def _fft_propagate(self, prop_data):
        """Replicate HCIPy FastFourierTransform.forward() on device tensors.

        Exact sequence: insert → shift_output → ifftshift → fftn → fftshift
        → crop → shift_input → norm_factor.
        """
        work = self._t_work
        shape_in = self._fp_shape_in

        # Insert into work buffer (zero-pad if needed)
        work.zero_()
        E_reshaped = self._t_E_pupil.reshape(shape_in)
        if self._fp_cutout_input is not None:
            work[self._fp_cutout_input] = E_reshaped
            if prop_data['shift_output'] is not None:
                work[self._fp_cutout_input] *= prop_data['shift_output'].reshape(shape_in)
        else:
            work.copy_(E_reshaped)
            if prop_data['shift_output'] is not None:
                work *= prop_data['shift_output'].reshape(shape_in)

        # FFT on device (MPS / CUDA / CPU)
        if not self._fp_emulate_fftshifts:
            work = torch.fft.ifftshift(work)
        result = torch.fft.fftn(work)
        if not self._fp_emulate_fftshifts:
            result = torch.fft.fftshift(result)

        # Extract and scale
        if self._fp_cutout_output is not None:
            self._t_E_focal[:] = (result[self._fp_cutout_output].reshape(-1)
                                  * prop_data['shift_input']
                                  * prop_data['norm_factor'])
        else:
            self._t_E_focal[:] = (result.reshape(-1)
                                  * prop_data['shift_input']
                                  * prop_data['norm_factor'])

    def _mft_propagate(self, prop_data):
        """Replicate HCIPy MatrixFourierTransform.forward() on device tensors.

        HCIPy 2D MFT (Soummer 2007) — separable DFT via two matmuls:
            M1: (Nv, Ny) = exp(-j * outer(v, y))
            M2: (Nx, Nu) = exp(-j * outer(x, u))

        HCIPy BLAS sequence (C-order equivalence):
            inter = f_2d @ M2                       (Nx, Nu)
            C     = alpha * inter.T @ M1.T          (Nu, Nv)
            result = C.T.reshape(-1)                (Nv*Nu,)

        alpha = scalar weights_input (or 1 if non-scalar, pre-applied).
        Final result is multiplied by Fraunhofer norm_factor.
        """
        shape_in = self._mft_shape_in
        w = prop_data['weights_input']
        M1 = prop_data['M1']
        M2 = prop_data['M2']
        nf = prop_data['norm_factor']

        # Apply weights and reshape to 2D grid
        if w.dim() == 0:
            # Scalar weight: defer to alpha multiplier
            f_2d = self._t_E_pupil.reshape(shape_in)
            alpha = w.to(torch.complex64)
        else:
            f_2d = (self._t_E_pupil * w.to(torch.complex64)).reshape(shape_in)
            alpha = torch.tensor(1.0, dtype=torch.complex64, device=f_2d.device)

        # Step 1: inter = f_2d @ M2,  shape (Nx, Nu)
        torch.mm(f_2d, M2, out=self._t_mft_intermediate)

        # Step 2: C = alpha * inter.T @ M1.T,  shape (Nu, Nv)
        C = torch.mm(self._t_mft_intermediate.T, M1.T) * alpha

        # Step 3: result = C.T.reshape(-1) * norm_factor
        self._t_E_focal[:] = C.T.reshape(-1) * nf

    def _propagate(self, prop_data):
        """Dispatch to FFT or MFT propagation based on detected transform type."""
        if self._ft_mode == 'fft':
            self._fft_propagate(prop_data)
        else:
            self._mft_propagate(prop_data)

    def _compute_science_image_torch(self, integration_sec):
        """PSF → convolve with object → science image.  All on device.

        Returns a 2D torch tensor on self._device.
        """
        # PSF = |E_focal|^2 * focal_weights * dt
        psf = (torch.abs(self._t_E_focal) ** 2
               * self._t_focal_weights * integration_sec)
        psf_2d = psf.reshape(self.image_shape)

        # Convolve with object: IFFT(FFT(PSF) * object_spectrum)
        otf = torch.fft.fft2(psf_2d)
        image = torch.abs(torch.fft.fftshift(
            torch.fft.ifft2(self._t_object_spectrum * otf)))
        return image

    # ================================================================
    # Cache helpers (from v3)
    # ================================================================

    def _cache_reward_images(self):
        pi = self.optical_system.perfect_image
        pi_max = np.max(pi)
        self._norm_perfect_image = pi / pi_max if pi_max != 0 else pi
        ti = self.optical_system.target_image
        ti_max = np.max(ti)
        self._norm_target_image = ti / ti_max if ti_max != 0 else ti
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
        _v4_kv("perfect_image_max_dn", "%.4f" % self._perfect_image_max_dn)

    def _cache_center_masks(self):
        cfg = self.cfg
        num_px = cfg['focal_plane_image_size_pixels']
        ctr = num_px // 2
        radius = cfg['align_radius']
        y, x = np.ogrid[:num_px, :num_px]
        dist_sq = (y - ctr) ** 2 + (x - ctr) ** 2
        self._center_mask_standard = dist_sq <= radius ** 2
        radius_max = cfg['align_radius_max_expand']
        self._center_mask_expanded = dist_sq <= radius_max ** 2

    def _compute_wavelengths(self):
        """Compute centered wavelength samples across the configured bandwidth."""
        return _centered_wavelengths(
            self.optical_system.wavelength,
            self.cfg['bandwidth_nanometers'],
            self.cfg['bandwidth_sampling'])

    # ================================================================
    # Timing
    # ================================================================

    def _compute_timing_ratios(self):
        _v4_subsection("Timing Hierarchy")
        _v4_kv("control  interval", "%.2f ms" % self.control_interval_ms)
        _v4_kv("frame    interval", "%.2f ms" % self.frame_interval_ms)
        _v4_kv("decision interval", "%.2f ms" % self.decision_interval_ms)
        _v4_kv("AO       interval", "%.2f ms" % self.ao_interval_ms)
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
        self.metadata['commands_per_decision'] = self.commands_per_decision
        self.metadata['commands_per_frame'] = self.commands_per_frame
        self.metadata['frames_per_decision'] = self.frames_per_decision
        self.metadata['ao_steps_per_command'] = self.ao_steps_per_command
        self.metadata['ao_steps_per_frame'] = self.ao_steps_per_frame
        _v4_subsection("Derived Rates")
        _v4_kv("commands / decision", "%d" % self.commands_per_decision)
        _v4_kv("commands / frame",    "%d" % self.commands_per_frame)
        _v4_kv("AO steps / frame",    "%d" % self.ao_steps_per_frame)
        _v4_kv("frames / decision",   "%d" % self.frames_per_decision)

    # ================================================================
    # State storage
    # ================================================================

    def _init_state_storage(self):
        self.state_content = {
            "dm_surfaces": [], "atmos_layer_0_list": [], "action_times": [],
            "object_fields": [], "pre_atmosphere_object_wavefronts": [],
            "post_atmosphere_wavefronts": [], "segmented_mirror_surfaces": [],
            "pupil_wavefronts": [], "post_dm_wavefronts": [],
            "focal_plane_wavefronts": [], "readout_images": [],
            "instantaneous_psf": [],
        }

    # ================================================================
    # Optical system builder
    # ================================================================

    def _build_optical_system(self):
        self.optical_system = OpticalSystem(**self.cfg)

    def build_optical_system(self, **kwargs):
        merged = {**self.cfg, **kwargs}
        self.optical_system = OpticalSystem(**merged)

    # ================================================================
    # Action spaces  (anytree-free, from v3)
    # ================================================================

    def _build_action_spaces(self):
        command_space_list = []
        if self.command_secondaries:
            command_space_list.append(self._build_secondaries_space())
        if self.command_tensioners:
            command_space_list.append(self._build_tensioners_space())
        if self.command_dm:
            command_space_list.append(self._build_dm_space())
        single_command_space = spaces.Tuple(tuple(command_space_list))
        self.dict_action_space = spaces.Tuple(
            [single_command_space] * self.commands_per_decision)
        self._linear_to_tree_indices = self._build_index_map(
            self.dict_action_space)
        if self.discrete_control:
            self.action_space = spaces.MultiDiscrete(
                [1] * len(self._linear_to_tree_indices))
        else:
            self.action_space = self._flatten(
                self.dict_action_space, flat_space_low=-1.0, flat_space_high=1.0)
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
        lo = 0.0 if zero else -1.0
        hi = 0.0 if zero else 1.0
        t_space = spaces.Box(low=lo, high=hi, shape=(1,), dtype=np.float32)
        return spaces.Tuple(
            tuple([t_space] * self.optical_system.num_tensioners))

    def _build_dm_space(self, zero=False):
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

    def _build_observation_space(self):
        cfg = self.cfg
        self.image_shape = (cfg['focal_plane_image_size_pixels'],
                            cfg['focal_plane_image_size_pixels'])
        image_stack_shape = (self.frames_per_decision,
                             self.image_shape[0], self.image_shape[1])
        self.image_space = spaces.Box(
            low=0.0, high=1.0, shape=image_stack_shape, dtype=np.float32)
        if self._obs_image_only:
            self.observation_space = self.image_space
        elif self._obs_image_action:
            self.observation_space = spaces.Dict(
                {"image": self.image_space,
                 "prior_action": self.action_space}, seed=42)
        else:
            raise ValueError(
                "Invalid observation_mode: '%s'." % self.observation_mode)

    # --- Action helpers (anytree-free) ---

    def _flatten(self, dict_space, flat_space_high=1.0, flat_space_low=0.0):
        return spaces.Box(
            low=flat_space_low, high=flat_space_high,
            shape=(len(self._build_index_map(dict_space)),),
            dtype=np.float32)

    def _flat_to_dict(self, flat_action, dict_space):
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

    @staticmethod
    def _build_index_map(action_space):
        index_map = []
        for step_num, step in enumerate(action_space):
            for stage_num, stage in enumerate(step):
                for comp_num, comp in enumerate(stage):
                    if hasattr(comp, '__iter__'):
                        for cmd_num, cmd in enumerate(comp):
                            if hasattr(cmd, '__iter__'):
                                for elem_num, elem in enumerate(cmd):
                                    index_map.append(
                                        (step_num, stage_num, comp_num, cmd_num, elem_num))
                            else:
                                index_map.append(
                                    (step_num, stage_num, comp_num, cmd_num))
                    else:
                        index_map.append(
                            (step_num, stage_num, comp_num, 0))
        return index_map

    def _encode_action_from_vector(self, action_space, action_vector):
        index_map = self._linear_to_tree_indices
        action_list = self._tuple_to_list(action_space.sample())
        for n, val in enumerate(action_vector):
            indices = index_map[n]
            sub = action_list
            for idx in indices[:-1]:
                sub = sub[idx]
            sub[indices[-1]] = val
        return self._list_to_tuple(action_list)

    def _get_vector_action_size(self, hierarchical_space):
        return len(self._linear_to_tree_indices)

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
        _v4_section("Episode Reset")
        _v4("Rebuilding optical system...")
        self._build_optical_system()

        self._cached_wavelengths = self._compute_wavelengths()
        self._cache_reward_images()
        self._cache_center_masks()

        # Re-extract PyTorch tensors from rebuilt optical system
        self._init_fast_pipeline()

        if self.command_dm or self.ao_loop_active:
            self.optical_system.calibrate_dm_interaction_matrix(self.uuid)
            rcond = self.cfg['dm_interaction_rcond']
            self.reconstruction_matrix = hcipy.inverse_tikhonov(
                self.optical_system.interaction_matrix.transformation_matrix,
                rcond=rcond)
            self.episode_time_ms = 0.0

        _v4("Seeding initial action...")
        if self.cfg['init_differential_motion']:
            _v4("Applying initial differential motion...")
            self.optical_system._init_natural_diff_motion()
        self.optical_system._store_baseline_segment_displacements()

        _v4("Warm-up: %d initial frame(s)..." % self.frames_per_decision)
        for _ in range(self.frames_per_decision):
            (initial_state, _, _, _, info) = self.step(
                action=self.action_space.sample(),
                noisy_command=False, reset=True)

        self.state = initial_state
        self.steps_beyond_done = None
        _v4_section("Reset Complete")
        return self.state, info

    # ================================================================
    # Save state
    # ================================================================

    def save_state(self):
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
            copy.deepcopy(getattr(self.optical_system, 'pre_atmosphere_object_wavefront', None)))
        self.state_content["post_atmosphere_wavefronts"].append(
            copy.deepcopy(getattr(self.optical_system, 'post_atmosphere_wavefront', None)))
        self.state_content["segmented_mirror_surfaces"].append(
            copy.deepcopy(self.optical_system.segmented_mirror.surface))
        self.state_content["pupil_wavefronts"].append(
            copy.deepcopy(getattr(self.optical_system, 'pupil_wavefront', None)))
        self.state_content["post_dm_wavefronts"].append(
            copy.deepcopy(getattr(self.optical_system, 'post_dm_wavefront', None)))
        self.state_content["focal_plane_wavefronts"].append(
            copy.deepcopy(getattr(self.optical_system, 'focal_plane_wavefront', None)))
        self.state_content["instantaneous_psf"].append(
            copy.deepcopy(getattr(self.optical_system, 'instantaneous_psf', None)))
        self.state_content["readout_images"].append(
            copy.deepcopy(self.science_readout_raster))
        if self.report_time:
            _v4_timer("State deepcopy", time.time() - t0)

    # ================================================================
    # Step  (PyTorch fast path)
    # ================================================================

    def step(self, action, noisy_command=False, reset=False):
        """Execute one decision interval with GPU-accelerated optical sim."""
        # Local aliases
        _report = self.report_time
        _frames_per_decision = self.frames_per_decision
        _cmds_per_frame = self.commands_per_frame
        _ao_steps_per_cmd = self.ao_steps_per_command
        _wavelengths = self._cached_wavelengths
        _n_wl = float(len(_wavelengths))
        _ao_active = self.ao_loop_active
        _record = self.record_env_state_info
        _image_shape = self.image_shape
        _control_interval = self.control_interval_ms
        _ao_steps_per_frame = self.ao_steps_per_frame
        _frame_sec = self._frame_sec
        _optical_sys = self.optical_system
        _has_atm = self._has_atm
        _use_dm = _optical_sys.model_ao
        _dev = self._device

        if _report:
            step_t0 = time.time()

        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))

        if _record:
            for key in self.state_content:
                if isinstance(self.state_content[key], list):
                    self.state_content[key] = []
            self.state_content["shwfs_slopes"] = []

        self.action = self._flat_to_dict(action, self.dict_action_space)
        self.focal_plane_images = []
        self.shwfs_slopes_list = []
        _optical_sys._clipped_dof_count = 0
        _optical_sys._total_dof_count = 0

        integration_sec = _frame_sec / _ao_steps_per_frame

        # --- Main simulation loop ------------------------------------
        # Local refs to pre-allocated buffers (avoid dict/attr lookup)
        _atm_buf_np = self._atm_buf_np
        _t_atm_buf = self._t_atm_buf
        _t_seg_acts_buf = self._t_seg_acts_buf
        _t_seg_transform = self._t_seg_transform
        _t_dm_transform = getattr(self, '_t_dm_transform', None) if _use_dm else None
        _t_dm_acts_buf = getattr(self, '_t_dm_acts_buf', None) if _use_dm else None
        _t_zero_phase = self._t_zero_phase
        _t_props = self._t_props
        _wavenumbers = self._t_wavenumbers
        _inv_n_wl = 1.0 / _n_wl

        for frame_num in range(_frames_per_decision):
            # Accumulate frame on device
            frame_t = torch.zeros(_image_shape, dtype=torch.float32, device=_dev)

            for command_num in range(_cmds_per_frame):
                self.episode_time_ms += _control_interval

                # Atmosphere evolution (HCIPy -- cannot bypass)
                if _report:
                    t0 = time.time()
                _optical_sys.evolve_atmosphere_to(self.episode_time_ms)
                if _report:
                    _v4_timer("Atmosphere evolve", time.time() - t0)

                # Command dispatch (identical to v3)
                command = self.action[command_num]
                self._apply_commands(command)

                # Mirror actuators → GPU (once per command, not per wl)
                _t_seg_acts_buf.copy_(torch.from_numpy(
                    np.asarray(_optical_sys.segmented_mirror.actuators,
                               dtype=np.float32)))
                torch.mv(_t_seg_transform, _t_seg_acts_buf,
                         out=self._t_seg_surface)

                if _use_dm:
                    _t_dm_acts_buf.copy_(torch.from_numpy(
                        np.asarray(_optical_sys.dm.actuators,
                                   dtype=np.float32)))
                    torch.mv(_t_dm_transform, _t_dm_acts_buf,
                             out=self._t_dm_surface)

                # Atmosphere achromatic screen → GPU once per command
                # (phase_for(wl) = achromatic_screen / wl, so we transfer
                #  the achromatic screen once and scale by 1/wl on device)
                if _has_atm:
                    _atm_buf_np[:] = 0.0
                    for layer in _optical_sys.atmosphere_layers:
                        _atm_buf_np += np.asarray(
                            layer._shifted_achromatic_screen,
                            dtype=np.float32)
                    _t_atm_buf.copy_(torch.from_numpy(_atm_buf_np))

                for ao_step in range(_ao_steps_per_cmd):
                    for wl in _wavelengths:
                        if _report:
                            sim_t0 = time.time()

                        k = _wavenumbers[wl]
                        prop = _t_props[wl]

                        # --- FAST PATH: PyTorch on device ---

                        # 1. Atmosphere phase = achromatic / wl (on device)
                        if _has_atm:
                            atm_t = _t_atm_buf * (1.0 / wl)
                        else:
                            atm_t = _t_zero_phase

                        # 2. Fused optics (surfaces already on device)
                        self._fuse_optics(atm_t, k, _use_dm)

                        # 3. Propagation: pupil → focal (FFT or MFT)
                        self._propagate(prop)

                        # 4. PSF + science image (all on device)
                        image_t = self._compute_science_image_torch(
                            integration_sec)

                        if _report:
                            _v4_timer("Fast sim+image", time.time() - sim_t0,
                                      indent=2)

                        # 5. Accumulate
                        frame_t += image_t * _inv_n_wl

                        # AO correction (still HCIPy for SHWFS)
                        if _ao_active and not reset:
                            if _report:
                                ao_t0 = time.time()
                            # Reconstruct HCIPy wavefront for SHWFS
                            E_np = self._t_E_pupil.cpu().numpy().astype(
                                np.complex128)
                            wf = hcipy.Wavefront(
                                hcipy.Field(E_np, _optical_sys.pupil_grid), wl)
                            _optical_sys.post_dm_wavefront = wf
                            self._run_ao_correction(integration_sec)
                            if _report:
                                _v4_timer("AO correction",
                                          time.time() - ao_t0, indent=2)

                    # State recording
                    if _record and not reset:
                        # Transfer readout raster to CPU only when recording
                        self.science_readout_raster = image_t.cpu().numpy()
                        # Run full HCIPy sim to populate state objects
                        _optical_sys.simulate(wl)
                        _optical_sys.get_science_frame(
                            integration_seconds=integration_sec)
                        self.save_state()

            # --- Transfer frame to CPU for detector model + observation ---
            frame_np = frame_t.cpu().numpy()
            frame_np = self._apply_detector_model(frame_np)
            self.focal_plane_images.append(frame_np)

        # --- Encode observation --------------------------------------
        if self._obs_image_only:
            self.state = np.array(self.focal_plane_images)
        elif self._obs_image_action:
            self.state = {'image': np.array(self.focal_plane_images),
                          'prior_action': action}
        else:
            raise ValueError("Invalid observation_mode: '%s'" % self.observation_mode)

        # --- Compute reward ------------------------------------------
        reward = self._reward_fn()

        # L2 action penalty: reward -= w * |reward| * mean(a^2).
        # With actions in [-1, 1], mean(a^2) ∈ [0, 1].
        # Default w = 0.03 → 3% worse at full-magnitude action,
        # 0% at zero action. Works for both positive and negative rewards.
        if self._action_penalty:
            action_sq_mean = float(np.mean(np.square(action)))
            reward = reward - self._action_penalty_weight * abs(reward) * action_sq_mean

        # Out-of-bounds penalty: penalize when actuator commands hit
        # physical clipping limits.  oob_frac = fraction of DOFs that
        # were clipped during this step's command_secondaries() calls.
        oob_frac = 0.0
        if self._oob_penalty and self.optical_system._total_dof_count > 0:
            oob_frac = float(self.optical_system._clipped_dof_count) / float(self.optical_system._total_dof_count)
            reward = reward - self._oob_penalty_weight * oob_frac

        terminated = False
        truncated = False

        # --- Debug render: compare perfect vs observed (once) ---------
        if not getattr(self, '_debug_rendered', False):
            self._debug_rendered = True
            fpi0 = self.focal_plane_images[0]
            pdn = self._perfect_image_dn
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            im0 = axes[0].imshow(pdn, origin='lower')
            axes[0].set_title("perfect_image_dn\nmax=%.4f" % np.max(pdn))
            plt.colorbar(im0, ax=axes[0])
            im1 = axes[1].imshow(fpi0, origin='lower')
            axes[1].set_title("focal_plane_image\nmax=%.4f" % np.max(fpi0))
            plt.colorbar(im1, ax=axes[1])
            diff = fpi0 - pdn
            im2 = axes[2].imshow(diff, origin='lower', cmap='RdBu_r')
            axes[2].set_title("diff (obs - perfect)\nrange=[%.4f, %.4f]"
                              % (np.min(diff), np.max(diff)))
            plt.colorbar(im2, ax=axes[2])
            fig.suptitle("MSE=%.6f  Strehl=%.6f" % (
                float(np.mean((fpi0.flatten() - pdn.flatten()) ** 2))
                / (self._perfect_image_max_dn ** 2),
                float(np.max(fpi0)) / self._perfect_image_max_dn))
            plt.tight_layout()
            _dbg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tmp", "optomech_debug_reward.png")
            plt.savefig(_dbg_path, dpi=150)
            plt.close(fig)
            print("[DEBUG] Saved %s" % _dbg_path)

        # --- Diagnostic metrics (always reported) --------------------
        strehls = [self._absolute_strehl(fpi)
                   for fpi in self.focal_plane_images]
        _max_dn = self._perfect_image_max_dn
        _perfect_flat = self._perfect_image_dn.flatten()
        mses = [float(np.mean((fpi.flatten() - _perfect_flat) ** 2)) / (_max_dn ** 2)
                for fpi in self.focal_plane_images]

        info = {}
        info["strehl"] = float(np.mean(strehls))
        info["mse"] = float(np.mean(mses))
        info["reward_raw"] = float(reward)
        info["oob_frac"] = oob_frac
        if _record:
            info["state_content"] = self.state_content
            info["state"] = self.state

        if _report:
            _v4_timer("STEP TOTAL", time.time() - step_t0)

        reward = np.float32(reward)
        return self.state, reward, terminated, truncated, info

    # ================================================================
    # Command dispatch
    # ================================================================

    def _apply_commands(self, command):
        idx = 0
        sec_idx = ten_idx = dm_idx = None
        if self.command_secondaries:
            sec_idx = idx; idx += 1
        if self.command_tensioners:
            ten_idx = idx; idx += 1
        if self.command_dm:
            dm_idx = idx; idx += 1
        if self.command_tensioners:
            self.optical_system.command_tensioners(command[ten_idx])
        if self.command_secondaries:
            self.optical_system.command_secondaries(command[sec_idx])
        if self.command_dm:
            self.optical_system.command_dm(command[dm_idx])

    # ================================================================
    # AO loop
    # ================================================================

    def _run_ao_correction(self, integration_seconds):
        shwfs_vec = self.optical_system.get_shwfs_frame(
            integration_seconds=integration_seconds)
        slopes = self.optical_system.shwfse.estimate([shwfs_vec + 1e-10])
        slopes -= self.optical_system.reference_slopes
        self.shwfs_slopes = slopes.ravel()
        self.shwfs_slopes_list.append(self.shwfs_slopes)
        self.optical_system.dm.actuators = (
            (1 - self.dm_leakage) * self.optical_system.dm.actuators
            - self.dm_gain * self.reconstruction_matrix.dot(self.shwfs_slopes))
        self.optical_system.dm.actuators = np.clip(
            self.optical_system.dm.actuators,
            -self._dm_stroke_limit_m, self._dm_stroke_limit_m)

    # ================================================================
    # Detector model
    # ================================================================

    def _apply_detector_model(self, frame):
        energy_joules = frame * self._frame_sec
        n_photons = energy_joules / self._photon_energy
        n_electrons = n_photons * self._det_qe
        dn = n_electrons / self._det_gain
        return np.clip(dn, 0, self._det_max_dn)

    # ================================================================
    # Reward computation
    # ================================================================
    #
    # All Strehl-based rewards use ABSOLUTE Strehl:
    #   Strehl = max(observed_dn) / max(perfect_dn)
    # Both images in detector DN units.  No self-normalization.

    def _compute_reward(self):
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
        """Log-MSE alignment reward using absolute normalization."""
        _max_dn = self._perfect_image_max_dn
        norm_target = self._perfect_image_dn / _max_dn
        radius_max = None
        if not radius_max:
            center_mask = self._center_mask_standard
        else:
            center_mask = self._center_mask_expanded
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
        if not radius_max:
            center_mask = self._center_mask_standard
        else:
            center_mask = self._center_mask_expanded
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
