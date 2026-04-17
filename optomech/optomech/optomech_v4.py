"""
Optomech V4 -- GPU-accelerated Gymnasium environment for distributed-aperture
telescope control.

Built on top of v3, this version moves the optical propagation hot path to
GPU via PyTorch tensors.  HCIPy is used **only at init/reset time** to
construct aperture geometry, segment influence functions, and Fraunhofer
propagator phase arrays.  All per-step computation (wavefront propagation,
FFT convolution, detector model, reward) runs on GPU.

Functionally identical to v3 -- same action/observation spaces, same reward
values given identical seeds and configurations (within float32 tolerance).

Author: Justin Fletcher (original), refactored for v2, optimized for v3,
        GPU-accelerated for v4.
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
    "init_wind_piston_micron_std": 3.0,
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
    "minimum_absolute_action": 0.0,
    "env_action_scale": 1.0,

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
    "reward_weight_centering": 0.0,
    "reward_weight_flux": 0.0,
    "reward_weight_convex_flux": 0.0,
    "convex_flux_power": 2.0,
    "reward_weight_dist": 0.0,
    "reward_weight_concentration": 0.0,
    "reward_weight_peak": 0.0,
    "reward_weight_centered_strehl": 0.0,
    "centering_sigma_fraction": 0.25,
    "centering_mode": "gaussian",           # "gaussian" or "circular"
    "centering_radius_fraction": 0.25,      # radius as fraction of image size (circular mode)
    "reward_weight_shape": 1.0,
    "reward_threshold": 25.0,
    "align_radius": 32,
    "align_radius_max_expand": 64,
    "align_mse_expand_threshold": -1.25,
    "ao_closed_inv_slope_threshold": 2e6,
    "dark_hole_alpha": 0.0,
    "action_penalty": True,
    "action_penalty_weight": 0.03,
    "oob_penalty": True,
    "oob_penalty_weight": 2.0,
    "holding_bonus_weight": 0.0,
    "holding_bonus_min_reward": -1.0,
    "holding_bonus_threshold": 0.0,

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
    "detector_poisson_noise": False,
    "detector_quantize_adc": False,

    # --- State recording ---------------------------------------------
    "record_env_state_info": False,
    "write_env_state_info": False,
    "state_info_save_dir": "./tmp/",

    # --- Episode -----------------------------------------------------
    "num_episodes": 1,
    "num_steps": 16,

    # --- Optomech version tag ----------------------------------------
    "optomech_version": "v4",

    # --- GPU acceleration (v4) ----------------------------------------
    "device": "auto",  # "auto", "cuda", "mps", "cpu"

    # --- Incremental bootstrapping ------------------------------------
    # When bootstrap_phase is True, bootstrap_phased_count segments
    # (indices 0..phased_count-1) start co-phased, the target segment
    # (index phased_count) is perturbed, and all remaining segments are
    # tipped/tilted to push their light fully off the focal plane.
    # Reference images are computed for the goal state (phased_count+1
    # segments co-phased).  When False, everything works as before.
    "bootstrap_phase": False,
    "bootstrap_phased_count": 0,
    # Per-DOF action penalty multiplier for non-target segments during
    # bootstrap.  Applied to all segments except the target (phased_count).
    # Set to 0.0 to disable (only the base action_penalty_weight applies).
    "bootstrap_nontarget_penalty_multiplier": 10.0,
}


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def _resolve_device(cfg_device):
    """Resolve 'auto' to the best available torch device."""
    if cfg_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(cfg_device)


# ===================================================================
# V4 Logging  (green tag to distinguish from v3 magenta)
# ===================================================================

_V3_TAG = "\033[92m[optomech-v4]\033[0m"
_V3_TAG_PLAIN = "[optomech-v4]"

_TOP    = "\u2554" + "\u2550" * 58 + "\u2557"
_BOT    = "\u255A" + "\u2550" * 58 + "\u255D"
_SIDE   = "\u2551"
_MID    = "\u2560" + "\u2550" * 58 + "\u2563"
_THIN   = "\u2502"
_HTHIN  = "\u2500" * 58


def _v3(msg, indent=0):
    """Print with the v3 prefix tag and optional indent."""
    pad = "  " * indent
    print(f"{_V3_TAG} {pad}{msg}")


def _v3_section(title):
    """Print a box-drawn section header."""
    inner = f" {title} ".center(58)
    print(f"{_V3_TAG} {_TOP}")
    print(f"{_V3_TAG} {_SIDE}{inner}{_SIDE}")
    print(f"{_V3_TAG} {_BOT}")


def _v3_subsection(title):
    """Print a lighter sub-section separator."""
    inner = f" {title} ".center(58, "\u2500")
    print(f"{_V3_TAG} {inner}")


def _v3_kv(key, value, indent=1):
    """Print a key-value pair."""
    pad = "  " * indent
    print(f"{_V3_TAG} {pad}{key:<42s} {value}")


def _v3_timer(label, elapsed, indent=1):
    """Print a timing measurement."""
    pad = "  " * indent
    bar_len = min(int(elapsed * 200), 30)
    bar = "\u2588" * bar_len
    print(f"{_V3_TAG} {pad}\u23f1  {label:<34s} {elapsed:8.4f}s {bar}")


def _print_config_banner(cfg, title="Optomech V3 Configuration"):
    """Pretty-print all configuration key-value pairs inside a box."""
    _v3_section(title)
    max_key_len = max(len(k) for k in cfg)
    prev_section = None
    for k in sorted(cfg.keys()):
        section = k.split("_")[0]
        if section != prev_section and prev_section is not None:
            print(f"{_V3_TAG}   {'':>{max_key_len}}   {'':>10}")
        prev_section = section
        print(f"{_V3_TAG}   {k:<{max_key_len}}  =  {cfg[k]}")
    print(f"{_V3_TAG} {_BOT}")


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
        """Two point sources separated by a configurable angular distance.

        Parameters (via kwargs)
        -----------------------
        binary_separation_arcsec : float
            Angular separation between the two sources in arcseconds.
            Default 0.2 arcsec.
        ifov : float
            Instantaneous field of view in arcsec/pixel.  If not provided
            the legacy NanoELF value (0.0165012) is used.
        """
        sep_arcsec = kwargs.get("binary_separation_arcsec", 0.2)
        ifov = kwargs.get("ifov", 0.0165012)
        separation_pixels = max(2, int(round(sep_arcsec / ifov)))
        cx = self.extent_pixels // 2
        cy = self.extent_pixels // 2
        half_sep = separation_pixels // 2
        array_value = 1.02e-8
        arr = np.zeros((self.extent_pixels, self.extent_pixels))
        arr[cx, cy - half_sep] = array_value
        arr[cx, cy + half_sep] = array_value
        return arr

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
    """End-to-end optical simulation with pre-computed caches for speed."""

    def __init__(self, **kwargs):

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

        # [v3-opt] Pre-compute DM stroke limit (used in command_dm and AO loop)
        self._dm_stroke_limit_m = (
            self.stroke_count_limit * self.microns_opd_per_actuator_bit * 1e-6 / 2.0)

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
        _v3_subsection("Aperture: %s" % aperture_type)
        (aperture_func, segments_func,
         focal_length, pupil_diameter,
         focal_plane_image_size_meters) = self._build_aperture(
             aperture_type, cfg)

        # Store for bootstrap off-axis computation
        self._focal_length = focal_length
        self._focal_plane_image_size_meters = focal_plane_image_size_meters

        # --- Optical grid parameters ---------------------------------
        num_px = cfg['focal_plane_image_size_pixels']
        self.wavelength = cfg['wavelength']
        oversampling_factor = cfg['oversampling_factor']

        # --- Atmosphere parameters -----------------------------------
        seeing = cfg['seeing_arcsec']
        outer_scale = cfg['outer_scale_meters']
        tau0 = cfg['tau0_seconds']

        fried_parameter = hcipy.seeing_to_fried_parameter(seeing)
        _v3_kv("fried_parameter", fried_parameter)
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

        _v3_subsection("Focal Plane Geometry")
        _v3_kv("grid pixels",            "%d" % num_px)
        _v3_kv("extent (m)",             "%.6e" % focal_plane_extent_metres)
        _v3_kv("pixel extent (m)",       "%.6e" % focal_plane_pixel_extent_meters)
        _v3_kv("resolution element (m)", "%.6e" % focal_plane_resolution_element)
        _v3_kv("pixels per meter",       "%.2f" % focal_plane_pixels_per_meter)
        _v3_kv("num airy (res-el radii)","%.4f" % num_airy)
        _v3_kv("sampling (px/res-el)",   "%.4f" % sampling)
        _v3_kv("iFOV (arcsec/px)",       "%.7f" % self.ifov)
        _v3_kv("FOV (arcsec)",           "%.4f" % fov)
        _v3_kv("incremental_control",    str(self.incremental_control))

        # --- Object plane --------------------------------------------
        _v3("Building object plane...")
        self.object_plane = ObjectPlane(
            object_type=cfg['object_type'],
            object_plane_extent_pixels=num_px,
            object_plane_extent_meters=cfg['object_plane_extent_meters'],
            object_plane_distance_meters=cfg['object_plane_distance_meters'],
            ifov=self.ifov,
        )

        # [v3-opt] Pre-compute object spectrum (object never changes within episode)
        self._cached_object_spectrum = sp_fft.fft2(self.object_plane.array)

        # --- Pupil grid ----------------------------------------------
        _v3("Building pupil grid...")
        self.pupil_grid = hcipy.make_pupil_grid(
            dims=num_px, diameter=pupil_diameter)

        # --- Atmosphere layers ---------------------------------------
        self.atmosphere_layers = []
        _v3("Building %d atmosphere layer(s)..." % cfg['num_atmosphere_layers'])
        for _ in range(cfg['num_atmosphere_layers']):
            layer = hcipy.InfiniteAtmosphericLayer(
                self.pupil_grid, Cn_squared, outer_scale, velocity)
            self.atmosphere_layers.append(layer)

        # --- Focal grid & propagator ---------------------------------
        focal_grid = hcipy.make_pupil_grid(
            dims=num_px, diameter=focal_plane_extent_metres)
        focal_grid = focal_grid.shifted(focal_grid.delta / 2)

        _v3("Building Fraunhofer propagator...")
        self.pupil_to_focal_propagator = hcipy.FraunhoferPropagator(
            self.pupil_grid, focal_grid, focal_length)

        # --- Evaluate aperture / segments ----------------------------
        aperture_field = hcipy.evaluate_supersampled(
            aperture_func, self.pupil_grid, oversampling_factor)
        segments_field = hcipy.evaluate_supersampled(
            segments_func, self.pupil_grid, oversampling_factor)

        self.segmented_mirror = hcipy.SegmentedDeformableMirror(segments_field)
        self.aperture = aperture_field

        # --- Bootstrap: push excluded segments off focal plane -------
        # For the perfect-image and reference-flux computation, use the
        # GOAL state of phase N: segments 0..phased_count (inclusive —
        # i.e. phased_count+1 segments) are aligned, the rest are
        # off-axis. This matches the start state of phase N+1, so a
        # composite rollout can chain phase N's "goal" straight into
        # phase N+1's "start" without discontinuity. The reset-time
        # training setup uses phased_count directly (start state).
        _bootstrap = cfg.get('bootstrap_phase', False)
        _phased_count = cfg.get('bootstrap_phased_count', 0)
        if _bootstrap:
            _v3("Bootstrap mode: %d co-phased at start, target=%d, "
                "%d excluded at start; reference uses goal state "
                "(%d co-phased)"
                % (_phased_count, _phased_count,
                   self.num_apertures - _phased_count - 1,
                   _phased_count + 1))
            # phased_count + 1 here → segs (phased_count+1)..N-1 off-axis,
            # segs 0..phased_count aligned (the goal for phase N).
            # Clamps inside _init_bootstrap_segments via range(); when
            # phased_count+1 == num_apertures the off-axis loop is empty.
            self._init_bootstrap_segments(_phased_count + 1, noisy=False)

        # --- Perfect (reference) image -------------------------------
        # Polychromatic perfect PSF: average across all sampled wavelengths
        _perfect_wavelengths = _centered_wavelengths(
            self.wavelength,
            cfg['bandwidth_nanometers'],
            cfg['bandwidth_sampling'])
        _v3("Computing polychromatic perfect PSF (%d wavelength(s): %s nm)"
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

        # --- Science camera ------------------------------------------
        # Created before diff-motion so the reference flux can be
        # measured with aligned segments.
        _v3("Building science camera...")
        self.camera = hcipy.NoiselessDetector(focal_grid)

        # --- Reference total flux (aligned system) -------------------
        # One forward pass through the full pipeline with aligned
        # segments.  Stored as the expected sum(DN) when all light
        # lands on the detector.  Used by _absolute_strehl to detect
        # and correct for flux loss when tip/tilt pushes the PSF off
        # the focal plane.
        self._compute_reference_flux(cfg)

        # --- Initialize differential motion --------------------------
        if self.init_differential_motion:
            _v3("Applying initial differential motion...")
            self._init_natural_diff_motion()

        self._store_baseline_segment_displacements()

        # [v3-opt] Pre-compute secondary correction limits (constant for lifetime)
        self._max_p_m = cfg["max_piston_correction_micron"] * 1e-6
        self._max_t_r = cfg["max_tip_correction_arcsec"] * np.pi / (180 * 3600)
        self._max_tl_r = cfg["max_tilt_correction_arcsec"] * np.pi / (180 * 3600)

        # ============================================================
        # V4: Extract GPU tensors from HCIPy objects
        # ============================================================
        self._torch_device = _resolve_device(cfg.get("device", "auto"))
        _v3("GPU device: %s" % self._torch_device)

        num_px_2d = num_px  # save for reshape

        # Aperture field → complex tensor (num_px, num_px)
        _ap_np = np.array(self.aperture).reshape(num_px, num_px)
        self._aperture_t = torch.tensor(_ap_np, dtype=torch.complex64,
                                         device=self._torch_device)

        # Segment influence functions from SegmentedDeformableMirror
        # HCIPy stores them as a ModeBasis (list of Field objects).
        # Stack into (n_modes, n_pixels) array.
        _inf_list = [np.array(mode) for mode in self.segmented_mirror._influence_functions]
        _inf_np = np.stack(_inf_list, axis=0)  # (n_modes, n_pixels)
        n_modes = _inf_np.shape[0]  # 3 * num_apertures
        self._influence_t = torch.tensor(
            _inf_np.reshape(n_modes, num_px, num_px),
            dtype=torch.float32, device=self._torch_device)

        # Actuator state tensor (mirrors segmented_mirror.actuators layout:
        # [piston_0..n, tip_0..n, tilt_0..n])
        self._actuators_t = torch.zeros(n_modes, dtype=torch.float32,
                                         device=self._torch_device)

        # Pre-compute object spectrum on GPU
        self._object_spectrum_t = torch.fft.fft2(
            torch.tensor(self.object_plane.array, dtype=torch.complex64,
                         device=self._torch_device))

        # ---- Extract MFT matrices from HCIPy's Fraunhofer propagator ----
        # HCIPy uses a MatrixFourierTransform for this grid configuration.
        # The MFT computes: E_focal = norm * w_in * (M1 @ E_pupil_2d @ M2)
        # where M1 = exp(-1j * outer(v,y)), M2 = exp(-1j * outer(x,u))
        # are separable DFT kernels, w_in = pupil grid weight (scalar),
        # and norm = 1/(1j * f * lambda).
        #
        # IMPORTANT: M1,M2 are wavelength-dependent because the uv_grid
        # is scaled by 2*pi/(f*lambda).  We pre-compute and cache
        # per-wavelength matrices for all sampled wavelengths.
        _all_wl = _centered_wavelengths(
            self.wavelength,
            cfg['bandwidth_nanometers'],
            cfg['bandwidth_sampling'])
        _v3("Pre-computing MFT matrices for %d wavelength(s)..." % len(_all_wl))

        self._mft_cache = {}  # {wavelength: (M1_t, M2_t, norm_scale)}
        _focal_gw = None
        for _wl in _all_wl:
            _wf = hcipy.Wavefront(self.aperture, _wl)
            _result = self.pupil_to_focal_propagator(_wf)
            _idata = self.pupil_to_focal_propagator.get_instance_data(
                self.pupil_grid, None, _wl)
            _mft = _idata.fourier_transform
            _norm = _idata.norm_factor  # = 1/(1j*f*wl)

            # Extract input weight (scalar for uniform grid)
            _w_in = float(_mft.weights_input)
            # Combined scalar: norm * w_in
            _scale = complex(_norm * _w_in)

            _M1_t = torch.tensor(
                _mft.M1, dtype=torch.complex64, device=self._torch_device)
            _M2_t = torch.tensor(
                _mft.M2, dtype=torch.complex64, device=self._torch_device)
            self._mft_cache[_wl] = (_M1_t, _M2_t, _scale)

            if _focal_gw is None:
                _gw = np.array(_result.grid.weights)
                _focal_gw = float(_gw) if np.isscalar(_gw) or _gw.ndim == 0 else float(_gw[0])

        # Focal-plane grid weights (for camera readout: PSF = |E|^2 * gw * dt)
        self._focal_grid_weight = _focal_gw

        # Detector model constants on GPU (scalars, but kept as Python floats)
        _h = 6.62607015e-34
        _c_light = 2.99792458e8
        self._photon_energy = _h * _c_light / cfg['wavelength']
        self._det_qe = cfg['detector_quantum_efficiency']
        self._det_gain = cfg['detector_gain_e_per_dn']
        self._det_max_dn = cfg['detector_max_dn']

        # Store grid size for reshaping
        self._num_px = num_px

        _v3_subsection("Optical System Ready")

    # ================================================================
    # Reference flux measurement
    # ================================================================

    def _compute_reference_flux(self, cfg):
        """Forward pass with aligned segments through the full pipeline.

        Measures the expected total DN when all light lands on the
        detector.  Stored as ``self._reference_fpi_sum`` for use by
        the Strehl flux-fraction correction in OptomechEnv.

        Must be called while segments are still aligned (before
        ``_init_natural_diff_motion``).  The detector model is inlined
        here so we don't depend on OptomechEnv.
        """
        import math as _math

        wavelengths = _centered_wavelengths(
            self.wavelength,
            cfg['bandwidth_nanometers'],
            cfg['bandwidth_sampling'])

        # Compute integration time that matches the step loop.
        ao_interval_ms = cfg['ao_interval_ms']
        control_interval_ms = cfg['control_interval_ms']
        frame_interval_ms = cfg['frame_interval_ms']
        ao_steps_per_cmd = _math.ceil(control_interval_ms / ao_interval_ms)
        cmds_per_frame = _math.ceil(frame_interval_ms / control_interval_ms)
        ao_steps_per_frame = ao_steps_per_cmd * cmds_per_frame
        frame_sec = frame_interval_ms / 1000.0
        integration_sec = frame_sec / ao_steps_per_frame

        num_px = cfg['focal_plane_image_size_pixels']
        image_shape = (num_px, num_px)
        n_wl = float(len(wavelengths))

        # Accumulate one aligned science frame per wavelength (mirrors
        # the inner step loop; with aligned segments every AO sub-step
        # is identical, so one pass is equivalent to ao_steps_per_frame
        # identical passes).
        frame = np.zeros(image_shape, dtype=np.float32)
        for wl in wavelengths:
            self.simulate_cpu(wl)
            science = self.get_science_frame_cpu(
                integration_seconds=integration_sec)
            frame += np.reshape(science, image_shape) / n_wl

        # Scale by ao_steps_per_frame to match the accumulated frame
        # in the step loop (each AO sub-step contributes identically
        # when the mirror is aligned).
        frame *= ao_steps_per_frame

        # Inline detector model: power → DN
        _h = 6.62607015e-34
        _c = 2.99792458e8
        photon_energy = _h * _c / cfg['wavelength']
        det_qe = cfg['detector_quantum_efficiency']
        det_gain = cfg['detector_gain_e_per_dn']
        det_max_dn = cfg['detector_max_dn']

        energy_joules = frame * frame_sec
        n_photons = energy_joules / photon_energy
        n_electrons = n_photons * det_qe
        dn = n_electrons / det_gain
        ref_fpi = np.clip(dn, 0, det_max_dn)

        self._reference_fpi_sum = float(np.sum(ref_fpi))
        self._reference_fpi_max = float(np.max(ref_fpi))

        # Weighted reference sum for the centering reward.
        ctr = num_px // 2
        y, x = np.ogrid[:num_px, :num_px]
        dist_sq = (y - ctr) ** 2 + (x - ctr) ** 2
        centering_mode = cfg.get('centering_mode', 'gaussian')
        if centering_mode == 'circular':
            r_px = cfg.get('centering_radius_fraction', 0.25) * num_px
            cen_w = (dist_sq <= r_px ** 2).astype(np.float64)
        else:
            sigma_px = cfg.get('centering_sigma_fraction', 0.25) * num_px
            cen_w = np.exp(-dist_sq.astype(np.float64) / (2.0 * sigma_px ** 2))
        self._reference_centering_sum = float(np.sum(ref_fpi * cen_w))

        # Reference distance-weighted sum for the dist penalty.
        # This is the distance score of the perfectly centred PSF — the
        # best achievable value.  We normalise by ref_sum so the metric
        # is the flux-weighted mean normalised distance in [0, 1].
        dist = np.sqrt(dist_sq.astype(np.float64))
        max_dist = np.sqrt(2.0) * ctr
        dist_norm = dist / max_dist  # [0, 1]
        if self._reference_fpi_sum > 0:
            self._reference_dist_score = float(
                np.sum(ref_fpi * dist_norm)) / self._reference_fpi_sum
        else:
            self._reference_dist_score = 0.0

        _v3("Reference flux sum (DN): %.4f" % self._reference_fpi_sum)
        _v3("Reference flux max (DN): %.4f" % self._reference_fpi_max)
        _v3("Reference centering sum: %.4f" % self._reference_centering_sum)
        _v3("Reference dist score: %.4f" % self._reference_dist_score)

        # Reference concentration (inverse participation ratio).
        # C = sum(I²) / sum(I)²  — higher when light is more peaked.
        ref_sum_sq = float(np.sum(ref_fpi.astype(np.float64) ** 2))
        if self._reference_fpi_sum > 0:
            self._reference_concentration = ref_sum_sq / (self._reference_fpi_sum ** 2)
        else:
            self._reference_concentration = 0.0
        _v3("Reference concentration: %.6f" % self._reference_concentration)

    # ================================================================
    # Aperture construction
    # ================================================================

    def _build_aperture(self, aperture_type, cfg):
        """Build the selected aperture and return (aperture, segments,
        focal_length, pupil_diameter, focal_plane_image_size_meters)."""
        if aperture_type == "elf":
            self.num_apertures = 15
            focal_plane_image_size_meters = 3.611e-4
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

        dm_model_type = cfg['dm_model_type']
        _v3("Building SHWFS (f/%.0f, %d lenslets)..." % (f_number, num_lenslets))
        self.shwfs = hcipy.SquareShackHartmannWavefrontSensorOptics(
            self.pupil_grid.scaled(magnification),
            f_number, num_lenslets, sh_diameter)
        self.shwfse = hcipy.ShackHartmannWavefrontSensorEstimator(
            self.shwfs.mla_grid,
            self.shwfs.micro_lens_array.mla_index)

        self.shwfs_camera = hcipy.NoiselessDetector(focal_grid)

        _v3("Building DM (model=%s)..." % dm_model_type)
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
        # [v3-opt] Use pre-computed stroke limit instead of recomputing
        dm_stroke_meters = self._dm_stroke_limit_m * 2.0
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

    def simulate_cpu(self, wavelength):
        """Propagate a wavefront through the full optical train (CPU/HCIPy)."""
        _report = self.report_time

        t0 = time.time()
        self.object_wavefront = hcipy.Wavefront(self.aperture, wavelength)
        self.pre_atmosphere_object_wavefront = self.object_wavefront
        if _report:
            _v3_timer("Object wavefront", time.time() - t0, indent=2)

        t0 = time.time()
        wf = self.pre_atmosphere_object_wavefront
        for atm_layer in self.atmosphere_layers:
            wf = atm_layer.forward(wf)
        self.post_atmosphere_wavefront = wf
        if _report:
            _v3_timer("Atmosphere forward", time.time() - t0, indent=2)

        if self.simulate_differential_motion:
            t0 = time.time()
            self._simulate_natural_diff_motion()
            if _report:
                _v3_timer("Diff-motion step", time.time() - t0, indent=2)

        t0 = time.time()
        self.pupil_wavefront = self.segmented_mirror(
            self.post_atmosphere_wavefront)
        if _report:
            _v3_timer("Segments forward", time.time() - t0, indent=2)

        if self.model_ao:
            t0 = time.time()
            self.post_dm_wavefront = self.dm.forward(self.pupil_wavefront)
            if _report:
                _v3_timer("DM forward", time.time() - t0, indent=2)
        else:
            self.post_dm_wavefront = self.pupil_wavefront

        t0 = time.time()
        self.focal_plane_wavefront = self.pupil_to_focal_propagator(
            self.post_dm_wavefront)
        if _report:
            _v3_timer("Pupil-to-focal prop", time.time() - t0, indent=2)

    def simulate(self, wavelength):
        """GPU-accelerated wavefront propagation via Matrix Fourier Transform.

        Replicates HCIPy's FraunhoferPropagator + MatrixFourierTransform
        pipeline exactly:
            E_focal = norm * w_in * (M1 @ E_pupil @ M2)
        where norm = 1/(1j*f*lambda), w_in = pupil grid weight,
        M1 and M2 are pre-extracted DFT kernels.
        """
        dev = self._torch_device
        k = 2.0 * math.pi / wavelength

        # Start with aperture field (complex amplitude, zero phase)
        E = self._aperture_t.clone()

        # Atmosphere: keep on CPU via HCIPy if layers exist
        if self.atmosphere_layers:
            # Fall back to HCIPy for atmosphere, transfer result
            wf = hcipy.Wavefront(self.aperture, wavelength)
            for atm_layer in self.atmosphere_layers:
                wf = atm_layer.forward(wf)
            atm_field = np.array(wf.electric_field).reshape(self._num_px, self._num_px)
            E = torch.tensor(atm_field, dtype=torch.complex64, device=dev)

        # Segmented mirror: surface = sum(actuators_i * influence_i)
        # then E *= exp(2j * k * surface)
        surface = torch.einsum('i,ihw->hw', self._actuators_t, self._influence_t)
        E = E * torch.exp(torch.complex(
            torch.zeros_like(surface),
            (2.0 * k * surface)))

        # Fraunhofer propagation via MFT (two separable matrix multiplies)
        # E_focal = scale * (M1 @ E @ M2)
        # where scale = norm_factor * w_in = 1/(1j*f*lambda) * grid_weight
        # M1, M2, scale are pre-computed per wavelength.
        if wavelength not in self._mft_cache:
            # On-the-fly: compute MFT matrices for uncached wavelength
            self._cache_mft_for_wavelength(wavelength)
        _M1, _M2, _scale = self._mft_cache[wavelength]
        E_focal = torch.matmul(torch.matmul(_M1, E), _M2) * _scale

        self._focal_field_t = E_focal
        # Also store as HCIPy-compatible for reference flux code paths
        self.focal_plane_wavefront = None  # signal that GPU path was used

    def _cache_mft_for_wavelength(self, wavelength):
        """Compute and cache MFT matrices for an uncached wavelength."""
        import hcipy as _hcipy
        _wf = _hcipy.Wavefront(self.aperture, wavelength)
        _result = self.pupil_to_focal_propagator(_wf)
        _idata = self.pupil_to_focal_propagator.get_instance_data(
            self.pupil_grid, None, wavelength)
        _mft = _idata.fourier_transform
        _norm = _idata.norm_factor
        _w_in = float(_mft.weights_input)
        _scale = complex(_norm * _w_in)
        _M1_t = torch.tensor(
            _mft.M1, dtype=torch.complex64, device=self._torch_device)
        _M2_t = torch.tensor(
            _mft.M2, dtype=torch.complex64, device=self._torch_device)
        self._mft_cache[wavelength] = (_M1_t, _M2_t, _scale)

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

        wf = hcipy.Wavefront(self.aperture, self.wavelength)
        wf.total_power = 1
        self.shwfs_camera.integrate(self.shwfs(self.magnifier(wf)), 1)
        reference_image = self.shwfs_camera.read_out()

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

        self.reference_slopes = self.shwfse.estimate([reference_image])

        dm_cache_path = os.path.join(cfg['dm_cache_dir'], str(env_uuid))
        if os.path.exists(dm_cache_path):
            _v3("Found cached interaction matrix at %s" % dm_cache_path)
            with open(os.path.join(dm_cache_path,
                      'dm_interaction_matrix.pkl'), 'rb') as f:
                self.interaction_matrix = pickle.load(f)
                return

        n_act = len(self.dm.actuators)
        _v3_subsection("DM Calibration (%d actuators)" % n_act)
        response_matrix = []
        for i in range(n_act):
            if i % 50 == 0 or i == n_act - 1:
                _v3("  actuator %d / %d" % (i + 1, n_act))
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

    # ================================================================
    # Science camera readout
    # ================================================================

    def get_science_frame_cpu(self, integration_seconds=1.0):
        """Read the science camera, applying geometric-optics convolution (CPU/HCIPy).

        [v3-opt] Uses pre-cached object_spectrum and scipy.fft with workers.
        """
        _report = self.report_time

        t0 = time.time()
        self.camera.integrate(self.focal_plane_wavefront, integration_seconds)
        if _report:
            _v3_timer("Camera integrate", time.time() - t0, indent=2)

        t0 = time.time()
        effective_psf = self.camera.read_out()
        side = int(np.sqrt(effective_psf.size))
        effective_psf = effective_psf.reshape((side, side))
        self.instantaneous_psf = effective_psf
        if _report:
            _v3_timer("Camera readout", time.time() - t0, indent=2)

        # [v3-opt] scipy.fft with workers=-1 for multi-threaded FFT,
        # plus pre-cached object spectrum.
        t0 = time.time()
        effective_otf = sp_fft.fft2(effective_psf, workers=-1)
        image_spectrum = self._cached_object_spectrum * effective_otf
        self.readout_image = np.abs(
            sp_fft.fftshift(sp_fft.ifft2(image_spectrum, workers=-1)))
        if _report:
            _v3_timer("FFT convolution", time.time() - t0, indent=2)

        return self.readout_image

    def get_science_frame(self, integration_seconds=1.0):
        """GPU-accelerated science frame readout.

        Replicates HCIPy NoiselessDetector: PSF = |E|^2 * grid_weight * dt,
        then convolves with pre-cached object spectrum.
        """
        # PSF = |E|^2 * grid_weight * dt  (matches HCIPy camera.integrate)
        psf = (torch.abs(self._focal_field_t) ** 2) * (
            self._focal_grid_weight * integration_seconds)

        # Convolve with object spectrum: image = |IFFT(FFT(PSF) * FFT(object))|
        otf = torch.fft.fft2(psf)
        image = torch.abs(
            torch.fft.fftshift(
                torch.fft.ifft2(self._object_spectrum_t * otf)))

        self._science_frame_t = image
        return image  # returns GPU tensor

    # ================================================================
    # Optomechanical interaction
    # ================================================================

    def _optomechanical_interaction(self, tension_forces):
        """Map tensioner forces to PTT displacements via placeholder MLP."""
        tension_forces = np.transpose(np.array(tension_forces))
        optomech_embedding = tension_forces.dot(self._optomech_encoder)
        optomech_ptt = optomech_embedding.dot(self._optomech_decoder)
        optomech_ptt = optomech_ptt.reshape((self.num_apertures, 3))
        optomech_ptt = np.zeros((self.num_apertures, 3))
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
        """Apply initial PTT perturbations at episode start."""
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

            # [DEBUG] Wavelength-aware piston initialization:
            # Mean at λ/4 physical (= λ/2 OPD, maximally destructive),
            # std at λ/6 physical.  Random ±sign per segment ensures
            # relative errors between segments, not a global offset.
            _wl = self.wavelength
            _piston_mean = _wl / 4.0     # λ/4 physical
            _piston_std = _wl / 6.0      # λ/6 physical
            _signs = np.random.choice([-1.0, 1.0], size=self.num_apertures)
            wind_ptt = np.zeros((self.num_apertures, 3))
            wind_ptt[:, 0] = _signs * (
                _piston_mean + np.random.randn(self.num_apertures) * _piston_std)
            clip_m = cfg["init_wind_piston_clip_m"]
            wind_ptt[:, 0] = np.clip(wind_ptt[:, 0], -clip_m, clip_m)
            # Tip/tilt initialization: random ±sign × (mean + randn × std)
            # so the error is always away from zero (same pattern as piston).
            # Mean = std (1-sigma offset), spread = std/3.
            _tip_signs = np.random.choice([-1.0, 1.0], size=self.num_apertures)
            _tilt_signs = np.random.choice([-1.0, 1.0], size=self.num_apertures)
            wind_ptt[:, 1] = _tip_signs * (
                tip_std + np.random.randn(self.num_apertures) * tip_std / 3.0
            ) * np.pi / (180 * 3600)
            wind_ptt[:, 2] = _tilt_signs * (
                tilt_std + np.random.randn(self.num_apertures) * tilt_std / 3.0
            ) * np.pi / (180 * 3600)
            self._apply_ptt_displacements(wind_ptt)

        if self.model_temp_diff_motion:
            temp_ptt = np.random.randn(self.num_apertures, 3)
            temp_ptt[:, 0] *= 0.0
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
        """Apply piston/tip/tilt displacements to the segmented mirror."""
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

        # V4: sync HCIPy actuators → GPU tensor
        if hasattr(self, '_actuators_t'):
            self._actuators_t = torch.tensor(
                np.array(self.segmented_mirror.actuators),
                dtype=torch.float32, device=self._torch_device)

    # ================================================================
    # Incremental bootstrapping
    # ================================================================

    def _bootstrap_off_axis_angle(self):
        """Tip/tilt angle (radians) that moves a segment's light fully
        off the focal plane.  Uses 1.5× the half-extent for margin."""
        half_extent = self._focal_plane_image_size_meters / (2.0 * self._focal_length)
        return half_extent * 1.5

    def _bootstrap_segment_angles(self):
        """Return angular position (radians) of each segment on the ring."""
        return np.array([2.0 * np.pi * i / self.num_apertures
                         for i in range(self.num_apertures)])

    def _init_bootstrap_segments(self, phased_count, noisy=False):
        """Set segment states for incremental bootstrapping.

        - Segments 0..phased_count-1: left at zero PTT (co-phased).
        - Segment phased_count (the target): left at zero PTT here;
          perturbation is applied separately by the caller.
        - Segments phased_count+1..N-1: tipped/tilted to push their
          light fully off the focal plane.

        Args:
            phased_count: number of already co-phased segments.
            noisy: if True, add ±10% uniform noise to the off-axis
                   offset (for episode variation).  Use False when
                   computing reference images.
        """
        off_axis = self._bootstrap_off_axis_angle()
        seg_angles = self._bootstrap_segment_angles()

        # Off-axis push applies to the target segment AND every later
        # segment (target..n_seg-1). Only the already-co-phased segments
        # (0..target-1) stay at zero PTT.
        ptt = np.zeros((self.num_apertures, 3))
        for seg_id in range(phased_count, self.num_apertures):
            scale = 1.0
            if noisy:
                scale += np.random.uniform(-0.1, 0.1)
            theta = seg_angles[seg_id]
            ptt[seg_id, 1] = off_axis * np.sin(theta) * scale   # tip
            ptt[seg_id, 2] = off_axis * np.cos(theta) * scale   # tilt

        self._apply_ptt_displacements(ptt)

    def _init_bootstrap_target_perturbation(self, phased_count):
        """Apply an additional random piston kick to the target segment.

        The target's tip/tilt is already set by ``_init_bootstrap_segments``
        (off-axis push); this function only adds a piston disturbance
        drawn from the wind model so the policy has a non-trivial piston
        to resolve alongside the off-axis tip/tilt. Tip/tilt are
        intentionally NOT touched here so the off-axis push survives.
        """
        cfg = self._cfg
        _wl = self.wavelength
        _piston_mean = _wl / 4.0
        _piston_std = _wl / 6.0

        ptt = np.zeros((self.num_apertures, 3))
        seg = phased_count
        sign = np.random.choice([-1.0, 1.0])
        ptt[seg, 0] = sign * (_piston_mean + np.random.randn() * _piston_std)
        clip_m = cfg["init_wind_piston_clip_m"]
        ptt[seg, 0] = np.clip(ptt[seg, 0], -clip_m, clip_m)

        self._apply_ptt_displacements(ptt, incremental=True)

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

        # V4: sync HCIPy actuators → GPU tensor
        if hasattr(self, '_actuators_t'):
            self._actuators_t = torch.tensor(
                np.array(self.segmented_mirror.actuators),
                dtype=torch.float32, device=self._torch_device)

    # ================================================================
    # Tensioner & secondary commands
    # ================================================================

    def command_tensioners(self, tensioner_commands):
        """Apply tensioner commands via the optomechanical interaction."""
        self._optomechanical_interaction(tensioner_commands)

    def command_secondaries(self, secondaries_commands):
        """Command secondary mirror PTT for all segments.

        [v3-opt] Uses pre-computed correction limits from __init__.
        """
        max_p_m = self._max_p_m
        max_t_r = self._max_t_r
        max_tl_r = self._max_tl_r

        self.max_piston_correction = max_p_m
        self.max_tip_correction = max_t_r
        self.max_tilt_correction = max_tl_r

        # Local aliases for inner loop
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

                inc_p = seg_piston_cmd[0]
                dec_p = seg_piston_cmd[1]
                inc_t = seg_tip_cmd[0]
                dec_t = seg_tip_cmd[1]
                inc_tl = seg_tilt_cmd[0]
                dec_tl = seg_tilt_cmd[1]

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
                _n_dof = 3 if self.command_tip_tilt else 1
                self._clipped_dof_count += int(np.sum(pre_clip[:_n_dof] != post_clip[:_n_dof]))
                self._total_dof_count += _n_dof

                # Rail noise: when clipped, replace with stochastic state
                # around the rail so the agent can't exploit hard limits.
                if pre_clip[0] != post_clip[0]:
                    piston_state += np.random.normal(0.0, _noise_frac * 2.0 * max_p_m)
                if _n_dof > 1:
                    if pre_clip[1] != post_clip[1]:
                        tip_state += np.random.normal(0.0, _noise_frac * 2.0 * max_t_r)
                    if pre_clip[2] != post_clip[2]:
                        tilt_state += np.random.normal(0.0, _noise_frac * 2.0 * max_tl_r)

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
                _n_dof = 3 if self.command_tip_tilt else 1
                self._clipped_dof_count += int(np.sum(pre_clip[:_n_dof] != post_clip[:_n_dof]))
                self._total_dof_count += _n_dof

                # Rail noise: when clipped, replace with stochastic state
                # around the rail so the agent can't exploit hard limits.
                if pre_clip[0] != post_clip[0]:
                    piston_state += np.random.normal(0.0, _noise_frac * 2.0 * max_p_m)
                if _n_dof > 1:
                    if pre_clip[1] != post_clip[1]:
                        tip_state += np.random.normal(0.0, _noise_frac * 2.0 * max_t_r)
                    if pre_clip[2] != post_clip[2]:
                        tilt_state += np.random.normal(0.0, _noise_frac * 2.0 * max_tl_r)

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
                _n_dof = 3 if self.command_tip_tilt else 1
                self._clipped_dof_count += int(np.sum(pre_clip[:_n_dof] != post_clip[:_n_dof]))
                self._total_dof_count += _n_dof

                # Rail noise: when clipped, replace with stochastic state
                # around the rail so the agent can't exploit hard limits.
                if pre_clip[0] != post_clip[0]:
                    piston_state += np.random.normal(0.0, _noise_frac * 2.0 * max_p_m)
                if _n_dof > 1:
                    if pre_clip[1] != post_clip[1]:
                        tip_state += np.random.normal(0.0, _noise_frac * 2.0 * max_t_r)
                    if pre_clip[2] != post_clip[2]:
                        tilt_state += np.random.normal(0.0, _noise_frac * 2.0 * max_tl_r)

            # Actuator repeatability noise: Gaussian perturbation
            # proportional to correction range (models real hardware).
            # Only applied when the actuator actually moved (non-zero cmd);
            # a dead-zoned (zero) command means the actuator is idle.
            # No re-clip: tiny noise (1e-4 fraction) may push slightly
            # past rails, which is physically realistic and prevents the
            # agent from exploiting hard limits for state certainty.
            if _noisy:
                if seg_piston_cmd != 0.0:
                    piston_state += np.random.normal(0.0, _noise_frac * 2.0 * max_p_m)
                if seg_tip_cmd != 0.0:
                    tip_state += np.random.normal(0.0, _noise_frac * 2.0 * max_t_r)
                if seg_tilt_cmd != 0.0:
                    tilt_state += np.random.normal(0.0, _noise_frac * 2.0 * max_tl_r)

            _seg_mirror.set_segment_actuators(
                seg_id, piston_state, tip_state, tilt_state)

        # V4: sync HCIPy actuators → GPU tensor
        self._actuators_t = torch.tensor(
            np.array(self.segmented_mirror.actuators),
            dtype=torch.float32, device=self._torch_device)


# ===================================================================
# OptomechEnv -- Gymnasium environment
# ===================================================================

class OptomechEnv(gym.Env):
    """Gymnasium environment for distributed-aperture telescope control.

    High-performance v3: functionally identical to v2 with optimised
    hot paths.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50,
    }

    def __init__(self, **kwargs):

        self.cfg = {**DEFAULT_CONFIG, **kwargs}
        cfg = self.cfg

        _print_config_banner(cfg)

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

        # --- Actuator noise (stochastic repeatability) ---------------
        self._actuator_noise = cfg['actuator_noise']
        self._actuator_noise_fraction = cfg['actuator_noise_fraction']
        self._action_scale = cfg.get('env_action_scale', 1.0)

        # --- Action penalty ------------------------------------------
        self._action_penalty = cfg['action_penalty']
        self._action_penalty_weight = cfg['action_penalty_weight']
        self._oob_penalty = cfg['oob_penalty']
        self._oob_penalty_weight = cfg['oob_penalty_weight']
        self._holding_bonus_weight = cfg.get('holding_bonus_weight', 0.0)
        self._holding_bonus_min_reward = cfg.get('holding_bonus_min_reward', -1.0)
        self._holding_bonus_threshold = cfg.get('holding_bonus_threshold', 0.0)
        self._minimum_absolute_action = cfg.get('minimum_absolute_action', 0.0)
        self._reward_scale = cfg.get('reward_scale', 1.0)

        # --- Per-DOF action penalty weights (bootstrap) -----------------
        # When bootstrap_phase is True, non-target segments get a heavier
        # action penalty to discourage the policy from moving them.
        # _action_penalty_weights is a 1-D array matching the flat action
        # vector, or None when uniform penalty applies (non-bootstrap).
        self._action_penalty_weights = None  # built in _build_bootstrap_penalty

        self.reward_function = cfg['reward_function']

        # --- Composite reward weights --------------------------------
        self._reward_weight_strehl = cfg['reward_weight_strehl']
        self._reward_weight_dark_hole = cfg['reward_weight_dark_hole']
        self._reward_weight_image_quality = cfg['reward_weight_image_quality']
        self._reward_weight_centering = cfg['reward_weight_centering']
        self._reward_weight_flux = cfg.get('reward_weight_flux', 0.0)
        self._reward_weight_convex_flux = cfg.get('reward_weight_convex_flux', 0.0)
        self._convex_flux_power = cfg.get('convex_flux_power', 2.0)
        self._reward_weight_dist = cfg.get('reward_weight_dist', 0.0)
        self._reward_weight_concentration = cfg.get('reward_weight_concentration', 0.0)
        self._reward_weight_peak = cfg.get('reward_weight_peak', 0.0)
        self._reward_weight_centered_strehl = cfg.get('reward_weight_centered_strehl', 0.0)
        self._reward_weight_shape = cfg['reward_weight_shape']

        self.ao_loop_active = cfg['ao_loop_active']
        self.observation_mode = cfg['observation_mode']

        # [v3-opt] Cache observation mode as booleans
        self._obs_image_only = (self.observation_mode == "image_only")
        self._obs_image_action = (self.observation_mode == "image_action")

        # --- AO model flag -------------------------------------------
        if self.command_dm or self.ao_loop_active:
            cfg['model_ao'] = True
        else:
            cfg['model_ao'] = False

        # --- DM physics (also stored at env level for AO loop) -------
        self.microns_opd_per_actuator_bit = cfg['microns_opd_per_actuator_bit']
        self.stroke_count_limit = cfg['stroke_count_limit']
        self.dm_gain = cfg['dm_gain']
        self.dm_leakage = cfg['dm_leakage']

        # [v3-opt] Pre-compute stroke limit for AO loop
        self._dm_stroke_limit_m = (
            self.microns_opd_per_actuator_bit
            * self.stroke_count_limit * 1e-6 / 2.0)

        # [v3-opt] Pre-compute detector constants (h*c / wavelength, frame_sec)
        _h = 6.62607015e-34
        _c = 2.99792458e8
        self._photon_energy = _h * _c / cfg['wavelength']
        self._det_qe = cfg['detector_quantum_efficiency']
        self._det_gain = cfg['detector_gain_e_per_dn']
        self._det_max_dn = cfg['detector_max_dn']
        self._det_poisson = cfg['detector_poisson_noise']
        self._det_quantize = cfg['detector_quantize_adc']
        self._frame_sec = self.frame_interval_ms / 1000.0

        _v3_section("Initializing OptomechEnv")

        # --- Internal state ------------------------------------------
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self._init_state_storage()

        # --- Compute timing ratios -----------------------------------
        self._compute_timing_ratios()

        # --- Print dark hole settings --------------------------------
        if cfg.get('dark_hole', False):
            _v3_subsection("Dark Hole")
            _v3_kv("angular location (deg)", cfg.get('dark_hole_angular_location_degrees', 'N/A'))
            _v3_kv("location radius frac",   cfg.get('dark_hole_location_radius_fraction', 'N/A'))
            _v3_kv("size radius",            cfg.get('dark_hole_size_radius', 'N/A'))
        else:
            _v3("Dark hole: disabled")

        _v3_subsection("Actuator Noise")
        if self._actuator_noise:
            _v3_kv("enabled", "True")
            _v3_kv("noise fraction", "%.2e" % self._actuator_noise_fraction)
        else:
            _v3("Actuator noise: disabled")

        _v3_subsection("Action Penalty")
        if self._action_penalty:
            _v3_kv("enabled", "True")
            _v3_kv("weight", "%.4f" % self._action_penalty_weight)
        else:
            _v3("Action penalty: disabled")

        if self._minimum_absolute_action > 0.0:
            _v3_kv("action deadzone", "%.4f" % self._minimum_absolute_action)
        if self._action_scale != 1.0:
            _v3_kv("env action scale", "%.4f" % self._action_scale)

        _v3_subsection("Holding Bonus")
        if self._holding_bonus_weight > 0:
            _v3_kv("enabled", "True")
            _v3_kv("weight", "%.4f" % self._holding_bonus_weight)
            _v3_kv("min_reward", "%.4f" % self._holding_bonus_min_reward)
            _v3_kv("threshold", "%.4f" % self._holding_bonus_threshold)
        else:
            _v3("Holding bonus: disabled")

        _v3_subsection("OOB Penalty")
        if self._oob_penalty:
            _v3_kv("enabled", "True")
            _v3_kv("weight", "%.4f" % self._oob_penalty_weight)
        else:
            _v3("OOB penalty: disabled")

        _v3_subsection("Reward")
        _v3_kv("function", self.reward_function)
        _v3_kv("reward_scale", "%.1f" % self._reward_scale)
        if self.reward_function == "composite":
            _v3_kv("weight_strehl", "%.3f" % self._reward_weight_strehl)
            _v3_kv("weight_centering", "%.3f" % self._reward_weight_centering)
            _v3_kv("weight_dark_hole", "%.3f" % self._reward_weight_dark_hole)
            _v3_kv("weight_image_quality", "%.3f" % self._reward_weight_image_quality)
        if self.reward_function == "factored":
            _v3_kv("weight_shape", "%.3f" % self._reward_weight_shape)
            _v3_kv("weight_dark_hole", "%.3f" % self._reward_weight_dark_hole)
            _v3_kv("weight_strehl", "%.3f" % self._reward_weight_strehl)
            _v3_kv("weight_centering", "%.3f" % self._reward_weight_centering)
            _v3_kv("weight_flux", "%.3f" % self._reward_weight_flux)
            _v3_kv("weight_convex_flux", "%.3f" % self._reward_weight_convex_flux)
            if self._reward_weight_convex_flux > 0:
                _v3_kv("convex_flux_power", "%.1f" % self._convex_flux_power)
            _v3_kv("weight_dist", "%.3f" % self._reward_weight_dist)
            _v3_kv("weight_concentration", "%.3f" % self._reward_weight_concentration)
            _v3_kv("weight_peak", "%.3f" % self._reward_weight_peak)
            _v3_kv("weight_centered_strehl", "%.3f" % self._reward_weight_centered_strehl)

        # --- Build optical system ------------------------------------
        self._build_optical_system()

        # --- Episode clock -------------------------------------------
        self.episode_time_ms = 0.0

        # --- Build action spaces -------------------------------------
        self._build_action_spaces()
        self._build_bootstrap_penalty()

        # --- Build observation space ---------------------------------
        self._build_observation_space()

        # --- Cache wavelengths (constant for env lifetime) -----------
        # [v3-opt] Compute once instead of every AO step
        self._cached_wavelengths = self._compute_wavelengths()

        # --- Cache normalized perfect image for reward functions -----
        # [v3-opt] Computed once, reused across all reward calls
        self._cache_reward_images()

        # --- Pre-compute center masks for align/dark_hole rewards ----
        # [v3-opt] Vectorized with np.ogrid instead of Python double-loop
        self._cache_center_masks()

        # [v3-opt] Reward dispatch dict instead of if/elif chain
        self._reward_dispatch = {
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
            "composite": self._reward_composite,
            "factored": self._reward_factored,
        }
        if self.reward_function not in self._reward_dispatch:
            raise ValueError("Unknown reward_function: '%s'" % self.reward_function)
        self._reward_fn = self._reward_dispatch[self.reward_function]

        # --- Environment-level state storage -------------------------
        self.state_content["wavelength"] = self.optical_system.wavelength

        _v3_section("Environment Ready")

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
        # power values are enormous.  Since MSE and Strehl now normalize by
        # each image's own max, we just need the correct PSF *shape*.
        _pi_2d = np.array(pi).reshape(self.image_shape)
        self._perfect_image_dn = _pi_2d
        self._perfect_image_max_dn = float(np.max(self._perfect_image_dn))
        # Peak-concentration of perfect PSF (unit-independent Strehl reference).
        self._perfect_peak_over_sum = self._perfect_image_max_dn / float(np.sum(self._perfect_image_dn))
        _v3_kv("perfect_image_max_dn", "%.4f" % self._perfect_image_max_dn)

    def _cache_center_masks(self):
        """Pre-compute circular center masks for align/dark_hole rewards."""
        cfg = self.cfg
        num_px = cfg['focal_plane_image_size_pixels']
        ctr = num_px // 2

        # Standard radius mask
        radius = cfg['align_radius']
        y, x = np.ogrid[:num_px, :num_px]
        dist_sq = (y - ctr) ** 2 + (x - ctr) ** 2
        self._center_mask_standard = dist_sq <= radius ** 2

        # Expanded radius mask
        radius_max = cfg['align_radius_max_expand']
        self._center_mask_expanded = dist_sq <= radius_max ** 2

        # Centering weight map for centering reward.
        centering_mode = cfg.get('centering_mode', 'gaussian')
        if centering_mode == 'circular':
            # Flat top-hat: 1 inside circle, 0 outside.
            # Radius is a fraction of the image size.
            r_px = cfg.get('centering_radius_fraction', 0.25) * num_px
            self._centering_weight = (dist_sq <= r_px ** 2).astype(np.float32)
        else:
            # Soft Gaussian: sigma is a fraction of the frame size so
            # the weight falls off smoothly from center to edge.
            sigma_px = cfg['centering_sigma_fraction'] * num_px
            self._centering_weight = np.exp(
                -dist_sq.astype(np.float64) / (2.0 * sigma_px ** 2)
            ).astype(np.float32)

        # L2 distance map for distance-penalty reward.
        # Normalized to [0, 1] by dividing by the max possible distance
        # (corner to center).
        dist = np.sqrt(dist_sq.astype(np.float64))
        max_dist = np.sqrt(2.0) * ctr  # corner distance
        self._dist_weight = (dist / max_dist).astype(np.float32)

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
        """Derive the multi-rate control hierarchy from intervals."""
        cfg = self.cfg
        _v3_subsection("Timing Hierarchy")
        _v3_kv("control  interval", "%.2f ms" % self.control_interval_ms)
        _v3_kv("frame    interval", "%.2f ms" % self.frame_interval_ms)
        _v3_kv("decision interval", "%.2f ms" % self.decision_interval_ms)
        _v3_kv("AO       interval", "%.2f ms" % self.ao_interval_ms)

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

        _v3_subsection("Derived Rates")
        _v3_kv("commands / decision", "%d" % self.commands_per_decision)
        _v3_kv("commands / frame",    "%d" % self.commands_per_frame)
        _v3_kv("AO steps / frame",    "%d" % self.ao_steps_per_frame)
        _v3_kv("frames / decision",   "%d" % self.frames_per_decision)

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
    # Action spaces  (anytree-free)
    # ================================================================

    def _build_action_spaces(self):
        """Construct the hierarchical Tuple action space, then flatten.

        [v3-opt] Replaces anytree with a simple list of index tuples.
        """
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

        # [v3-opt] Build list-based index mapper instead of anytree
        self._linear_to_tree_indices = self._build_index_map(
            self.dict_action_space)

        # Flatten to a simple Box or MultiDiscrete.
        if self.discrete_control:
            flat = spaces.MultiDiscrete(
                [1] * len(self._linear_to_tree_indices))
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

    def _build_bootstrap_penalty(self):
        """Build per-DOF action penalty weights for bootstrap training.

        In bootstrap mode, the target segment (index == phased_count) gets
        the base action_penalty_weight, while all other segments get
        base_weight * bootstrap_nontarget_penalty_multiplier.

        The flat action vector is structured as commands_per_decision
        repetitions of the per-command vector.  Within each command, the
        secondaries DOFs are interleaved per segment:
          [p0, tip0, tilt0, p1, tip1, tilt1, ..., pN, tipN, tiltN]

        When bootstrap_phase is False, self._action_penalty_weights stays
        None and the reward code falls back to the uniform scalar weight.
        """
        cfg = self.cfg
        if not cfg.get('bootstrap_phase', False):
            self._action_penalty_weights = None
            return

        n_seg = self.optical_system.num_apertures
        target_seg = cfg.get('bootstrap_phased_count', 0)
        multiplier = cfg.get('bootstrap_nontarget_penalty_multiplier', 10.0)
        base_w = self._action_penalty_weight

        action_dim = self.action_space.shape[0]
        cpd = self.commands_per_decision

        # DOFs per segment within a single command step
        dofs_per_seg = 1  # piston
        if self.command_tip_tilt:
            dofs_per_seg = 3  # piston + tip + tilt

        # Build per-command weight vector
        dofs_per_cmd = n_seg * dofs_per_seg
        cmd_weights = np.ones(dofs_per_cmd, dtype=np.float32) * base_w * multiplier

        # Set target segment DOFs to base weight
        if target_seg < n_seg:
            start = target_seg * dofs_per_seg
            end = start + dofs_per_seg
            cmd_weights[start:end] = base_w

        # Tile across all command steps
        weights = np.tile(cmd_weights, cpd)

        # If there are extra DOFs (tensioners, DM) beyond secondaries,
        # pad with base weight so they aren't over-penalized
        if len(weights) < action_dim:
            extra = np.ones(action_dim - len(weights), dtype=np.float32) * base_w
            weights = np.concatenate([weights, extra])

        assert len(weights) == action_dim, (
            f"Bootstrap penalty weights length {len(weights)} != action_dim {action_dim}")

        self._action_penalty_weights = weights

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

        if self._obs_image_only:
            self.observation_space = self.image_space
        elif self._obs_image_action:
            self.observation_space = spaces.Dict(
                {"image": self.image_space,
                 "prior_action": self.action_space},
                seed=42)
        else:
            raise ValueError(
                "Invalid observation_mode: '%s'. Use 'image_only' or "
                "'image_action'." % self.observation_mode)

    # ================================================================
    # Action space helpers  (anytree-free)
    # ================================================================

    def _flatten(self, dict_space, flat_space_high=1.0, flat_space_low=0.0):
        """Flatten a hierarchical Tuple space to a Box."""
        return spaces.Box(
            low=flat_space_low, high=flat_space_high,
            shape=(len(self._build_index_map(dict_space)),),
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

    @staticmethod
    def _build_index_map(action_space):
        """Build a list of index tuples mapping linear address -> tree path.

        [v3-opt] Replaces anytree entirely. Each entry is a tuple of ints
        that indexes into the nested list structure.
        """
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
        """Map a flat action vector into the hierarchical Tuple structure.

        [v3-opt] Uses pre-computed index tuples instead of anytree + string ops.
        """
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
        """Return the total number of leaf nodes (flat action size)."""
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
        _v3_section("Episode Reset")
        _v3("Rebuilding optical system...")
        self._build_optical_system()

        # Refresh caches that depend on the optical system
        self._cached_wavelengths = self._compute_wavelengths()
        self._cache_reward_images()
        self._cache_center_masks()

        if self.command_dm or self.ao_loop_active:
            self.optical_system.calibrate_dm_interaction_matrix(self.uuid)
            rcond = self.cfg['dm_interaction_rcond']
            self.reconstruction_matrix = hcipy.inverse_tikhonov(
                self.optical_system.interaction_matrix.transformation_matrix,
                rcond=rcond)
            self.episode_time_ms = 0.0

        _v3("Seeding initial action...")

        _bootstrap = self.cfg.get('bootstrap_phase', False)
        _phased_count = self.cfg.get('bootstrap_phased_count', 0)

        if _bootstrap:
            # Re-push excluded segments with per-episode noise
            # (deterministic push already happened in OpticalSystem init
            # for the reference images; now add noise for training variety).
            self.optical_system._init_bootstrap_segments(
                _phased_count, noisy=True)

            # Perturb the target segment only (index = phased_count)
            if self.cfg['init_differential_motion']:
                _v3("Bootstrap: perturbing target segment %d..." % _phased_count)
                self.optical_system._init_bootstrap_target_perturbation(
                    _phased_count)
        else:
            if self.cfg['init_differential_motion']:
                _v3("Applying initial differential motion...")
                self.optical_system._init_natural_diff_motion()

        self.optical_system._store_baseline_segment_displacements()

        # Warm-up: run frames to generate the initial observation.
        # In bootstrap mode, use zero action so excluded segments stay
        # at their off-axis positions (random actions could overwrite them).
        _warmup_action = (np.zeros_like(self.action_space.sample())
                          if _bootstrap else self.action_space.sample())
        _v3("Warm-up: %d initial frame(s)..." % self.frames_per_decision)
        for _ in range(self.frames_per_decision):
            (initial_state, _, _, _, info) = self.step(
                action=_warmup_action,
                noisy_command=False,
                reset=True)

        self.state = initial_state
        self.steps_beyond_done = None
        _v3_section("Reset Complete")
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
            _v3_timer("State deepcopy", time.time() - t0)

    # ================================================================
    # Step
    # ================================================================

    def step(self, action, noisy_command=False, reset=False):
        """Execute one decision interval of the environment.

        [v3-opt] Uses local aliases to minimize attribute lookups in
        the hot inner loops.
        """
        # [v3-opt] Local aliases for hot-path attributes
        _report = self.report_time
        _frames_per_decision = self.frames_per_decision
        _cmds_per_frame = self.commands_per_frame
        _ao_steps_per_cmd = self.ao_steps_per_command
        _wavelengths = self._cached_wavelengths
        _ao_active = self.ao_loop_active
        _record = self.record_env_state_info
        _image_shape = self.image_shape
        _control_interval = self.control_interval_ms
        _ao_steps_per_frame = self.ao_steps_per_frame
        _frame_sec = self._frame_sec
        _optical_sys = self.optical_system

        if _report:
            step_t0 = time.time()

        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))

        # Work on a copy so caller's array is not mutated.
        action = action.copy()

        # Deadzone: zero out action dimensions below threshold.
        # Applied in raw [-1, 1] policy space so the threshold is
        # meaningful regardless of action_scale.
        _min_abs = self._minimum_absolute_action
        if _min_abs > 0.0:
            action[np.abs(action) < _min_abs] = 0.0

        # Stash the raw (pre-scale) action for the L1 penalty later.
        self._raw_action = action

        # Scale to physical command range.
        action = action * self._action_scale

        if _record:
            for key in self.state_content:
                if isinstance(self.state_content[key], list):
                    self.state_content[key] = []
            self.state_content["shwfs_slopes"] = []

        self.action = self._flat_to_dict(action, self.dict_action_space)

        self.focal_plane_images = []
        self.shwfs_slopes_list = []

        self.optical_system._clipped_dof_count = 0
        self.optical_system._total_dof_count = 0

        # [v3-opt] Pre-compute integration time (constant across step)
        integration_sec = _frame_sec / _ao_steps_per_frame

        # --- Main simulation loop (V4: GPU tensors) -------------------
        _dev = _optical_sys._torch_device
        _n_wl_inv = 1.0 / float(len(_wavelengths))

        for frame_num in range(_frames_per_decision):
            frame_t = torch.zeros(_image_shape, dtype=torch.float32,
                                  device=_dev)

            for command_num in range(_cmds_per_frame):
                self.episode_time_ms += _control_interval

                t0 = time.time()
                _optical_sys.evolve_atmosphere_to(self.episode_time_ms)
                if _report:
                    _v3_timer("Atmosphere evolve", time.time() - t0)

                command = self.action[command_num]
                self._apply_commands(command)

                for ao_step in range(_ao_steps_per_cmd):
                    for wl in _wavelengths:
                        if _report:
                            sim_t0 = time.time()

                        _optical_sys.simulate(wl)
                        if _report:
                            _v3_timer("Optical sim", time.time() - sim_t0, indent=2)

                        if _report:
                            ao_t0 = time.time()
                        if _ao_active and not reset:
                            self._run_ao_correction(integration_sec)
                        if _report:
                            _v3_timer("AO correction", time.time() - ao_t0, indent=2)

                        science_t = _optical_sys.get_science_frame(
                            integration_seconds=integration_sec)
                        # science_t is a GPU tensor (num_px, num_px)
                        frame_t += science_t * _n_wl_inv

                    if _record and not reset:
                        self.save_state()

            frame_t = self._apply_detector_model_gpu(frame_t, _dev)
            self.focal_plane_images.append(frame_t.cpu().numpy())

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

        # Holding bonus: reward stillness proportional to state quality.
        # quality ∈ [0, 1]: linearly maps reward from [min_reward, 0] → [0, 1].
        # stillness ∈ [0, 1] (1 = no movement, 0 = full action).
        # Bonus = weight × quality × stillness.
        #
        # Configurable parameters:
        #   min_reward (default -1.0): reward value that maps to quality=0.
        #     Determines the denominator for normalisation. With a single
        #     reward factor in [-1, 0], use -1.0; adjust if the raw reward
        #     has a different range.
        #   threshold (default 0.0): raw reward below this → no bonus.
        #     With threshold=0.0: bonus ∝ quality (original behavior).
        #     With threshold=-0.8: only rewards > -0.8 get bonuses,
        #       ramping linearly from 0 at -0.8 to full at 0.
        if self._holding_bonus_weight > 0:
            min_r = self._holding_bonus_min_reward  # e.g. -1.0
            thresh = self._holding_bonus_threshold   # e.g. 0.0 or -0.8
            # Map reward from [min_r, 0] → [0, 1]
            span = -min_r if min_r != 0 else 1.0
            raw_quality = float(np.clip((reward - min_r) / span, 0.0, 1.0))
            # Apply threshold: remap [thresh, 0] → [0, 1] in reward-space
            if thresh < 0:
                # Convert threshold to quality-space: (thresh - min_r) / span
                q_thresh = max((thresh - min_r) / span, 0.0)
                if raw_quality > q_thresh:
                    quality = (raw_quality - q_thresh) / (1.0 - q_thresh)
                else:
                    quality = 0.0
            else:
                quality = raw_quality
            action_l1_mean = float(np.mean(np.abs(self._raw_action)))
            stillness = 1.0 - action_l1_mean
            reward = reward + self._holding_bonus_weight * quality * stillness

        # Absolute L1 action penalty on raw (pre-scale) action.
        # Computed in [-1, 1] policy space so the penalty is independent
        # of action_scale.  L1 encourages exact-zero outputs (sparsity),
        # pairing with the deadzone to produce clean "hold" behavior.
        if self._action_penalty:
            abs_action = np.abs(self._raw_action)
            if self._action_penalty_weights is not None:
                # Per-DOF weighted penalty (bootstrap mode)
                action_penalty = float(np.mean(self._action_penalty_weights * abs_action))
                reward = reward - action_penalty
            else:
                # Uniform scalar penalty (standard mode)
                action_l1_mean = float(np.mean(abs_action))
                reward = reward - self._action_penalty_weight * action_l1_mean

        # Out-of-bounds penalty: penalize actuator clipping at physical rails.
        # Fraction of DOFs that were clipped, times penalty weight.
        oob_frac = 0.0
        if self._oob_penalty and self.optical_system._total_dof_count > 0:
            oob_frac = float(self.optical_system._clipped_dof_count) / float(self.optical_system._total_dof_count)
            reward = reward - self._oob_penalty_weight * oob_frac

        terminated = False
        truncated = False

        # --- Diagnostic metrics (always reported) --------------------
        strehls = [self._absolute_strehl(fpi)
                   for fpi in self.focal_plane_images]
        _norm_perfect_flat = (self._perfect_image_dn / self._perfect_image_max_dn).flatten()
        def _norm_mse(fpi):
            fpi_max = float(np.max(fpi))
            norm_fpi = fpi / fpi_max if fpi_max > 0 else fpi
            return float(np.mean((norm_fpi.flatten() - _norm_perfect_flat) ** 2))
        mses = [_norm_mse(fpi) for fpi in self.focal_plane_images]

        info = {}
        info["strehl"] = float(np.mean(strehls))
        info["mse"] = float(np.mean(mses))
        info["reward_raw"] = float(reward)
        info["oob_frac"] = oob_frac
        if self._reward_weight_centering > 0:
            cens = [self._centering_energy(fpi)
                    for fpi in self.focal_plane_images]
            info["centering"] = float(np.mean(cens))
        if _record:
            info["state_content"] = self.state_content
            info["state"] = self.state

        if _report:
            _v3_timer("STEP TOTAL", time.time() - step_t0)

        reward = np.float32(reward)
        return self.state, reward, terminated, truncated, info

    # ================================================================
    # Command dispatch
    # ================================================================

    def _apply_commands(self, command):
        """Dispatch the hierarchical command tuple to the optical system.

        Execution order matches v1: tensioners first, then secondaries,
        then dm.  Index into command tuple determined by construction order.
        """
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
    # AO loop
    # ================================================================

    def _run_ao_correction(self, integration_seconds):
        """One iteration of closed-loop Shack-Hartmann AO.

        [v3-opt] Uses pre-computed stroke limit.
        """
        shwfs_vec = self.optical_system.get_shwfs_frame(
            integration_seconds=integration_seconds)
        slopes = self.optical_system.shwfse.estimate([shwfs_vec + 1e-10])
        slopes -= self.optical_system.reference_slopes
        self.shwfs_slopes = slopes.ravel()
        self.shwfs_slopes_list.append(self.shwfs_slopes)

        self.optical_system.dm.actuators = (
            (1 - self.dm_leakage) * self.optical_system.dm.actuators
            - self.dm_gain * self.reconstruction_matrix.dot(self.shwfs_slopes))

        # [v3-opt] Use pre-computed stroke limit
        self.optical_system.dm.actuators = np.clip(
            self.optical_system.dm.actuators,
            -self._dm_stroke_limit_m, self._dm_stroke_limit_m)

    # ================================================================
    # Detector model
    # ================================================================

    def _apply_detector_model(self, frame):
        """Convert power-per-pixel frame to digital numbers (DN).

        [v3-opt] Uses pre-computed photon_energy, qe, gain, max_dn, frame_sec.
        Optionally applies Poisson photon noise and ADC quantization.
        """
        energy_joules = frame * self._frame_sec
        n_photons = energy_joules / self._photon_energy
        if self._det_poisson:
            n_photons = np.random.poisson(np.maximum(n_photons, 0)).astype(np.float32)
        n_electrons = n_photons * self._det_qe
        dn = n_electrons / self._det_gain
        if self._det_quantize:
            dn = np.floor(dn)
        return np.clip(dn, 0, self._det_max_dn)

    def _apply_detector_model_gpu(self, frame_t, device):
        """GPU version of _apply_detector_model.

        Optionally applies Poisson photon noise and ADC quantization.
        """
        os = self.optical_system
        energy = frame_t * self._frame_sec
        n_photons = energy / os._photon_energy
        if self._det_poisson:
            n_photons = torch.poisson(torch.clamp(n_photons, min=0))
        n_electrons = n_photons * os._det_qe
        dn = n_electrons / os._det_gain
        if self._det_quantize:
            dn = torch.floor(dn)
        return torch.clamp(dn, 0, os._det_max_dn)

    # ================================================================
    # Reward computation
    # ================================================================

    def _compute_reward(self):
        """Dispatch to the configured reward function."""
        return self._reward_fn()

    # --- Individual reward methods -----------------------------------

    def _absolute_strehl(self, fpi):
        """Flux-corrected Strehl: raw peak-concentration ratio scaled by
        the fraction of total flux that actually landed on the detector.

        Raw ratio:  (max/sum)_obs  /  (max/sum)_perfect
        Flux frac:  sum(obs)       /  sum(reference)

        where *reference* is the total DN measured with perfectly aligned
        segments (computed once per episode by _compute_reference_flux).

        When all light is captured, flux_frac ≈ 1 and the result equals
        the raw ratio — no change from the original metric.  When
        tip/tilt pushes the PSF off the detector, sum(obs) drops, the
        flux fraction falls below 1, and the Strehl is scaled down
        instead of being artificially inflated.
        """
        fpi_sum = float(np.sum(fpi))
        if fpi_sum <= 0:
            return 0.0
        obs_peak_over_sum = float(np.max(fpi)) / fpi_sum
        raw_strehl = obs_peak_over_sum / self._perfect_peak_over_sum

        # Flux fraction: 1.0 when all light is on detector, < 1 when
        # tip/tilt has displaced the PSF off the focal plane.
        ref_sum = self.optical_system._reference_fpi_sum
        if ref_sum > 0:
            flux_frac = min(fpi_sum / ref_sum, 1.0)
        else:
            flux_frac = 1.0

        return raw_strehl * flux_frac

    def _centering_energy(self, fpi):
        """Encircled energy fraction within the centering weight map.

        Measures what fraction of the *reference* weighted flux is present
        in the current observation.  Two modes:

        - ``centering_mode='gaussian'``: Gaussian weight map with sigma set
          by ``centering_sigma_fraction`` × frame size.
        - ``centering_mode='circular'``: flat top-hat mask with radius set
          by ``centering_radius_fraction`` × frame size.  Rewards both
          keeping flux on the detector and centering it within the circle.

        Returns a value in [0, 1]:
          ≈ 1  when the PSF is centred and all flux is on the detector
          ≈ 0  when the PSF is completely off the detector or outside circle
        """
        ref_csum = self.optical_system._reference_centering_sum
        if ref_csum <= 0:
            return 0.0
        weighted_sum = float(np.sum(fpi * self._centering_weight))
        return min(weighted_sum / ref_csum, 1.0)

    def _reward_composite(self):
        """Weighted multi-factor reward: Strehl + centering + dark_hole + image_quality."""
        total = 0.0
        w_s = self._reward_weight_strehl
        w_cen = self._reward_weight_centering
        w_dh = self._reward_weight_dark_hole
        w_iq = self._reward_weight_image_quality
        if w_s > 0:
            strehls = [self._absolute_strehl(fpi) for fpi in self.focal_plane_images]
            total += w_s * np.mean(strehls, dtype=np.float32)
        if w_cen > 0:
            cens = [self._centering_energy(fpi) for fpi in self.focal_plane_images]
            total += w_cen * np.mean(cens, dtype=np.float32)
        if w_dh > 0 and self._target_zero_mask is not None:
            dh_vals = []
            for fpi in self.focal_plane_images:
                dh_intensity = float(np.mean(fpi[self._target_zero_mask]))
                dh_vals.append(-dh_intensity / self._perfect_image_max_dn)
            total += w_dh * np.mean(dh_vals, dtype=np.float32)
        if w_iq > 0:
            norm_perfect_flat = (self._perfect_image_dn / self._perfect_image_max_dn).flatten()
            iq_vals = []
            for fpi in self.focal_plane_images:
                fpi_max = float(np.max(fpi))
                norm_fpi = fpi / fpi_max if fpi_max > 0 else fpi
                mse = -float(np.mean((norm_fpi.flatten() - norm_perfect_flat) ** 2))
                iq_vals.append(mse)
            total += w_iq * np.mean(iq_vals, dtype=np.float32)
        return np.float32(total)

    def _reward_strehl(self):
        """Strehl ratio computed from absolute Strehl."""
        strehls = [self._absolute_strehl(fpi) for fpi in self.focal_plane_images]
        return np.mean(strehls, dtype=np.float32)

    def _reward_negastrehl(self):
        strehls = [self._absolute_strehl(fpi) for fpi in self.focal_plane_images]
        return np.mean(strehls) - 1.0

    def _reward_negaexpstrehl(self):
        strehls = [self._absolute_strehl(fpi) for fpi in self.focal_plane_images]
        return (np.mean(strehls) ** 10) - 1.0

    def _reward_strehl_closed(self):
        strehls = [self._absolute_strehl(fpi) for fpi in self.focal_plane_images]
        return 1.0 if np.mean(strehls) >= 0.8 else 0.0

    def _reward_image_mse(self):
        norm_perfect = self._perfect_image_dn / self._perfect_image_max_dn
        norm_perfect_flat = norm_perfect.flatten()
        mses = []
        for fpi in self.focal_plane_images:
            fpi_max = float(np.max(fpi))
            norm_fpi = fpi / fpi_max if fpi_max > 0 else fpi
            mses.append(float(np.mean((norm_fpi.flatten() - norm_perfect_flat) ** 2)))
        return -np.mean(mses, dtype=np.float32)

    def _reward_factored(self):
        """Multi-factor reward: shape + dark_hole + strehl + centering.

        Each component is negative, approaching 0 as system improves.
        Weights of 0 disable a component entirely.
        """
        total = 0.0

        # --- Shape: log-MSE of normalized images, excluding dark hole ---
        if self._reward_weight_shape > 0:
            norm_perfect = self._perfect_image_dn / self._perfect_image_max_dn
            log_perfect = np.log(norm_perfect + 1e-10)
            shape_vals = []
            for fpi in self.focal_plane_images:
                fpi_max = float(np.max(fpi))
                norm_fpi = fpi / fpi_max if fpi_max > 0 else fpi
                log_fpi = np.log(norm_fpi + 1e-10)
                if self._target_zero_mask is not None:
                    mask = ~self._target_zero_mask
                    mse = float(np.mean((log_fpi[mask] - log_perfect[mask]) ** 2))
                else:
                    mse = float(np.mean((log_fpi - log_perfect) ** 2))
                shape_vals.append(-mse)
            total += self._reward_weight_shape * np.mean(shape_vals, dtype=np.float32)

        # --- Dark hole: mean normalized intensity in zero-target region ---
        if self._reward_weight_dark_hole > 0 and self._target_zero_mask is not None:
            dh_vals = []
            for fpi in self.focal_plane_images:
                fpi_max = float(np.max(fpi))
                norm_fpi = fpi / fpi_max if fpi_max > 0 else fpi
                dh_vals.append(-float(np.mean(norm_fpi[self._target_zero_mask])))
            total += self._reward_weight_dark_hole * np.mean(dh_vals, dtype=np.float32)

        # --- Strehl: -(1 - strehl), zero when perfect ---
        if self._reward_weight_strehl > 0:
            strehls = [self._absolute_strehl(fpi) for fpi in self.focal_plane_images]
            total += self._reward_weight_strehl * (-(1.0 - float(np.mean(strehls))))

        # --- Centering: -(1 - centering), zero when centred ---
        if self._reward_weight_centering > 0:
            cens = [self._centering_energy(fpi) for fpi in self.focal_plane_images]
            total += self._reward_weight_centering * (-(1.0 - float(np.mean(cens))))

        # --- Flux: -(1 - flux_frac), zero when all light on detector ---
        if self._reward_weight_flux > 0:
            ref_sum = self.optical_system._reference_fpi_sum
            if ref_sum > 0:
                flux_fracs = [min(float(np.sum(fpi)) / ref_sum, 1.0)
                              for fpi in self.focal_plane_images]
                total += self._reward_weight_flux * (-(1.0 - float(np.mean(flux_fracs))))

        # --- Convex flux: -(1 - flux_frac^N), zero when all light on detector ---
        # Superlinear in flux so the second mirror is worth much more than
        # the first (e.g. N=2: one mirror ≈ 0.25, both ≈ 1.0).
        if self._reward_weight_convex_flux > 0:
            ref_sum = self.optical_system._reference_fpi_sum
            if ref_sum > 0:
                N = self._convex_flux_power
                conv_vals = [min(float(np.sum(fpi)) / ref_sum, 1.0) ** N
                             for fpi in self.focal_plane_images]
                total += self._reward_weight_convex_flux * (-(1.0 - float(np.mean(conv_vals))))

        # --- Dist: -( dist_score - ref_dist_score ), zero when centred ---
        # dist_score = flux-weighted mean normalised L2 distance to centre.
        # Higher means light is farther from centre.  We subtract the
        # reference (perfect-PSF) baseline so the term is 0 when perfect.
        # When the PSF is off-detector (fpi_sum=0), dist_score is set to 1
        # (maximum penalty).
        if self._reward_weight_dist > 0:
            ref_sum = self.optical_system._reference_fpi_sum
            ref_ds = self.optical_system._reference_dist_score
            dist_scores = []
            for fpi in self.focal_plane_images:
                fpi_sum = float(np.sum(fpi))
                if fpi_sum > 0:
                    ds = float(np.sum(fpi * self._dist_weight)) / fpi_sum
                else:
                    ds = 1.0  # worst case: no light on detector
                dist_scores.append(ds)
            mean_ds = float(np.mean(dist_scores))
            total += self._reward_weight_dist * (-(mean_ds - ref_ds))

        # --- Concentration: -(1 - C/C_ref), zero when perfectly merged ---
        # C = sum(I²)/sum(I)² (inverse participation ratio).
        # Higher when light is concentrated; no spatial bias.
        # Normalised by reference so the term is in [−1, 0].
        if self._reward_weight_concentration > 0:
            ref_c = self.optical_system._reference_concentration
            if ref_c > 0:
                conc_vals = []
                for fpi in self.focal_plane_images:
                    fpi_f64 = fpi.astype(np.float64)
                    fpi_sum = float(np.sum(fpi_f64))
                    if fpi_sum > 0:
                        c = float(np.sum(fpi_f64 ** 2)) / (fpi_sum ** 2)
                    else:
                        c = 0.0
                    conc_vals.append(c)
                mean_c = float(np.mean(conc_vals))
                total += self._reward_weight_concentration * (-(1.0 - min(mean_c / ref_c, 1.0)))

        # --- Centered Strehl: -(1 - S × centering) ---
        # S × centering is only high when both Strehl and centering are good.
        # centering = 1 - dist/max_dist ∈ [0, 1] (1 = centered, 0 = corner).
        # Result is in [-1, 0], zero only when perfectly aligned and centered.
        if self._reward_weight_centered_strehl > 0:
            cs_vals = []
            for fpi in self.focal_plane_images:
                s = self._absolute_strehl(fpi)
                h, w = fpi.shape[-2], fpi.shape[-1]
                cy, cx = h / 2.0, w / 2.0
                max_dist = (cy ** 2 + cx ** 2) ** 0.5
                peak_idx = np.argmax(fpi)
                py, px = np.unravel_index(peak_idx, fpi.shape[-2:])
                dist = ((py - cy) ** 2 + (px - cx) ** 2) ** 0.5
                centering = 1.0 - dist / max_dist
                cs_vals.append(-(1.0 - s * centering))
            total += self._reward_weight_centered_strehl * float(np.mean(cs_vals))

        # --- Peak pixel: -(1 - max(I)/max(I_ref)), zero when merged ---
        # Constructive interference when spots merge ~quadruples peak.
        # No spatial bias — purely rewards the brightest pixel.
        if self._reward_weight_peak > 0:
            ref_max = self.optical_system._reference_fpi_max
            if ref_max > 0:
                peak_vals = [min(float(np.max(fpi)) / ref_max, 1.0)
                             for fpi in self.focal_plane_images]
                total += self._reward_weight_peak * (-(1.0 - float(np.mean(peak_vals))))

        return np.float32(total)

    def _reward_align(self):
        """Log-MSE within a circular centre mask.

        [v3-opt] Uses pre-computed center masks (np.ogrid vectorized).
        Radius set once before loop (v1 behavior preserved).
        """
        perfect_dn = self._perfect_image_dn
        max_dn = self._perfect_image_max_dn
        radius_max = None

        # Radius is evaluated once before the frame loop (v1 behavior).
        if not radius_max:
            center_mask = self._center_mask_standard
        else:
            center_mask = self._center_mask_expanded

        rewards = []
        for fpi in self.focal_plane_images:
            norm_img = fpi / max_dn
            norm_tgt = perfect_dn / max_dn
            loss_img = np.round(65534.0 * norm_img) + 1
            loss_tgt = np.round(65534.0 * norm_tgt) + 1

            mse = np.power(
                np.log(loss_img[center_mask]) - np.log(loss_tgt[center_mask]), 2)
            mse = -np.mean(mse.flatten())

            if mse < self.cfg['align_mse_expand_threshold']:
                radius_max = self.cfg['align_radius_max_expand']

            rewards.append(mse)

        return np.mean(rewards, dtype=np.float32)

    def _reward_dark_hole(self):
        """Log-MSE within centre mask + dark-hole penalty.

        [v3-opt] Uses pre-computed center masks and target zero mask.
        """
        perfect_dn = self._perfect_image_dn
        max_dn = self._perfect_image_max_dn
        target_mask = self._target_zero_mask
        radius_max = None
        alpha_hole = self.cfg['dark_hole_alpha']

        # Radius is evaluated once before the frame loop (v1 behavior).
        if not radius_max:
            center_mask = self._center_mask_standard
        else:
            center_mask = self._center_mask_expanded

        rewards = []
        for fpi in self.focal_plane_images:
            norm_img = fpi / max_dn
            norm_tgt = perfect_dn / max_dn
            loss_img = np.round(65534.0 * norm_img) + 1
            loss_tgt = np.round(65534.0 * norm_tgt) + 1

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
        cfg = self.cfg
        threshold = cfg['ao_closed_inv_slope_threshold']
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
