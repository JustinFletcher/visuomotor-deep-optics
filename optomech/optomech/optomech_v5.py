"""
Optomech V5 -- Batched GPU-accelerated VectorEnv for distributed-aperture
telescope control.

Manages N environment states as batched GPU tensors.  All per-step physics
(MFT propagation, object convolution, detector model, reward) runs as
single batched GPU kernel calls across all N envs.  CPU<->GPU sync
happens exactly once at the end of step().

Architecture:
  - At init: creates ONE V4 env to extract shared constants (aperture,
    influence functions, MFT matrices, object spectrum, detector params,
    reference flux values).  Then discards the V4 env.
  - Per-env state (actuators, baselines, step counts) is batched tensors
    on GPU.
  - step() does everything on GPU, returns numpy at the very end.
  - Implements gymnasium.vector.VectorEnv directly -- no SyncVectorEnv
    or AsyncVectorEnv wrapper needed.

Supported configurations (raises ValueError otherwise):
  - num_atmosphere_layers=0
  - command_secondaries=True, command_tensioners=False, command_dm=False
  - incremental_control=True
  - ao_loop_active=False
  - frames_per_decision=1, commands_per_frame=1, ao_steps_per_command=1

Author: Justin Fletcher (original v1-v4), GPU-batched v5.
"""

import math
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces


def _resolve_device(cfg_device):
    """Resolve 'auto' to the best available torch device."""
    if cfg_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(cfg_device)


def _centered_wavelengths(center_wl, bandwidth_nm, n_samples):
    """Compute centered wavelength samples across a bandwidth."""
    bw_m = bandwidth_nm / 1e9
    if n_samples <= 1:
        return [center_wl]
    bin_width = bw_m / n_samples
    first = center_wl - bw_m / 2.0 + bin_width / 2.0
    return [first + i * bin_width for i in range(n_samples)]


class BatchedOptomechEnv(gym.vector.VectorEnv):
    """Batched GPU-accelerated VectorEnv for telescope segment alignment.

    Manages N environments as batched GPU tensors.  All physics runs in
    single batched kernel calls.  Implements gymnasium.vector.VectorEnv
    so it can be used directly in place of SyncVectorEnv/AsyncVectorEnv.
    """

    def __init__(self, num_envs, device="auto", silence=True, **env_kwargs):
        self.dev = _resolve_device(device)
        self._num_envs = num_envs
        self._env_kwargs = env_kwargs
        self._silence = silence

        # -----------------------------------------------------------------
        # Validate supported configuration
        # -----------------------------------------------------------------
        assert env_kwargs.get("num_atmosphere_layers", 0) == 0, \
            "V5 requires num_atmosphere_layers=0"
        assert env_kwargs.get("command_secondaries", True), \
            "V5 requires command_secondaries=True"
        assert not env_kwargs.get("command_tensioners", False), \
            "V5 does not support command_tensioners"
        assert not env_kwargs.get("command_dm", False), \
            "V5 does not support command_dm"
        self._incremental_control = env_kwargs.get("incremental_control", True)
        assert not env_kwargs.get("ao_loop_active", False), \
            "V5 does not support ao_loop_active"
        assert not env_kwargs.get("discrete_control", False), \
            "V5 does not support discrete_control"

        # -----------------------------------------------------------------
        # Create ONE V4 env to extract all shared constants
        # -----------------------------------------------------------------
        from optomech.optomech.optomech_v4 import OptomechEnv as V4Env

        v4_kwargs = dict(env_kwargs)
        v4_kwargs["optomech_version"] = "v4"
        v4_kwargs["device"] = "cpu"  # extract on CPU, then move to target
        v4_kwargs["silence"] = silence
        v4 = V4Env(**v4_kwargs)

        os4 = v4.optical_system
        cfg = os4._cfg

        # --- Grid dimensions ---
        self._H = os4._num_px
        self._W = os4._num_px
        self._num_apertures = os4.num_apertures
        self._n_modes = len(os4.segmented_mirror.actuators)
        self._command_tip_tilt = os4.command_tip_tilt
        self._n_dof_per_seg = 3 if self._command_tip_tilt else 1

        # --- Timing ---
        self._max_episode_steps = env_kwargs.get("max_episode_steps", 100)
        ao_interval = cfg.get("ao_interval_ms", 100.0)
        control_interval = cfg.get("control_interval_ms", 100.0)
        frame_interval = cfg.get("frame_interval_ms", 100.0)
        decision_interval = cfg.get("decision_interval_ms", 100.0)
        cmds_per_frame = math.ceil(frame_interval / control_interval)
        frames_per_decision = math.ceil(decision_interval / frame_interval)
        ao_per_cmd = math.ceil(control_interval / ao_interval)
        assert frames_per_decision == 1, \
            f"V5 requires frames_per_decision=1, got {frames_per_decision}"
        assert cmds_per_frame == 1, \
            f"V5 requires commands_per_frame=1, got {cmds_per_frame}"
        assert ao_per_cmd == 1, \
            f"V5 requires ao_steps_per_command=1, got {ao_per_cmd}"
        self._ao_per_frame = ao_per_cmd * cmds_per_frame
        self._frame_sec = frame_interval / 1000.0
        self._integration_sec = self._frame_sec / self._ao_per_frame

        # --- Wavelengths ---
        self._wavelengths = _centered_wavelengths(
            os4.wavelength,
            cfg["bandwidth_nanometers"],
            cfg["bandwidth_sampling"])
        self._n_wl_inv = 1.0 / len(self._wavelengths)

        # --- Shared constants → GPU ---
        # Aperture (complex64)
        ap_np = np.array(os4.aperture).reshape(self._H, self._W)
        self._aperture_t = torch.tensor(ap_np, dtype=torch.complex64, device=self.dev)

        # Influence functions [n_modes, H, W]
        inf_list = [np.array(m) for m in os4.segmented_mirror._influence_functions]
        inf_np = np.stack(inf_list, axis=0).reshape(self._n_modes, self._H, self._W)
        self._influence_t = torch.tensor(inf_np, dtype=torch.float32, device=self.dev)

        # MFT matrices per wavelength
        self._mft_cache = {}
        for wl in self._wavelengths:
            v4_entry = os4._mft_cache[wl]
            M1_t = v4_entry[0].to(self.dev)
            M2_t = v4_entry[1].to(self.dev)
            scale = v4_entry[2]
            self._mft_cache[wl] = (M1_t, M2_t, scale)

        # Object spectrum [H, W] complex64
        self._object_spectrum_t = os4._object_spectrum_t.to(self.dev)

        # Focal grid weight (scalar)
        self._focal_grid_weight = os4._focal_grid_weight

        # Detector model constants
        self._photon_energy = os4._photon_energy
        self._det_qe = os4._det_qe
        self._det_gain = os4._det_gain
        self._det_max_dn = os4._det_max_dn

        # Reference flux values
        self._reference_fpi_sum = os4._reference_fpi_sum
        self._reference_fpi_max = os4._reference_fpi_max

        # Perfect image stats for Strehl
        self._perfect_peak_over_sum = v4._perfect_peak_over_sum

        # Perfect image (for shape reward): normalised log on GPU
        perfect_dn = v4._perfect_image_dn.astype(np.float32)
        perfect_max = float(np.max(perfect_dn))
        if perfect_max > 0:
            norm_perfect = perfect_dn / perfect_max
        else:
            norm_perfect = perfect_dn
        self._norm_perfect_t = torch.tensor(
            norm_perfect, dtype=torch.float32, device=self.dev)
        self._log_norm_perfect_t = torch.tensor(
            np.log(norm_perfect + 1e-10), dtype=torch.float32, device=self.dev)

        # Dark-hole mask (inherited from v4 at build time). If dark_hole
        # is disabled in cfg, v4 leaves _target_zero_mask as None and we
        # mirror that here.
        _v4_mask = getattr(v4, "_target_zero_mask", None)
        if _v4_mask is None:
            self._hole_mask_t = None
        else:
            self._hole_mask_t = torch.tensor(
                _v4_mask.astype(bool), dtype=torch.bool, device=self.dev)

        # --- DEBUG: show perfect image ---
        if cfg.get("_debug_show_perfect", False):
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            im0 = axes[0].imshow(perfect_dn, origin="lower")
            axes[0].set_title(f"perfect_image_dn\nmax={perfect_max:.1f}  "
                              f"shape={perfect_dn.shape}")
            plt.colorbar(im0, ax=axes[0])
            peak_y, peak_x = np.unravel_index(np.argmax(perfect_dn), perfect_dn.shape)
            axes[0].plot(peak_x, peak_y, 'r+', markersize=12, markeredgewidth=2)
            im1 = axes[1].imshow(np.log(norm_perfect + 1e-10), origin="lower")
            axes[1].set_title("log(norm_perfect)")
            plt.colorbar(im1, ax=axes[1])
            axes[1].plot(peak_x, peak_y, 'r+', markersize=12, markeredgewidth=2)
            plt.suptitle(f"V5 perfect image — peak at ({peak_y},{peak_x})")
            plt.tight_layout()
            plt.show()

        # Centering weight map [H, W]
        if hasattr(v4, '_centering_weight'):
            self._centering_weight_t = torch.tensor(
                v4._centering_weight, dtype=torch.float32, device=self.dev)
        else:
            self._centering_weight_t = None
        self._reference_centering_sum = os4._reference_centering_sum

        # --- Correction limits ---
        self._max_p_m = cfg["max_piston_correction_micron"] * 1e-6
        self._max_t_r = cfg["max_tip_correction_arcsec"] * math.pi / (180 * 3600)
        self._max_tl_r = cfg["max_tilt_correction_arcsec"] * math.pi / (180 * 3600)

        # Per-DOF max corrections as [n_modes] tensor for batched clip
        max_corr = []
        for _ in range(self._num_apertures):
            max_corr.append(self._max_p_m)
        for _ in range(self._num_apertures):
            max_corr.append(self._max_t_r)
        for _ in range(self._num_apertures):
            max_corr.append(self._max_tl_r)
        self._max_corr_t = torch.tensor(max_corr, dtype=torch.float32, device=self.dev)

        # Action scaling: maps [-1, 1] to physical correction range
        # For incremental: action * max_correction
        self._action_scale = float(cfg.get("env_action_scale", 1.0))

        # Actuator noise
        self._actuator_noise = cfg.get("actuator_noise", False)
        self._noise_frac = cfg.get("actuator_noise_fraction", 1e-4)
        # Noise std per DOF
        noise_std = []
        for _ in range(self._num_apertures):
            noise_std.append(self._noise_frac * self._max_p_m)
        for _ in range(self._num_apertures):
            noise_std.append(self._noise_frac * self._max_t_r)
        for _ in range(self._num_apertures):
            noise_std.append(self._noise_frac * self._max_tl_r)
        self._noise_std_t = torch.tensor(noise_std, dtype=torch.float32, device=self.dev)

        # --- Non-stationary disturbance: per-step random walk on the
        # secondaries' PTT actuator state. Each step adds
        #     multiplier * base_std * randn()
        # to every actuator DOF, where base_std is a fixed fraction
        # (per DOF type) of that DOF's correctable range. The base
        # fractions are meant to stay constant across experiments;
        # the multiplier is the knob to sweep.
        # Default multiplier is 0.0 so this is inert unless asked for.
        self._ns_multiplier = float(
            cfg.get("nonstationary_noise_multiplier", 0.0))
        ns_p_frac = float(cfg.get("nonstationary_noise_piston_frac", 0.01))
        ns_t_frac = float(cfg.get("nonstationary_noise_tip_frac", 0.01))
        ns_tl_frac = float(cfg.get("nonstationary_noise_tilt_frac", 0.01))
        if self._ns_multiplier > 0.0:
            ns_std = []
            ns_std.extend([self._max_p_m * ns_p_frac] * self._num_apertures)
            ns_std.extend([self._max_t_r * ns_t_frac] * self._num_apertures)
            ns_std.extend([self._max_tl_r * ns_tl_frac] * self._num_apertures)
            self._ns_std_t = torch.tensor(
                ns_std, dtype=torch.float32, device=self.dev)
        else:
            self._ns_std_t = None

        # --- Whole-structure vibration: additive plane of OPD applied
        # over the entire aperture each step, oscillating sinusoidally.
        # Magnitude is specified in arcsec of maximum tilt of the
        # whole structure. Direction is a fixed angle in the pupil
        # plane. Frequency defaults to 100 Hz; phase defaults to 0.
        # Default enabled=False → no vibration; call sites can turn it
        # on explicitly via env kwargs.
        self._vibration_enabled = bool(cfg.get("vibration_enabled", False))
        self._vibration_amp_arcsec = float(
            cfg.get("vibration_amplitude_arcsec", 0.0))
        self._vibration_freq_hz = float(
            cfg.get("vibration_frequency_hz", 100.0))
        self._vibration_direction_rad = float(
            cfg.get("vibration_direction_rad", 0.0))
        self._vibration_phase_rad = float(
            cfg.get("vibration_phase_rad", 0.0))
        # Decision interval in seconds (step index × dt = episode time).
        self._decision_dt_s = float(
            cfg.get("decision_interval_ms", 100.0)) / 1000.0
        # Pre-compute the unit tilt pattern at init time. Coordinates
        # are in meters from the aperture centre, consistent with the
        # HCIPy pupil grid. Stored as (H, W) float32 tensor on device;
        # no-op when vibration is disabled.
        self._vibration_tilt_pattern_t = None
        if self._vibration_enabled and self._vibration_amp_arcsec > 0.0:
            dx = float(os4.pupil_grid.delta[0])  # metres per pixel
            xs = (np.arange(self._W, dtype=np.float32)
                  - (self._W - 1) / 2.0) * dx
            ys = (np.arange(self._H, dtype=np.float32)
                  - (self._H - 1) / 2.0) * dx
            xx_m, yy_m = np.meshgrid(xs, ys, indexing="xy")
            # Direction vector for the tilt axis.
            dx_hat = np.cos(self._vibration_direction_rad)
            dy_hat = np.sin(self._vibration_direction_rad)
            # Unit-amplitude tilt map: OPD in meters per radian of tilt.
            # When multiplied by the current tilt angle (radians) this
            # gives the additive OPD map (meters).
            unit_tilt = (dx_hat * xx_m + dy_hat * yy_m).astype(np.float32)
            self._vibration_tilt_pattern_t = torch.tensor(
                unit_tilt, dtype=torch.float32, device=self.dev)
            self._vibration_amp_rad = (
                self._vibration_amp_arcsec * math.pi / (180.0 * 3600.0))
        else:
            self._vibration_amp_rad = 0.0

        # Action penalty / holding bonus config
        self._action_penalty = cfg.get("action_penalty", False)
        self._action_penalty_weight = cfg.get("action_penalty_weight", 0.03)
        self._holding_bonus_weight = cfg.get("holding_bonus_weight", 0.0)
        self._holding_bonus_min_reward = cfg.get("holding_bonus_min_reward", -1.0)
        self._holding_bonus_threshold = cfg.get("holding_bonus_threshold", 0.0)
        self._minimum_absolute_action = cfg.get("minimum_absolute_action", 0.0)

        # Bootstrap config
        self._bootstrap = env_kwargs.get("bootstrap_phase", False)
        self._phased_count = env_kwargs.get("bootstrap_phased_count", 0)
        # Rescale centered_strehl reward so -1.0 corresponds to the
        # prior phase's solved state (phase N's ideal episode start) and
        # 0.0 corresponds to this phase's solved state. Restores the
        # full [-1, 0] learnable range for high phases whose raw reward
        # is otherwise compressed into a narrow slice near zero.
        self._bootstrap_rescale_reward = bool(
            env_kwargs.get("bootstrap_rescale_reward", False))
        self._rescale_start = 0.0   # populated in _measure_rescale_baselines
        self._rescale_end = 1.0
        # Hard DOF mask: when True, zero out every action DOF except
        # the target segment's. Applied before _batched_command and
        # before _prior_actions is stored, so the agent's outputs on
        # non-target DOFs are structurally inert. This is the strict
        # equivalent of SMAES's hard free_segments=[target] constraint
        # and replaces the soft action-penalty approach.
        self._bootstrap_mask_nontarget = bool(
            env_kwargs.get("bootstrap_mask_nontarget", False))
        self._bootstrap_action_mask_t = None
        if self._bootstrap and self._bootstrap_mask_nontarget:
            n_seg = self._num_apertures
            dof_per_seg = self._n_dof_per_seg
            n_dof = n_seg * dof_per_seg
            mask = np.zeros(n_dof, dtype=np.float32)
            target = self._phased_count
            if target < n_seg:
                mask[target * dof_per_seg] = 1.0           # piston
                if self._command_tip_tilt:
                    mask[target * dof_per_seg + 1] = 1.0   # tip
                    mask[target * dof_per_seg + 2] = 1.0   # tilt
            self._bootstrap_action_mask_t = torch.tensor(
                mask, dtype=torch.float32, device=self.dev).unsqueeze(0)  # [1, n_dof]

        # Per-DOF action penalty weights (bootstrap mode)
        # Action layout is per-segment grouped: [p0,t0,tl0, p1,t1,tl1, ...]
        # _prior_actions uses this layout, so the weight tensor must match.
        # Skipped when bootstrap_mask_nontarget is on (the mask already
        # makes non-target DOFs inert; no per-DOF penalty needed).
        self._action_penalty_weights_t = None
        if self._bootstrap and not self._bootstrap_mask_nontarget:
            n_seg = self._num_apertures
            dof_per_seg = self._n_dof_per_seg
            n_dof = n_seg * dof_per_seg
            multiplier = env_kwargs.get("bootstrap_nontarget_penalty_multiplier", 10.0)
            base_w = self._action_penalty_weight
            weights = np.ones(n_dof, dtype=np.float32) * base_w * multiplier
            target = self._phased_count
            if target < n_seg:
                # Target segment's DOFs in per-segment grouped layout
                weights[target * dof_per_seg] = base_w          # piston
                if self._command_tip_tilt:
                    weights[target * dof_per_seg + 1] = base_w  # tip
                    weights[target * dof_per_seg + 2] = base_w  # tilt
            self._action_penalty_weights_t = torch.tensor(
                weights, dtype=torch.float32, device=self.dev)

        # Reward function config
        self._reward_function = cfg.get("reward_function", "factored")
        # Vector reward: when enabled, _batched_reward always computes ALL
        # raw reward components (regardless of their weights) and stashes
        # them so step() can return them in infos["reward_components"].
        # Each component is in "higher-is-better" form so the env's scalar
        # reward equals sum_i(weight_i * component_i).
        # Default off: zero overhead, output identical to legacy behavior.
        self._reward_vector_enabled = bool(
            cfg.get("reward_vector_enabled", False))
        self._reward_components_t = {}   # name -> [N] tensor (when enabled)
        self._rw_centered_strehl = cfg.get("reward_weight_centered_strehl", 0.0)
        self._rw_contrast_strehl = cfg.get("reward_weight_contrast_strehl", 0.0)
        self._rw_strehl = cfg.get("reward_weight_strehl", 0.0)
        self._rw_centering = cfg.get("reward_weight_centering", 0.0)
        self._rw_flux = cfg.get("reward_weight_flux", 0.0)
        self._rw_peak = cfg.get("reward_weight_peak", 0.0)
        self._rw_shape = cfg.get("reward_weight_shape", 0.0)
        self._rw_image_quality = cfg.get("reward_weight_image_quality", 0.0)
        self._rw_centering_l1 = cfg.get("reward_weight_centering_l1", 0.0)
        self._rw_piston_noise_mse = cfg.get("reward_weight_piston_noise_mse", 0.0)
        self._rw_piston_shape_alignment = cfg.get(
            "reward_weight_piston_shape_alignment", 0.0)
        self._rw_psf_rms_radius = cfg.get(
            "reward_weight_psf_rms_radius", 0.0)
        self._piston_noise_n_trials = cfg.get("piston_noise_n_trials", 1000)
        self._piston_noise_sigma = cfg.get("piston_noise_sigma", 1.0)

        # Centering geometry (precomputed on GPU)
        cy, cx = self._H / 2.0, self._W / 2.0
        self._cy = cy
        self._cx = cx
        self._max_dist = math.sqrt(cy ** 2 + cx ** 2)

        # Pixel coordinate meshes (used by psf_rms_radius reward)
        _yy, _xx = torch.meshgrid(
            torch.arange(self._H, dtype=torch.float32, device=self.dev),
            torch.arange(self._W, dtype=torch.float32, device=self.dev),
            indexing='ij')
        self._pixel_yy_t = _yy            # [H, W]
        self._pixel_xx_t = _xx            # [H, W]

        # Reference R_rms for the psf_rms_radius reward.  Computed once
        # from the perfect (fully cophased) PSF — this is the floor that
        # the reward saturates against.  The ceiling is the half-diagonal
        # of the focal plane (corresponds to a worst-case spread).
        with torch.no_grad():
            _norm_perfect = torch.exp(self._log_norm_perfect_t) - 1e-10
            _norm_perfect = _norm_perfect.clamp(min=0.0)
            _tot = _norm_perfect.sum().clamp(min=1e-30)
            _cy_p = (_norm_perfect * _yy).sum() / _tot
            _cx_p = (_norm_perfect * _xx).sum() / _tot
            _r2_p = (_yy - _cy_p) ** 2 + (_xx - _cx_p) ** 2
            _m2_p = (_norm_perfect * _r2_p).sum() / _tot
            self._psf_rms_ref_min = float(torch.sqrt(_m2_p).item())
        self._psf_rms_ref_max = float(
            ((self._H ** 2 + self._W ** 2) ** 0.5) / 2.0)

        # L1 distance map [H, W] normalised to [0, 1] — for L1-centering reward
        yy_np, xx_np = np.mgrid[0:self._H, 0:self._W].astype(np.float32)
        l1_map_np = (np.abs(yy_np - cy) + np.abs(xx_np - cx))
        l1_max = float(l1_map_np.max())
        if l1_max > 0:
            l1_map_np /= l1_max
        self._l1_dist_map_t = torch.tensor(
            l1_map_np, dtype=torch.float32, device=self.dev)

        # OOB penalty
        self._oob_penalty = cfg.get("oob_penalty", False)
        self._oob_penalty_weight = cfg.get("oob_penalty_weight", 2.0)

        # Init diff motion config
        self._init_diff_motion = cfg.get("init_differential_motion", False)
        self._model_wind = cfg.get("model_wind_diff_motion", False)
        self._init_piston_std = cfg.get("init_wind_piston_micron_std", 3.0)
        self._init_piston_clip_m = cfg.get("init_wind_piston_clip_m", 4e-6)
        self._init_tip_std = cfg.get("init_wind_tip_arcsec_std_tt", 0.0) if self._command_tip_tilt else 0.0
        self._init_tilt_std = cfg.get("init_wind_tilt_arcsec_std_tt", 0.0) if self._command_tip_tilt else 0.0

        # Configurable init-diff-motion path. When enabled, the init-time
        # piston/tip/tilt perturbation is drawn from these fields instead
        # of the hard-coded λ/4 wind model. All defaults are zero, so
        # enabling the flag with no overrides produces no perturbation.
        self._init_diff_motion_cfg = cfg.get(
            "init_differential_motion_configurable", False)
        self._init_p_mean_m = float(cfg.get("init_piston_micron_mean", 0.0)) * 1e-6
        self._init_p_std_m = float(cfg.get("init_piston_micron_std", 0.0)) * 1e-6
        self._init_p_clip_m = float(cfg.get("init_piston_clip_micron", 0.0)) * 1e-6
        self._init_t_std_arcsec = float(cfg.get("init_tip_arcsec_std", 0.0))
        self._init_tl_std_arcsec = float(cfg.get("init_tilt_arcsec_std", 0.0))
        self._wl = os4.wavelength

        # Observation shape: frames_per_decision (matches V4 behavior).
        # observation_window_size is deprecated and ignored.
        self._obs_window = 1  # frames_per_decision is always 1 for V5

        # --- Gymnasium VectorEnv setup ---
        single_obs_space = spaces.Box(
            low=0.0, high=np.float32(self._det_max_dn),
            shape=(self._obs_window, self._H, self._W),
            dtype=np.float32)
        single_action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self._num_apertures * self._n_dof_per_seg,),
            dtype=np.float32)
        super().__init__(
            num_envs=num_envs,
            observation_space=single_obs_space,
            action_space=single_action_space)

        # --- Allocate per-env state on GPU ---
        N = num_envs
        self._actuators_t = torch.zeros(N, self._n_modes, dtype=torch.float32, device=self.dev)
        # Baselines: [N, n_modes] — the baseline actuator values around which
        # incremental commands are centered and clipped.
        self._baselines_t = torch.zeros(N, self._n_modes, dtype=torch.float32, device=self.dev)
        self._step_counts = torch.zeros(N, dtype=torch.long, device=self.dev)
        self._prior_actions = torch.zeros(N, single_action_space.shape[0],
                                          dtype=torch.float32, device=self.dev)

        # Observation history buffer for obs_window > 1
        self._obs_history = torch.zeros(
            N, self._obs_window, self._H, self._W,
            dtype=torch.float32, device=self.dev)

        # Episode return tracking
        self._episode_returns = torch.zeros(N, dtype=torch.float32, device=self.dev)
        self._episode_lengths = torch.zeros(N, dtype=torch.long, device=self.dev)

        # Bootstrap off-axis constants (extracted from V4 optical system)
        if self._bootstrap:
            self._bootstrap_off_axis = os4._bootstrap_off_axis_angle()
            self._bootstrap_seg_angles = torch.tensor(
                os4._bootstrap_segment_angles(),
                dtype=torch.float32, device=self.dev)

        # Discard V4 env
        v4.close()
        del v4, os4

        # --- Piston-noise target image --------------------------------------
        # Run n random piston-only perturbations (tip/tilt held at zero),
        # average the resulting normalised frames to create a target shape.
        # Used by both ``piston_noise_mse`` (pixelwise MSE penalty) and
        # ``piston_shape_alignment`` (flux-weighted alignment score).
        self._piston_noise_target_t = None
        if self._rw_piston_noise_mse > 0 or self._rw_piston_shape_alignment > 0:
            self._piston_noise_target_t = self._compute_piston_noise_target()

        # Bootstrap reward-rescale baselines (after everything else is set).
        if self._bootstrap and self._bootstrap_rescale_reward:
            self._measure_rescale_baselines()
            print(f"[optomech-v5] Bootstrap reward rescale: "
                  f"s_prior={self._rescale_start:.4f}, "
                  f"s_goal={self._rescale_end:.4f}  "
                  f"(phase {self._phased_count})")

        if not silence:
            print(f"[optomech-v5] BatchedOptomechEnv ready: {num_envs} envs on {self.dev}")
            print(f"[optomech-v5]   {self._num_apertures} segments, {self._n_modes} modes")
            print(f"[optomech-v5]   {len(self._wavelengths)} wavelengths")
            print(f"[optomech-v5]   image: {self._H}x{self._W}")

    # =================================================================
    # Batched physics
    # =================================================================

    def _batched_simulate(self):
        """Run batched MFT propagation for all envs and wavelengths.

        Returns the accumulated science frame [N, H, W] on GPU.
        """
        N = self._num_envs
        dev = self.dev

        # Surface = sum(actuators_i * influence_i) for each env
        # actuators: [N, n_modes], influence: [n_modes, H, W]
        surface = torch.einsum('ni,ihw->nhw', self._actuators_t, self._influence_t)

        # Additive whole-structure vibration OPD. Each env carries its
        # own step counter, so phase is evaluated per-env. Inert when
        # vibration is disabled (tilt_pattern is None), so this branch
        # is zero-cost at default settings.
        if self._vibration_tilt_pattern_t is not None:
            t_s = self._step_counts.to(torch.float32) * self._decision_dt_s
            theta_t = (self._vibration_amp_rad
                       * torch.sin(
                           2.0 * math.pi * self._vibration_freq_hz * t_s
                           + self._vibration_phase_rad))
            # theta_t: [N] → [N, 1, 1]; tilt_pattern: [H, W] broadcasts.
            surface = surface + (theta_t.view(N, 1, 1)
                                 * self._vibration_tilt_pattern_t.unsqueeze(0))

        frame = torch.zeros(N, self._H, self._W, dtype=torch.float32, device=dev)

        for wl in self._wavelengths:
            k = 2.0 * math.pi / wl
            M1, M2, scale = self._mft_cache[wl]

            # E = aperture * exp(2jk * surface)
            # aperture_t: [H, W] broadcasts to [N, H, W]
            phase = torch.zeros_like(surface)
            E = self._aperture_t.unsqueeze(0) * torch.exp(
                torch.complex(phase, 2.0 * k * surface))

            # Batched MFT: E_focal = M1 @ E @ M2 * scale
            # M1: [H, H], E: [N, H, W] → matmul broadcasts over batch
            E_focal = torch.matmul(M1, E)
            E_focal = torch.matmul(E_focal, M2) * scale

            # PSF = |E_focal|^2 * grid_weight * integration_time
            psf = (torch.abs(E_focal) ** 2) * (
                self._focal_grid_weight * self._integration_sec)

            # Object convolution: image = |IFFT(FFT(PSF) * obj_spectrum)|
            otf = torch.fft.fft2(psf)
            # obj_spectrum: [H, W] broadcasts to [N, H, W]
            image = torch.abs(torch.fft.fftshift(
                torch.fft.ifft2(self._object_spectrum_t.unsqueeze(0) * otf),
                dim=(-2, -1)))

            frame += image * self._n_wl_inv

        # Scale by ao_steps_per_frame (= 1 for nanoelf_tt)
        frame *= self._ao_per_frame

        return frame

    def _batched_detector_model(self, frame):
        """Convert power-per-pixel frame to DN. All on GPU."""
        energy = frame * self._frame_sec
        n_photons = energy / self._photon_energy
        n_electrons = n_photons * self._det_qe
        dn = n_electrons / self._det_gain
        return torch.clamp(dn, 0, self._det_max_dn)

    def compute_surface_for_action(self, action, env_slot=0):
        """Return the wavefront surface (OPD) for an arbitrary action
        without advancing env state.

        action : np.ndarray of shape (D,) or (N, D), in [-1, 1].
        env_slot : which env's baseline to use as the reference (default 0).

        Returns surface as a numpy array shaped (H, W) for a single
        action, or (N, H, W) for a batch.  Units: meters of optical
        path difference (matches the env's internal _actuators_t units
        for piston DOFs; tip/tilt contributions enter via the influence
        functions).

        This intentionally does NOT clip, add noise, or write any env
        state — it's purely a render helper.
        """
        a = torch.as_tensor(action, dtype=torch.float32, device=self.dev)
        squeezed = (a.dim() == 1)
        if squeezed:
            a = a.unsqueeze(0)                  # [1, D]
        N = a.shape[0]
        n_seg = self._num_apertures
        n_dof = self._n_dof_per_seg
        n_active = n_seg * n_dof

        # Per-segment grouping → blocked actuator order
        a_reshaped = a.reshape(N, n_seg, n_dof)
        a_reordered = a_reshaped.permute(0, 2, 1).reshape(N, -1)

        delta_full = torch.zeros(
            N, self._n_modes, dtype=torch.float32, device=self.dev)
        delta_full[:, :n_active] = (
            a_reordered * self._max_corr_t[:n_active].unsqueeze(0))

        baseline = self._baselines_t[env_slot:env_slot + 1]   # [1, n_modes]
        actuators = baseline + delta_full                       # [N, n_modes]

        surface = torch.einsum(
            'ni,ihw->nhw', actuators, self._influence_t)        # [N, H, W]
        out = surface.detach().cpu().numpy()
        if squeezed:
            return out[0]
        return out

    def _batched_command(self, actions_t):
        """Apply incremental commands to actuators. All on GPU.

        actions_t: [N, action_dim] in [-1, 1], already scaled by action_scale.

        For nanoelf with tip/tilt: action_dim = 6 (2 segments × 3 DOF).
        The actuator layout is [piston_0, piston_1, tip_0, tip_1, tilt_0, tilt_1].
        The action layout is [p0, t0, tl0, p1, t1, tl1] (per-segment grouping).
        We need to remap action→actuator ordering.
        """
        N = self._num_envs
        n_seg = self._num_apertures
        n_dof = self._n_dof_per_seg

        # Remap action (per-segment: [p0,t0,tl0, p1,t1,tl1, ...]) to
        # actuator (per-DOF: [p0,p1,..., t0,t1,..., tl0,tl1,...])
        action_reshaped = actions_t.reshape(N, n_seg, n_dof)  # [N, n_seg, n_dof]
        # Transpose to [N, n_dof, n_seg] then flatten to [N, n_active]
        action_reordered = action_reshaped.permute(0, 2, 1).reshape(N, -1)

        # Build full-mode delta (zero for inactive DOFs like tip/tilt in piston-only)
        n_active = n_seg * n_dof
        if n_active < self._n_modes:
            delta_full = torch.zeros(N, self._n_modes, dtype=torch.float32, device=self.dev)
            delta_full[:, :n_active] = action_reordered * self._max_corr_t[:n_active].unsqueeze(0)
        else:
            delta_full = action_reordered * self._max_corr_t.unsqueeze(0)

        # Incremental: actuators += delta; Absolute: actuators = baseline + scaled_action
        if self._incremental_control:
            pre_clip = self._actuators_t + delta_full
        else:
            pre_clip = self._baselines_t + delta_full

        # Clip to [baseline - max, baseline + max]
        low = self._baselines_t - self._max_corr_t.unsqueeze(0)
        high = self._baselines_t + self._max_corr_t.unsqueeze(0)
        post_clip = torch.clamp(pre_clip, low, high)

        # Track OOB fraction (only count active DOFs)
        clipped_full = (pre_clip != post_clip)  # [N, n_modes]
        if self._command_tip_tilt:
            n_active = self._n_modes
        else:
            n_active = self._num_apertures  # piston only
        self._oob_frac = clipped_full[:, :n_active].float().mean(dim=1)  # [N]

        # Rail noise: add noise where clipped (full tensor)
        noise = torch.randn_like(post_clip) * self._noise_std_t.unsqueeze(0)
        post_clip = torch.where(clipped_full, post_clip + noise, post_clip)

        # Actuator repeatability noise on non-zero commands
        if self._actuator_noise:
            # Build full-size nonzero mask
            cmd_nonzero = torch.zeros(N, self._n_modes, dtype=torch.bool, device=self.dev)
            cmd_nonzero[:, :n_active] = (action_reordered != 0.0)
            rep_noise = torch.randn_like(post_clip) * self._noise_std_t.unsqueeze(0)
            post_clip = torch.where(cmd_nonzero, post_clip + rep_noise, post_clip)

        self._actuators_t = post_clip

    def _measure_rescale_baselines(self):
        """Measure (strehl * centering) at this phase's prior-solved
        and goal-solved states, used to rescale the centered_strehl
        reward into a full [-1, 0] per-phase range.

        prior-solved = segs 0..phased_count-1 aligned, segs
                       phased_count..n-1 off-axis. This is the state
                       phase N-1 is supposed to produce (and phase N's
                       ideal starting condition).
        goal-solved  = segs 0..phased_count aligned, segs
                       phased_count+1..n-1 off-axis. This is what phase
                       N is supposed to achieve.

        Both configurations are built WITHOUT the ±10% off-axis noise
        and WITHOUT the wind-piston kick on the target, so the
        measurements are fully deterministic. All envs are set to the
        same state, so the mean across envs is a clean scalar.

        Side effect: temporarily overwrites _actuators_t (restored
        before return). Safe to call at the tail of __init__ before
        reset() has ever run.
        """
        saved_actuators = self._actuators_t.clone()
        try:
            for tag, target in (
                ("start", self._phased_count),
                ("end", self._phased_count + 1),
            ):
                self._set_deterministic_bootstrap_state(target)
                frames = self._batched_simulate()
                frames = self._batched_detector_model(frames)

                # Same strehl + centering math as _batched_reward.
                fpi_sum = frames.sum(dim=(1, 2))
                fpi_max = frames.amax(dim=(1, 2))
                safe_sum = fpi_sum.clamp(min=1e-30)
                obs_peak_over_sum = fpi_max / safe_sum
                raw_strehl = obs_peak_over_sum / self._perfect_peak_over_sum
                flux_frac = (fpi_sum / self._reference_fpi_sum).clamp(0.0, 1.0)
                strehl = raw_strehl * flux_frac

                H, W = frames.shape[-2], frames.shape[-1]
                peak_flat = frames.reshape(self._num_envs, -1).argmax(dim=1)
                py = (peak_flat // W).float()
                px = (peak_flat % W).float()
                dist = ((py - self._cy) ** 2 + (px - self._cx) ** 2).sqrt()
                centering = 1.0 - dist / self._max_dist

                s_c = float((strehl * centering).mean().item())
                if tag == "start":
                    self._rescale_start = s_c
                else:
                    self._rescale_end = s_c
        finally:
            self._actuators_t = saved_actuators

    def _set_deterministic_bootstrap_state(self, target):
        """Set _actuators_t so segs [0..target-1] are aligned (zero PTT)
        and segs [target..n-1] carry the deterministic off-axis tip/tilt
        push (no ±10% noise, no wind kick).
        """
        n_seg = self._num_apertures
        N = self._num_envs
        off_axis = self._bootstrap_off_axis
        seg_angles = self._bootstrap_seg_angles

        piston = torch.zeros(N, n_seg, device=self.dev)
        tip = torch.zeros(N, n_seg, device=self.dev)
        tilt = torch.zeros(N, n_seg, device=self.dev)
        for seg_id in range(target, n_seg):
            theta = seg_angles[seg_id]
            tip[:, seg_id] = off_axis * torch.sin(theta)
            tilt[:, seg_id] = off_axis * torch.cos(theta)
        self._actuators_t = torch.cat([piston, tip, tilt], dim=1)

    def _compute_piston_noise_target(self):
        """Generate a target shape by averaging frames from random piston noise.

        Runs ``piston_noise_n_trials`` piston-only perturbations (tip/tilt
        held fixed at zero), normalises each frame by its own max, and
        averages them.  This produces a fuzzy, nominally centered target
        that can be compared via MSE.  Actuator / baseline / episode state
        is saved and restored so this has no side-effect on future episodes.
        """
        N = self._num_envs
        n_seg = self._num_apertures
        n_trials = int(self._piston_noise_n_trials)
        sigma = float(self._piston_noise_sigma)

        if n_trials <= 0:
            return None

        # Save state that the warmup will perturb
        saved_actuators = self._actuators_t.clone()
        saved_baselines = self._baselines_t.clone()
        saved_prior_actions = self._prior_actions.clone()
        saved_obs_history = self._obs_history.clone()
        saved_incremental = self._incremental_control

        # Force absolute-control for warmup so each batch is independent
        self._incremental_control = False
        self._actuators_t.zero_()
        self._baselines_t.zero_()

        accum = torch.zeros(self._H, self._W, dtype=torch.float32, device=self.dev)
        n_accumulated = 0

        action_dim = n_seg * self._n_dof_per_seg
        trials_remaining = n_trials
        while trials_remaining > 0:
            batch = min(N, trials_remaining)
            # Random piston-only actions
            actions = torch.zeros(N, action_dim, dtype=torch.float32, device=self.dev)
            if self._n_dof_per_seg == 1:
                # Piston-only env
                actions[:batch, :n_seg] = (
                    torch.rand(batch, n_seg, device=self.dev) * 2.0 - 1.0) * sigma
            else:
                # [p0, t0, tl0, p1, t1, tl1, ...] — set piston slots only
                for s in range(n_seg):
                    actions[:batch, s * self._n_dof_per_seg] = (
                        torch.rand(batch, device=self.dev) * 2.0 - 1.0) * sigma

            self._batched_command(actions)
            frames = self._batched_simulate()
            frames = self._batched_detector_model(frames)

            # Normalise each frame by its own max before accumulating
            fmax = frames.reshape(N, -1).amax(dim=1).clamp(min=1e-30)
            norm_frames = frames / fmax.unsqueeze(1).unsqueeze(2)
            accum += norm_frames[:batch].sum(dim=0)
            n_accumulated += batch

            trials_remaining -= batch

        target = accum / max(n_accumulated, 1)
        # Normalise the target itself so its max is 1.0 (for MSE comparison)
        tmax = float(target.max().item())
        if tmax > 0:
            target = target / tmax

        # Restore state
        self._actuators_t = saved_actuators
        self._baselines_t = saved_baselines
        self._prior_actions = saved_prior_actions
        self._obs_history = saved_obs_history
        self._incremental_control = saved_incremental

        if not self._silence:
            peak_idx = torch.argmax(target.reshape(-1)).item()
            py, px = peak_idx // self._W, peak_idx % self._W
            print(f"[optomech-v5] piston-noise target: {n_accumulated} trials, "
                  f"peak@({py},{px}), mean={float(target.mean()):.4f}")

        return target

    def _batched_reward(self, frames):
        """Compute batched reward on GPU.

        frames: [N, H, W] detector output in DN.
        Returns: [N] reward tensor on GPU.

        If self._reward_vector_enabled is True, every component is
        computed (regardless of weight) and stashed in
        self._reward_components_t as {name: [N] tensor}.  Each component
        is "higher-is-better" so that the scalar reward equals
        ``sum_i(weight_i * component_i)``.
        """
        N = self._num_envs
        H, W = self._H, self._W

        total = torch.zeros(N, dtype=torch.float32, device=self.dev)
        vec = self._reward_vector_enabled
        if vec:
            self._reward_components_t = {}

        # --- Strehl computation (shared by multiple reward components) ---
        fpi_sum = frames.sum(dim=(1, 2))  # [N]
        fpi_max = frames.amax(dim=(1, 2))  # [N]

        # Avoid div by zero
        safe_sum = fpi_sum.clamp(min=1e-30)
        obs_peak_over_sum = fpi_max / safe_sum
        raw_strehl = obs_peak_over_sum / self._perfect_peak_over_sum

        flux_frac = (fpi_sum / self._reference_fpi_sum).clamp(min=0.0, max=1.0)
        strehl = raw_strehl * flux_frac  # [N]

        # --- Centering: distance of peak pixel from center ---
        peak_flat = frames.reshape(N, -1).argmax(dim=1)  # [N]
        py = (peak_flat // W).float()
        px = (peak_flat % W).float()
        dist = ((py - self._cy) ** 2 + (px - self._cx) ** 2).sqrt()
        centering = 1.0 - dist / self._max_dist  # [N]

        # --- Pre-compute frequently shared image-wise quantities ---
        if vec or self._rw_shape > 0 or self._rw_image_quality > 0 \
                or self._rw_piston_noise_mse > 0:
            frame_max_safe = fpi_max.clamp(min=1e-30).unsqueeze(1).unsqueeze(2)
            norm_frames = frames / frame_max_safe  # [N, H, W]

        # --- centered_strehl (component value: -(1 - s*c)) ---
        # Under bootstrap_rescale_reward, the component is rescaled so
        # -1 corresponds to this phase's prior-solved state (N aligned)
        # and 0 corresponds to this phase's goal-solved state (N+1
        # aligned). Gives every phase the full [-1, 0] range regardless
        # of starting strehl ceiling. Values below -1 are clamped at
        # -5 so catastrophic states don't produce huge advantage spikes.
        cs_val = None
        if self._rw_centered_strehl > 0 or vec:
            s_c = strehl * centering
            if self._bootstrap_rescale_reward:
                denom = max(self._rescale_end - self._rescale_start, 1e-3)
                cs_val = ((s_c - self._rescale_end) / denom).clamp(min=-5.0)
            else:
                cs_val = -(1.0 - s_c)
            if self._rw_centered_strehl > 0:
                total += self._rw_centered_strehl * cs_val

        # --- strehl ---
        s_val = None
        if self._rw_strehl > 0 or vec:
            s_val = -(1.0 - strehl)
            if self._rw_strehl > 0:
                total += self._rw_strehl * s_val

        # --- contrast-weighted strehl ---
        # contrast = 1 - max(fpi[hole]) / max(fpi[~hole]).  Multiplies
        # strehl to only credit the PSF core when the hole is dark
        # relative to it. If no hole is configured, contrast = 1 and
        # this degenerates to plain strehl.
        ct_val = None
        if self._rw_contrast_strehl > 0 or vec:
            if self._hole_mask_t is None:
                contrast = torch.ones(N, dtype=torch.float32, device=self.dev)
            else:
                hole = self._hole_mask_t  # [H, W]
                frames_flat = frames.reshape(N, -1)
                hole_flat = hole.reshape(-1)
                # Mask values outside the hole to -inf / inside to -inf
                # in the opposing view so amax picks the intended peak.
                neg_inf = torch.full_like(frames_flat, float("-inf"))
                in_hole = torch.where(hole_flat.unsqueeze(0), frames_flat,
                                      neg_inf)
                out_hole = torch.where(~hole_flat.unsqueeze(0), frames_flat,
                                       neg_inf)
                hole_max = in_hole.amax(dim=1)
                out_max = out_hole.amax(dim=1)
                safe_out = out_max.clamp(min=1e-30)
                contrast = (1.0 - hole_max / safe_out).clamp(min=0.0, max=1.0)
                # If the outside region had no finite max (degenerate),
                # fall back to zero contrast.
                contrast = torch.where(out_max > 0, contrast,
                                       torch.zeros_like(contrast))
            ct_val = -(1.0 - contrast * strehl)
            if self._rw_contrast_strehl > 0:
                total += self._rw_contrast_strehl * ct_val

        # --- centering ---
        cen_component = None
        if self._rw_centering > 0 or vec:
            if self._centering_weight_t is not None:
                weighted = (frames * self._centering_weight_t.unsqueeze(0)).sum(dim=(1, 2))
                cen_val = (weighted / self._reference_centering_sum).clamp(max=1.0)
            else:
                cen_val = centering
            cen_component = -(1.0 - cen_val)
            if self._rw_centering > 0:
                total += self._rw_centering * cen_component

        # --- flux ---
        flux_component = None
        if self._rw_flux > 0 or vec:
            flux_component = -(1.0 - flux_frac)
            if self._rw_flux > 0:
                total += self._rw_flux * flux_component

        # --- peak ---
        peak_component = None
        if self._rw_peak > 0 or vec:
            peak_ratio = (fpi_max / self._reference_fpi_max).clamp(max=1.0)
            peak_component = -(1.0 - peak_ratio)
            if self._rw_peak > 0:
                total += self._rw_peak * peak_component

        # --- shape: log-space MSE vs perfect image ---
        shape_component = None
        if self._rw_shape > 0 or vec:
            log_frames = torch.log(norm_frames + 1e-10)
            diff = log_frames - self._log_norm_perfect_t.unsqueeze(0)
            shape_mse = (diff ** 2).mean(dim=(1, 2))
            shape_component = -shape_mse
            if self._rw_shape > 0:
                total += self._rw_shape * shape_component

        # --- image_quality: linear-space MSE ---
        iq_component = None
        if self._rw_image_quality > 0 or vec:
            diff = norm_frames - self._norm_perfect_t.unsqueeze(0)
            iq_mse = (diff ** 2).mean(dim=(1, 2))
            iq_component = -iq_mse
            if self._rw_image_quality > 0:
                total += self._rw_image_quality * iq_component

        # --- centering_l1: flux-weighted mean L1 distance ---
        l1_component = None
        if self._rw_centering_l1 > 0 or vec:
            safe = fpi_sum.clamp(min=1e-30).unsqueeze(1).unsqueeze(2)
            prob = frames / safe
            l1_mean = (prob * self._l1_dist_map_t.unsqueeze(0)).sum(dim=(1, 2))
            l1_component = -l1_mean
            if self._rw_centering_l1 > 0:
                total += self._rw_centering_l1 * l1_component

        # --- piston_noise_mse ---
        pn_component = None
        if (self._rw_piston_noise_mse > 0 or vec) \
                and self._piston_noise_target_t is not None:
            diff = norm_frames - self._piston_noise_target_t.unsqueeze(0)
            pn_mse = (diff ** 2).mean(dim=(1, 2))
            pn_component = -pn_mse
            if self._rw_piston_noise_mse > 0:
                total += self._rw_piston_noise_mse * pn_component

        # --- piston_shape_alignment ---
        # Flux-weighted alignment with the piston-noise target shape:
        #   score = sum(I * T) / sum(I)
        # i.e. the average target value at the locations weighted by
        # image flux.  Bounded in [T.min(), T.max()] = [~0, 1.0] since
        # the target is normalised to peak 1.0.  Reward = score - 1
        # so 0 = all flux at the target peak, -1 = all flux outside
        # the target's support.  Unlike piston_noise_mse, this does
        # NOT penalise dark regions of the image — only flux that
        # lands outside the target shape.
        psa_component = None
        if (self._rw_piston_shape_alignment > 0 or vec) \
                and self._piston_noise_target_t is not None:
            target = self._piston_noise_target_t.unsqueeze(0)  # [1,H,W]
            flux_total = frames.sum(dim=(1, 2)).clamp(min=1e-30)
            flux_weighted = (frames * target).sum(dim=(1, 2))
            align_score = flux_weighted / flux_total            # ~[0, 1]
            psa_component = align_score - 1.0                    # [-1, 0]
            if self._rw_piston_shape_alignment > 0:
                total += self._rw_piston_shape_alignment * psa_component

        # --- psf_rms_radius ---
        # Flux-weighted RMS radius of the focal plane image about its
        # OWN centroid (translation-invariant).  This isolates the
        # *spread* of the flux, which is dominated by tip/tilt error
        # rather than piston: piston-only error keeps the speckles
        # within the single-segment Airy disc, while TT errors push
        # each segment's sub-PSF to a different position and inflate
        # the second moment.
        #
        # Normalised against (R_min, R_max) where R_min is the perfect
        # cophased PSF's R_rms (computed at env init) and R_max is the
        # focal plane half-diagonal.  Reward in [-1, 0]: 0 = at the
        # diffraction-limited floor, -1 = saturated wide.
        prr_component = None
        if self._rw_psf_rms_radius > 0 or vec:
            fpi_safe = fpi_sum.clamp(min=1e-30)
            cy_pred = (frames * self._pixel_yy_t.unsqueeze(0)).sum(
                dim=(1, 2)) / fpi_safe                          # [N]
            cx_pred = (frames * self._pixel_xx_t.unsqueeze(0)).sum(
                dim=(1, 2)) / fpi_safe                          # [N]
            dy = (self._pixel_yy_t.unsqueeze(0)
                   - cy_pred.unsqueeze(1).unsqueeze(2))
            dx = (self._pixel_xx_t.unsqueeze(0)
                   - cx_pred.unsqueeze(1).unsqueeze(2))
            r2 = dy * dy + dx * dx
            m2 = (frames * r2).sum(dim=(1, 2)) / fpi_safe       # [N]
            r_rms = torch.sqrt(m2.clamp(min=0.0))               # [N]
            lo = self._psf_rms_ref_min
            hi = self._psf_rms_ref_max
            denom = max(hi - lo, 1e-9)
            norm = ((r_rms - lo) / denom).clamp(0.0, 1.0)
            prr_component = -norm                                # [-1, 0]
            if self._rw_psf_rms_radius > 0:
                total += self._rw_psf_rms_radius * prr_component

        # --- Stash components for vector-reward output ----------------
        if vec:
            self._reward_components_t["centered_strehl"] = cs_val
            self._reward_components_t["strehl"] = s_val
            self._reward_components_t["centering"] = cen_component
            self._reward_components_t["flux"] = flux_component
            self._reward_components_t["peak"] = peak_component
            if shape_component is not None:
                self._reward_components_t["shape"] = shape_component
            if iq_component is not None:
                self._reward_components_t["image_quality"] = iq_component
            self._reward_components_t["centering_l1"] = l1_component
            if pn_component is not None:
                self._reward_components_t["piston_noise_mse"] = pn_component
            if psa_component is not None:
                self._reward_components_t["piston_shape_alignment"] = psa_component
            if prr_component is not None:
                self._reward_components_t["psf_rms_radius"] = prr_component

        # Store raw reward before bonuses/penalties
        self._raw_reward = total.clone()

        # --- Holding bonus ---
        if self._holding_bonus_weight > 0:
            min_r = self._holding_bonus_min_reward
            thresh = self._holding_bonus_threshold
            span = -min_r if min_r != 0 else 1.0
            raw_quality = ((total - min_r) / span).clamp(0.0, 1.0)

            if thresh < 0:
                q_thresh = max((thresh - min_r) / span, 0.0)
                quality = torch.where(
                    raw_quality > q_thresh,
                    (raw_quality - q_thresh) / (1.0 - q_thresh),
                    torch.zeros_like(raw_quality))
            else:
                quality = raw_quality

            # When the bootstrap mask is active, the 42 non-target DOFs
            # are structurally zero in _prior_actions. Averaging over
            # all 45 makes stillness ~1 even when the target is
            # pegged, erasing the differential between "moving" and
            # "stopped". Normalize by the number of actually-actuating
            # DOFs (sum of the mask) so stillness reflects what the
            # policy is commanding.
            if self._bootstrap_action_mask_t is not None:
                n_eff = self._bootstrap_action_mask_t.sum().clamp(min=1.0)
                action_l1 = self._prior_actions.abs().sum(dim=1) / n_eff
            else:
                action_l1 = self._prior_actions.abs().mean(dim=1)  # [N]
            stillness = (1.0 - action_l1).clamp(min=0.0)
            total = total + self._holding_bonus_weight * quality * stillness

        # --- Action penalty ---
        if self._action_penalty:
            abs_actions = self._prior_actions.abs()  # [N, action_dim]
            if self._action_penalty_weights_t is not None:
                # Per-DOF weighted penalty (bootstrap mode, legacy path
                # when mask is disabled).
                action_penalty = (self._action_penalty_weights_t * abs_actions).mean(dim=1)
                total = total - action_penalty
            else:
                # Uniform scalar penalty. Same dilution story as
                # holding bonus above: when the mask pins 42 DOFs to
                # zero, averaging over 45 gives ~15x too small a
                # penalty. Normalize by the active DOFs only.
                if self._bootstrap_action_mask_t is not None:
                    n_eff = self._bootstrap_action_mask_t.sum().clamp(min=1.0)
                    action_l1 = abs_actions.sum(dim=1) / n_eff
                else:
                    action_l1 = abs_actions.mean(dim=1)  # [N]
                total = total - self._action_penalty_weight * action_l1

        # --- OOB penalty ---
        if self._oob_penalty:
            total = total - self._oob_penalty_weight * self._oob_frac

        # Store diagnostic values
        self._last_strehl = strehl
        self._last_centering = centering
        self._last_fpi_sum = fpi_sum

        return total

    # =================================================================
    # Reset
    # =================================================================

    def _init_actuators(self, env_mask=None):
        """Initialize actuators with random perturbations.

        env_mask: bool tensor [N] indicating which envs to reset.
                  If None, reset all envs.
        """
        if env_mask is None:
            env_mask = torch.ones(self._num_envs, dtype=torch.bool, device=self.dev)

        n_reset = int(env_mask.sum().item())
        if n_reset == 0:
            return

        n_seg = self._num_apertures

        # Start from zero actuators
        self._actuators_t[env_mask] = 0.0

        if self._bootstrap:
            # --- Bootstrap mode ---
            # Co-phased segments (0..target-1) stay at zero. The target
            # segment AND every later segment (target..n_seg-1) are tipped
            # off the focal plane so the only on-axis light at reset comes
            # from the already-aligned segments. The policy's job is to
            # bring the target segment from off-axis back to aligned; it
            # is also given a random piston kick drawn from the wind
            # disturbance model so it has a non-trivial piston to resolve.
            target = self._phased_count
            off_axis = self._bootstrap_off_axis
            seg_angles = self._bootstrap_seg_angles  # [n_seg] on GPU

            piston = torch.zeros(n_reset, n_seg, device=self.dev)
            tip = torch.zeros(n_reset, n_seg, device=self.dev)
            tilt = torch.zeros(n_reset, n_seg, device=self.dev)

            # Off-axis push: applies to the target AND all later segments.
            for seg_id in range(target, n_seg):
                noise = 1.0 + (torch.rand(n_reset, device=self.dev) * 0.2 - 0.1)
                theta = seg_angles[seg_id]
                tip[:, seg_id] = off_axis * torch.sin(theta) * noise
                tilt[:, seg_id] = off_axis * torch.cos(theta) * noise

            # Random piston kick on the target (wind disturbance model).
            # Does NOT touch tip/tilt — those carry the off-axis push above.
            if self._init_diff_motion_cfg:
                signs = torch.sign(torch.randn(n_reset, device=self.dev))
                signs = torch.where(signs == 0, torch.ones_like(signs), signs)
                p_target = signs * (
                    self._init_p_mean_m
                    + torch.randn(n_reset, device=self.dev) * self._init_p_std_m)
                if self._init_p_clip_m > 0.0:
                    p_target = p_target.clamp(
                        -self._init_p_clip_m, self._init_p_clip_m)
                piston[:, target] = p_target
            elif self._init_diff_motion and self._model_wind:
                wl = self._wl
                piston_mean = wl / 4.0
                piston_std = wl / 6.0
                clip_m = self._init_piston_clip_m

                signs = torch.sign(torch.randn(n_reset, device=self.dev))
                signs = torch.where(signs == 0, torch.ones_like(signs), signs)
                p_target = signs * (piston_mean + torch.randn(n_reset, device=self.dev) * piston_std)
                piston[:, target] = p_target.clamp(-clip_m, clip_m)

            actuators = torch.cat([piston, tip, tilt], dim=1)
            self._actuators_t[env_mask] = actuators

        elif self._init_diff_motion_cfg:
            # --- Configurable standard mode ---
            signs = torch.sign(torch.randn(n_reset, n_seg, device=self.dev))
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)

            piston = signs * (
                self._init_p_mean_m
                + torch.randn(n_reset, n_seg, device=self.dev) * self._init_p_std_m)
            if self._init_p_clip_m > 0.0:
                piston = piston.clamp(-self._init_p_clip_m, self._init_p_clip_m)

            if self._command_tip_tilt and self._init_t_std_arcsec > 0.0:
                tip_signs = torch.sign(torch.randn(n_reset, n_seg, device=self.dev))
                tip_signs = torch.where(tip_signs == 0, torch.ones_like(tip_signs), tip_signs)
                tip = tip_signs * (
                    self._init_t_std_arcsec
                    + torch.randn(n_reset, n_seg, device=self.dev)
                    * self._init_t_std_arcsec / 3.0
                ) * math.pi / (180 * 3600)
            else:
                tip = torch.zeros(n_reset, n_seg, device=self.dev)

            if self._command_tip_tilt and self._init_tl_std_arcsec > 0.0:
                tilt_signs = torch.sign(torch.randn(n_reset, n_seg, device=self.dev))
                tilt_signs = torch.where(tilt_signs == 0, torch.ones_like(tilt_signs), tilt_signs)
                tilt = tilt_signs * (
                    self._init_tl_std_arcsec
                    + torch.randn(n_reset, n_seg, device=self.dev)
                    * self._init_tl_std_arcsec / 3.0
                ) * math.pi / (180 * 3600)
            else:
                tilt = torch.zeros(n_reset, n_seg, device=self.dev)

            actuators = torch.cat([piston, tip, tilt], dim=1)
            self._actuators_t[env_mask] = actuators

        elif self._init_diff_motion and self._model_wind:
            # --- Standard mode ---
            wl = self._wl
            piston_mean = wl / 4.0
            piston_std = wl / 6.0
            clip_m = self._init_piston_clip_m

            # Random ±sign per segment per env
            signs = torch.sign(torch.randn(n_reset, n_seg, device=self.dev))
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)

            # Piston: signs * (mean + randn * std), clipped
            piston = signs * (piston_mean + torch.randn(n_reset, n_seg, device=self.dev) * piston_std)
            piston = piston.clamp(-clip_m, clip_m)

            # Tip/tilt
            if self._command_tip_tilt and self._init_tip_std > 0:
                tip_signs = torch.sign(torch.randn(n_reset, n_seg, device=self.dev))
                tip_signs = torch.where(tip_signs == 0, torch.ones_like(tip_signs), tip_signs)
                tip = tip_signs * (
                    self._init_tip_std + torch.randn(n_reset, n_seg, device=self.dev) * self._init_tip_std / 3.0
                ) * math.pi / (180 * 3600)
            else:
                tip = torch.zeros(n_reset, n_seg, device=self.dev)

            if self._command_tip_tilt and self._init_tilt_std > 0:
                tilt_signs = torch.sign(torch.randn(n_reset, n_seg, device=self.dev))
                tilt_signs = torch.where(tilt_signs == 0, torch.ones_like(tilt_signs), tilt_signs)
                tilt = tilt_signs * (
                    self._init_tilt_std + torch.randn(n_reset, n_seg, device=self.dev) * self._init_tilt_std / 3.0
                ) * math.pi / (180 * 3600)
            else:
                tilt = torch.zeros(n_reset, n_seg, device=self.dev)

            # Actuator layout: [p0, p1, ..., t0, t1, ..., tl0, tl1, ...]
            actuators = torch.cat([piston, tip, tilt], dim=1)  # [n_reset, n_modes]
            self._actuators_t[env_mask] = actuators

        # Store baselines
        self._baselines_t[env_mask] = self._actuators_t[env_mask].clone()

        # Reset step counts and episode tracking
        self._step_counts[env_mask] = 0
        self._prior_actions[env_mask] = 0.0
        self._episode_returns[env_mask] = 0.0
        self._episode_lengths[env_mask] = 0

    def reset(self, seed=None, options=None):
        """Reset all environments.

        seed can be an int (seeds all envs identically) or a list of ints
        (one per env, used for reproducible per-env resets).  Per-env seeds
        are applied via numpy only (torch global seed uses the first).
        """
        if seed is not None:
            if isinstance(seed, (list, tuple)):
                torch.manual_seed(seed[0])
                np.random.seed(seed[0])
            else:
                torch.manual_seed(seed)
                np.random.seed(seed)

        self._init_actuators(env_mask=None)

        # Run one batched pipeline to produce initial observations
        frames = self._batched_simulate()
        frames = self._batched_detector_model(frames)

        # Update observation history
        self._obs_history[:, :, :, :] = 0.0
        self._obs_history[:, -1, :, :] = frames

        obs = self._obs_history.cpu().numpy()
        infos = {}
        return obs, infos

    # =================================================================
    # Step
    # =================================================================

    def step(self, actions):
        """Step all environments with batched GPU computation.

        actions: np.ndarray [N, action_dim] in [-1, 1]
        Returns: (obs, rewards, terminateds, truncateds, infos)
        """
        N = self._num_envs

        # --- 1. Actions to GPU ---
        actions_np = np.asarray(actions, dtype=np.float32)
        actions_t = torch.from_numpy(actions_np).to(self.dev)

        # Deadzone
        if self._minimum_absolute_action > 0:
            actions_t = torch.where(
                actions_t.abs() < self._minimum_absolute_action,
                torch.zeros_like(actions_t), actions_t)

        # Bootstrap hard DOF mask: zero out every non-target action DOF.
        # Applied here so _prior_actions, _batched_command, and every
        # downstream consumer see only the masked action.
        if self._bootstrap_action_mask_t is not None:
            actions_t = actions_t * self._bootstrap_action_mask_t

        # Store raw action for penalty computation
        raw_actions_t = actions_t.clone()

        # Scale
        actions_t = actions_t * self._action_scale

        # --- 2. Apply commands (batched on GPU) ---
        self._batched_command(actions_t)

        # --- 2a. Non-stationary disturbance: random walk on the
        # secondaries' PTT state. Adds scaled randn to actuators each
        # step and reclips to [baseline ± max_corr]. Inert when the
        # multiplier is 0 (the default).
        if self._ns_std_t is not None and self._ns_multiplier > 0.0:
            noise = torch.randn_like(self._actuators_t) * self._ns_std_t
            self._actuators_t = self._actuators_t + noise * self._ns_multiplier
            low = self._baselines_t - self._max_corr_t.unsqueeze(0)
            high = self._baselines_t + self._max_corr_t.unsqueeze(0)
            self._actuators_t = torch.clamp(self._actuators_t, low, high)

        # --- 3. Batched simulate + object convolution + detector ---
        frames = self._batched_simulate()
        frames = self._batched_detector_model(frames)

        # --- 4. Update observation history ---
        if self._obs_window > 1:
            self._obs_history[:, :-1, :, :] = self._obs_history[:, 1:, :, :].clone()
        self._obs_history[:, -1, :, :] = frames

        # --- 5. Batched reward ---
        self._prior_actions = raw_actions_t
        rewards_t = self._batched_reward(frames)

        # --- 6. Episode management ---
        self._step_counts += 1
        self._episode_returns += rewards_t
        self._episode_lengths += 1

        truncated_mask = (self._step_counts >= self._max_episode_steps)
        terminated_mask = torch.zeros(N, dtype=torch.bool, device=self.dev)
        done_mask = truncated_mask | terminated_mask

        # --- 7. Build infos (on GPU, transfer at the end) ---
        strehl_np = self._last_strehl.cpu().numpy()
        oob_np = self._oob_frac.cpu().numpy()
        raw_reward_np = self._raw_reward.cpu().numpy()

        # --- 8. Auto-reset done envs ---
        infos = {
            "strehl": strehl_np,
            "mse": np.zeros(N, dtype=np.float32),  # placeholder (not computed in V5)
            "oob_frac": oob_np,
            "reward_raw": raw_reward_np,
        }
        # Vector reward: expose raw, unscaled "higher-is-better" components
        # so callers can pick (or re-weight) the optimization signal.
        if self._reward_vector_enabled and self._reward_components_t:
            comps = {}
            for name, t in self._reward_components_t.items():
                if t is None:
                    continue
                comps[name] = t.detach().cpu().numpy().astype(np.float32)
            infos["reward_components"] = comps

        if done_mask.any():
            # Store final info for done episodes
            done_indices = done_mask.nonzero(as_tuple=True)[0]
            final_returns = self._episode_returns[done_mask].cpu().numpy()
            final_lengths = self._episode_lengths[done_mask].cpu().numpy()

            final_infos = []
            for i, idx in enumerate(done_indices.cpu().numpy()):
                final_infos.append({
                    "episode": {
                        "r": float(final_returns[i]),
                        "l": int(final_lengths[i]),
                    }
                })

            # Auto-reset done envs
            self._init_actuators(env_mask=done_mask)

            # Run pipeline for reset envs to get initial observation
            reset_frames = self._batched_simulate()
            reset_frames = self._batched_detector_model(reset_frames)
            # Only update the reset envs' observation history
            self._obs_history[done_mask] = 0.0
            self._obs_history[done_mask, -1, :, :] = reset_frames[done_mask]

            # Build final_info in gymnasium format
            infos["final_info"] = [None] * N
            infos["_final_info"] = np.zeros(N, dtype=bool)
            for i, idx in enumerate(done_indices.cpu().numpy()):
                infos["final_info"][idx] = final_infos[i]
                infos["_final_info"][idx] = True

            # Final observation is the pre-reset frame
            infos["final_observation"] = [None] * N
            infos["_final_observation"] = np.zeros(N, dtype=bool)

        # --- 9. Single CPU sync ---
        obs = self._obs_history.cpu().numpy()
        rewards_np = rewards_t.cpu().numpy()
        terminated_np = terminated_mask.cpu().numpy()
        truncated_np = truncated_mask.cpu().numpy()

        return obs, rewards_np, terminated_np, truncated_np, infos

    # =================================================================
    # Gymnasium VectorEnv interface
    # =================================================================

    def close(self):
        pass

    # num_envs is provided by VectorEnv.__init__ — no override needed
