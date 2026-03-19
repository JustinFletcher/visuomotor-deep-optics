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
        assert env_kwargs.get("incremental_control", True), \
            "V5 requires incremental_control=True"
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
            noise_std.append(self._noise_frac * 2.0 * self._max_p_m)
        for _ in range(self._num_apertures):
            noise_std.append(self._noise_frac * 2.0 * self._max_t_r)
        for _ in range(self._num_apertures):
            noise_std.append(self._noise_frac * 2.0 * self._max_tl_r)
        self._noise_std_t = torch.tensor(noise_std, dtype=torch.float32, device=self.dev)

        # Action penalty / holding bonus config
        self._action_penalty = cfg.get("action_penalty", False)
        self._action_penalty_weight = cfg.get("action_penalty_weight", 0.03)
        self._holding_bonus_weight = cfg.get("holding_bonus_weight", 0.0)
        self._holding_bonus_min_reward = cfg.get("holding_bonus_min_reward", -1.0)
        self._holding_bonus_threshold = cfg.get("holding_bonus_threshold", 0.0)
        self._minimum_absolute_action = cfg.get("minimum_absolute_action", 0.0)

        # Reward function config
        self._reward_function = cfg.get("reward_function", "factored")
        self._rw_centered_strehl = cfg.get("reward_weight_centered_strehl", 0.0)
        self._rw_strehl = cfg.get("reward_weight_strehl", 0.0)
        self._rw_centering = cfg.get("reward_weight_centering", 0.0)
        self._rw_flux = cfg.get("reward_weight_flux", 0.0)
        self._rw_peak = cfg.get("reward_weight_peak", 0.0)

        # Centering geometry (precomputed on GPU)
        cy, cx = self._H / 2.0, self._W / 2.0
        self._cy = cy
        self._cx = cx
        self._max_dist = math.sqrt(cy ** 2 + cx ** 2)

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
        self._wl = os4.wavelength

        # Observation window size (for stacking frames)
        self._obs_window = cfg.get("observation_window_size", 1)

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

        # Discard V4 env
        v4.close()
        del v4, os4

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
        # Transpose to [N, n_dof, n_seg] then flatten to [N, n_modes]
        action_reordered = action_reshaped.permute(0, 2, 1).reshape(N, -1)

        # Scale action to physical units
        delta = action_reordered * self._max_corr_t.unsqueeze(0)

        # Incremental: actuators += delta
        pre_clip = self._actuators_t + delta

        # Clip to [baseline - max, baseline + max]
        low = self._baselines_t - self._max_corr_t.unsqueeze(0)
        high = self._baselines_t + self._max_corr_t.unsqueeze(0)
        post_clip = torch.clamp(pre_clip, low, high)

        # Track OOB fraction
        clipped_mask = (pre_clip != post_clip)  # [N, n_modes]
        if self._command_tip_tilt:
            n_active = self._n_modes  # all DOFs active
        else:
            n_active = self._num_apertures  # piston only
            clipped_mask = clipped_mask[:, :n_active]
        self._oob_frac = clipped_mask.float().mean(dim=1)  # [N]
        self._clipped_count = clipped_mask.sum(dim=1)  # [N]
        self._total_dof = n_active

        # Rail noise: add noise where clipped
        if self._actuator_noise or True:  # rail noise always applies to clipped DOFs
            noise = torch.randn_like(post_clip) * self._noise_std_t.unsqueeze(0)
            post_clip = torch.where(clipped_mask, post_clip + noise, post_clip)

        # Actuator repeatability noise on non-zero commands
        if self._actuator_noise:
            cmd_nonzero = (action_reordered != 0.0)
            rep_noise = torch.randn_like(post_clip) * self._noise_std_t.unsqueeze(0)
            post_clip = torch.where(cmd_nonzero, post_clip + rep_noise, post_clip)

        self._actuators_t = post_clip

    def _batched_reward(self, frames):
        """Compute batched reward on GPU.

        frames: [N, H, W] detector output in DN.
        Returns: [N] reward tensor on GPU.
        """
        N = self._num_envs
        H, W = self._H, self._W

        total = torch.zeros(N, dtype=torch.float32, device=self.dev)

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

        # --- Reward components (factored) ---
        if self._rw_centered_strehl > 0:
            cs = -(1.0 - strehl * centering)
            total += self._rw_centered_strehl * cs

        if self._rw_strehl > 0:
            total += self._rw_strehl * (-(1.0 - strehl))

        if self._rw_centering > 0:
            # Use centering_energy if weight map exists
            if self._centering_weight_t is not None:
                weighted = (frames * self._centering_weight_t.unsqueeze(0)).sum(dim=(1, 2))
                cen_val = (weighted / self._reference_centering_sum).clamp(max=1.0)
            else:
                cen_val = centering
            total += self._rw_centering * (-(1.0 - cen_val))

        if self._rw_flux > 0:
            total += self._rw_flux * (-(1.0 - flux_frac))

        if self._rw_peak > 0:
            peak_ratio = (fpi_max / self._reference_fpi_max).clamp(max=1.0)
            total += self._rw_peak * (-(1.0 - peak_ratio))

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

            action_l1 = self._prior_actions.abs().mean(dim=1)  # [N]
            stillness = 1.0 - action_l1
            total = total + self._holding_bonus_weight * quality * stillness

        # --- Action penalty ---
        if self._action_penalty:
            action_l1 = self._prior_actions.abs().mean(dim=1)  # [N]
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

        # Start from zero actuators
        self._actuators_t[env_mask] = 0.0

        if self._init_diff_motion and self._model_wind:
            wl = self._wl
            piston_mean = wl / 4.0
            piston_std = wl / 6.0
            clip_m = self._init_piston_clip_m
            n_seg = self._num_apertures

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
        """Reset all environments."""
        if seed is not None:
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

        # Store raw action for penalty computation
        raw_actions_t = actions_t.clone()

        # Scale
        actions_t = actions_t * self._action_scale

        # --- 2. Apply commands (batched on GPU) ---
        self._batched_command(actions_t)

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
