"""
Numerical parity test: V3 (CPU) vs V4 (GPU).

Tests that V4's GPU-accelerated optical propagation produces identical
observations and rewards as V3, given identical actuator states.

Strategy:
  1. Create both V3 and V4 envs.
  2. Reset V3 (which seeds its own RNG).
  3. Copy V3's full actuator + baseline state to V4.
  4. Run the inner optical pipeline (simulate + get_science_frame)
     for both, with the same actuator states, and compare outputs.
  5. Also run full step() comparison by syncing state before each step.
"""
import sys
import copy
import numpy as np
import torch


# --- Shared env kwargs (nanoelf, tip-tilt, minimal config for speed) ---
ENV_KWARGS = {
    "aperture_type": "nanoelf",
    "focal_plane_image_size_pixels": 128,
    "command_secondaries": True,
    "command_tip_tilt": True,
    "action_type": "none",
    "observation_mode": "image_only",
    "observation_window_size": 1,
    "bandwidth_sampling": 2,
    "bandwidth_nanometers": 200.0,
    "init_differential_motion": True,
    "model_wind_diff_motion": True,
    "init_wind_piston_micron_std": 3.0,
    "init_wind_tip_arcsec_std_tt": 0.5,
    "init_wind_tilt_arcsec_std_tt": 0.5,
    "max_episode_steps": 10,
    "reward_function": "composite",
    "reward_weight_strehl": 0.0,
    "reward_weight_centered_strehl": 1.0,
    "holding_bonus_weight": 1.0,
    "holding_bonus_min_reward": -1.0,
    "holding_bonus_threshold": -0.7,
    "actuator_noise": False,  # deterministic for comparison
    "silence": True,
}


def make_env(version):
    """Create and return a single OptomechEnv."""
    if version == "v3":
        from optomech.optomech.optomech_v3 import OptomechEnv
    elif version == "v4":
        from optomech.optomech.optomech_v4 import OptomechEnv
    else:
        raise ValueError(f"Unknown version: {version}")
    kwargs = {**ENV_KWARGS, "optomech_version": version}
    if version == "v4":
        kwargs["device"] = "cpu"  # force CPU for numerical comparison
    env = OptomechEnv(**kwargs)
    return env


def _centered_wavelengths(center_wl, bandwidth_nm, n_samples):
    """Compute centered wavelength samples across a bandwidth."""
    bw_m = bandwidth_nm / 1e9
    if n_samples <= 1:
        return [center_wl]
    bin_width = bw_m / n_samples
    first = center_wl - bw_m / 2.0 + bin_width / 2.0
    return [first + i * bin_width for i in range(n_samples)]


def sync_v4_from_v3(env_v3, env_v4):
    """Copy V3's actuator/baseline/wind state into V4."""
    os3 = env_v3.optical_system
    os4 = env_v4.optical_system

    # Copy actuator state
    act = np.array(os3.segmented_mirror.actuators).copy()
    os4.segmented_mirror.actuators = act.copy()
    os4._actuators_t = torch.tensor(act, dtype=torch.float32)

    # Copy baselines and wind state
    if hasattr(os3, '_baselines'):
        os4._baselines = copy.deepcopy(os3._baselines)
    if hasattr(os3, 'wind_ptt_state'):
        os4.wind_ptt_state = copy.deepcopy(os3.wind_ptt_state)
    if hasattr(os3, 'current_ground_wind_speed_mps'):
        os4.current_ground_wind_speed_mps = os3.current_ground_wind_speed_mps

    # Copy episode time
    env_v4.episode_time_ms = env_v3.episode_time_ms

    # Copy reference flux values
    if hasattr(env_v3, '_reference_fpi_sum'):
        env_v4._reference_fpi_sum = env_v3._reference_fpi_sum
    if hasattr(env_v3, '_reference_fpi_max'):
        env_v4._reference_fpi_max = env_v3._reference_fpi_max
    if hasattr(os3, '_reference_fpi_sum'):
        os4._reference_fpi_sum = os3._reference_fpi_sum


def test_simulate_parity():
    """Test: V4 simulate() matches V3 for various actuator states."""
    print("\n--- Test 1: simulate() parity ---")
    from optomech.optomech.optomech_v3 import OptomechEnv as V3
    from optomech.optomech.optomech_v4 import OptomechEnv as V4

    kwargs3 = {**ENV_KWARGS, "optomech_version": "v3"}
    kwargs4 = {**ENV_KWARGS, "optomech_version": "v4", "device": "cpu"}
    env3 = V3(**kwargs3)
    env4 = V4(**kwargs4)
    os3, os4 = env3.optical_system, env4.optical_system

    wavelengths = _centered_wavelengths(
        os3.wavelength, ENV_KWARGS['bandwidth_nanometers'],
        ENV_KWARGS['bandwidth_sampling'])

    # Test multiple actuator configs
    test_configs = [
        np.zeros(6),                                    # aligned
        np.array([1e-6, -2e-6, 5e-5, -3e-5, 1e-4, -8e-5]),  # misaligned
        np.array([3e-6, 3e-6, 0, 0, 0, 0]),            # piston only
        np.array([0, 0, 1e-4, -1e-4, 0, 0]),           # tip only
    ]

    all_pass = True
    for ci, act in enumerate(test_configs):
        for wl in wavelengths:
            os3.segmented_mirror.actuators = act.copy()
            os4.segmented_mirror.actuators = act.copy()
            os4._actuators_t = torch.tensor(act, dtype=torch.float32)

            os3.simulate(wl)
            os4.simulate(wl)

            v3_E = np.array(os3.focal_plane_wavefront.electric_field).reshape(128, 128)
            v4_E = os4._focal_field_t.numpy()

            # Compare electric field
            max_abs = np.max(np.abs(v3_E - v4_E))
            max_ref = max(np.max(np.abs(v3_E)), 1e-30)
            rel = max_abs / max_ref

            # Compare intensity (PSF without object convolution)
            # V3: |E|^2 * gw * dt via camera
            dt = 0.001
            os3.camera.integrate(os3.focal_plane_wavefront, dt)
            v3_psf = np.array(os3.camera.read_out()).reshape(128, 128)
            # V4: |E|^2 * gw * dt directly
            v4_psf = (torch.abs(os4._focal_field_t) ** 2
                      * os4._focal_grid_weight * dt).numpy()

            psf_diff = np.max(np.abs(v3_psf - v4_psf))
            psf_ref = max(np.max(v3_psf), 1e-30)
            psf_rel = psf_diff / psf_ref

            ok = rel < 1e-4 and psf_rel < 1e-4
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"  Config {ci} wl={wl*1e9:.0f}nm: "
                  f"|E| rel={rel:.2e}  PSF rel={psf_rel:.2e}  [{status}]")

    return all_pass


def test_full_step_parity():
    """Test: V4 step() output matches V3 when given identical state."""
    print("\n--- Test 2: Full step() parity ---")
    from optomech.optomech.optomech_v3 import OptomechEnv as V3
    from optomech.optomech.optomech_v4 import OptomechEnv as V4

    kwargs3 = {**ENV_KWARGS, "optomech_version": "v3"}
    kwargs4 = {**ENV_KWARGS, "optomech_version": "v4", "device": "cpu"}
    env3 = V3(**kwargs3)
    env4 = V4(**kwargs4)

    # Reset V3 normally; V4 will be force-synced
    np.random.seed(42)
    obs3, _ = env3.reset(seed=42)

    # Reset V4 with its own seed (we'll override state)
    np.random.seed(42)
    obs4, _ = env4.reset(seed=42)

    # Sync V4 state from V3
    sync_v4_from_v3(env3, env4)

    # Now run steps with same actions
    all_pass = True
    np.random.seed(1000)
    for step_i in range(5):
        action = env3.action_space.sample()

        # Sync before step
        sync_v4_from_v3(env3, env4)

        obs3, r3, term3, trunc3, info3 = env3.step(action.copy())
        obs4, r4, term4, trunc4, info4 = env4.step(action.copy())

        if obs3.shape != obs4.shape:
            print(f"  Step {step_i}: SHAPE MISMATCH {obs3.shape} vs {obs4.shape}")
            all_pass = False
            continue

        abs_diff = np.abs(obs3.astype(np.float64) - obs4.astype(np.float64))
        max_abs = float(np.max(abs_diff))
        mean_abs = float(np.mean(abs_diff))

        denom = np.maximum(np.abs(obs3.astype(np.float64)), 1e-10)
        rel_diff = abs_diff / denom
        max_rel = float(np.max(rel_diff))

        rew_diff = abs(r3 - r4)

        obs_ok = max_rel < 1e-3
        rew_ok = rew_diff < 1e-3
        ok = obs_ok and rew_ok
        if not ok:
            all_pass = False

        status = "PASS" if ok else "FAIL"
        print(f"  Step {step_i}: obs_rel={max_rel:.2e}  obs_abs={max_abs:.2e}  "
              f"rew_diff={rew_diff:.2e}  [{status}]")

    return all_pass


def test_science_frame_convolution():
    """Test: V4 object convolution matches V3."""
    print("\n--- Test 3: Science frame convolution parity ---")
    from optomech.optomech.optomech_v3 import OptomechEnv as V3
    from optomech.optomech.optomech_v4 import OptomechEnv as V4

    kwargs3 = {**ENV_KWARGS, "optomech_version": "v3"}
    kwargs4 = {**ENV_KWARGS, "optomech_version": "v4", "device": "cpu"}
    env3 = V3(**kwargs3)
    env4 = V4(**kwargs4)
    os3, os4 = env3.optical_system, env4.optical_system

    # Set aligned actuators
    zeros = np.zeros(6)
    os3.segmented_mirror.actuators = zeros.copy()
    os4.segmented_mirror.actuators = zeros.copy()
    os4._actuators_t = torch.tensor(zeros, dtype=torch.float32)

    # Use first bandwidth-sampled wavelength (which is in MFT cache)
    wl = _centered_wavelengths(
        os3.wavelength, ENV_KWARGS['bandwidth_nanometers'],
        ENV_KWARGS['bandwidth_sampling'])[0]
    dt = 0.001

    # V3 full pipeline
    os3.simulate(wl)
    v3_frame = os3.get_science_frame(integration_seconds=dt)

    # V4 full pipeline
    os4.simulate(wl)
    v4_frame = os4.get_science_frame(integration_seconds=dt)
    if isinstance(v4_frame, torch.Tensor):
        v4_frame = v4_frame.numpy()

    max_abs = np.max(np.abs(v3_frame - v4_frame))
    max_ref = max(np.max(np.abs(v3_frame)), 1e-30)
    rel = max_abs / max_ref

    ok = rel < 1e-4
    status = "PASS" if ok else "FAIL"
    print(f"  Science frame: rel={rel:.2e}  abs={max_abs:.2e}  [{status}]")
    return ok


def main():
    print("=" * 60)
    print("V3 vs V4 Numerical Parity Test")
    print("=" * 60)

    results = []

    results.append(("simulate()", test_simulate_parity()))
    results.append(("science_frame", test_science_frame_convolution()))
    results.append(("full_step()", test_full_step_parity()))

    print("\n" + "=" * 60)
    all_pass = all(r for _, r in results)
    for name, passed in results:
        print(f"  {name:20s}  {'PASS' if passed else 'FAIL'}")
    print("=" * 60)
    if all_pass:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
