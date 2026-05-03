"""Parity tests for the v5 DM port against v4's HCIPy DM forward.

Three checks:

1. ``test_dm_basis_parity``: v5's GPU influence-function basis is
   bit-identical to v4's (HCIPy ModeBasis transferred through the same
   shared init path).

2. ``test_dm_surface_parity``: For a non-trivial DM actuator pattern
   (in meters), v5's matmul-built OPD surface matches v4's
   ``dm.surface`` to float tolerance.

3. ``test_dm_step_changes_obs``: A v5 step with a random DM action
   measurably perturbs the detector frame relative to a zero-action
   step. Sanity check that the simulate path picks up DM state.

The action-space layout for v5 with ``command_dm=True`` is
``[seg_actions ..., dm_actions ...]`` of size
``num_apertures * n_dof_per_seg + n_dm_acts``.
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(scope="module")
def dm_kwargs():
    from train.ppo.train_ppo_elf_dark_hole import ELF_DARK_HOLE_ENV_KWARGS
    kw = dict(ELF_DARK_HOLE_ENV_KWARGS)
    kw["silence"] = True
    kw["max_episode_steps"] = 4
    kw["observation_window_size"] = 1
    kw["dark_hole"] = True
    kw["dark_hole_angular_location_degrees"] = 0.0
    kw["dark_hole_location_radius_fraction"] = 0.16
    kw["dark_hole_size_radius"] = 0.095
    kw["command_dm"] = True
    # Small actuator count keeps the test fast; the math is identical
    # for any size.
    kw["dm_num_actuators_across"] = 9
    kw["actuator_noise"] = False
    kw["rail_baseline_random"] = False
    kw["init_differential_motion"] = False
    kw["init_differential_motion_configurable"] = False
    kw["env_action_scale"] = 1.0
    return kw


def _build_v4(kw):
    """Build a v4 env directly (not via gym.make so we can poke the
    optical_system attributes for parity inspection)."""
    from optomech.optomech.optomech_v4 import OptomechEnv
    v4_kw = dict(kw)
    v4_kw["optomech_version"] = "v4"
    v4_kw["device"] = "cpu"
    with contextlib.redirect_stdout(io.StringIO()):
        env = OptomechEnv(**v4_kw)
    return env


def _build_v5(kw, num_envs=2):
    from optomech.optomech.optomech_v5 import BatchedOptomechEnv
    with contextlib.redirect_stdout(io.StringIO()):
        env = BatchedOptomechEnv(num_envs=num_envs, device="cpu", **kw)
        env.reset(seed=0)
    return env


def test_dm_basis_parity(dm_kwargs):
    """v5's _dm_basis_t [A, H, W] equals v4's HCIPy influence functions."""
    v4 = _build_v4(dm_kwargs)
    v5 = _build_v5(dm_kwargs, num_envs=2)
    os4 = v4.optical_system
    H, W = v5._H, v5._W

    v4_basis = np.stack(
        [np.array(m).reshape(H, W) for m in os4.dm.influence_functions],
        axis=0)
    v5_basis = v5._dm_basis_t.cpu().numpy()
    assert v5_basis.shape == v4_basis.shape, (
        f"shape mismatch: v5 {v5_basis.shape} vs v4 {v4_basis.shape}")
    assert np.allclose(v5_basis, v4_basis, atol=1e-6), (
        "v5 _dm_basis_t differs from v4 dm.influence_functions")
    assert v5._n_dm_acts == v4_basis.shape[0]
    v4.close()


def test_dm_surface_parity(dm_kwargs):
    """v5's matmul-built DM OPD surface matches HCIPy's dm.surface."""
    import torch
    rng = np.random.default_rng(0)
    v4 = _build_v4(dm_kwargs)
    v5 = _build_v5(dm_kwargs, num_envs=2)
    os4 = v4.optical_system
    A = v5._n_dm_acts

    # Random actuator state in meters (within stroke limit).
    a_meters = rng.uniform(
        -v5._dm_stroke_limit_m, v5._dm_stroke_limit_m, size=A
    ).astype(np.float32)

    # v4 / HCIPy: set actuators directly (in meters), read .surface.
    os4.dm.actuators = a_meters
    v4_surface = np.array(os4.dm.surface).reshape(v5._H, v5._W)

    # v5: place same actuator state on env 0 and compute via the same
    # path that _batched_simulate uses.
    v5._dm_actuators_t.zero_()
    v5._dm_actuators_t[0] = torch.tensor(a_meters, device=v5.dev)
    flat = torch.matmul(v5._dm_actuators_t, v5._dm_basis_t_flat)
    v5_surface = flat.reshape(2, v5._H, v5._W).cpu().numpy()[0]

    assert v5_surface.shape == v4_surface.shape
    # HCIPy stores in float64; we operate in float32. Tolerance scales
    # with stroke limit (typically ~1.5e-6 m), so 1e-9 m is ample.
    err = float(np.abs(v5_surface - v4_surface).max())
    assert err < 1e-9, (
        f"v5 DM surface differs from v4 by max {err:.3e} m "
        f"(stroke_limit={v5._dm_stroke_limit_m:.3e} m)")
    v4.close()


def test_dm_step_changes_obs(dm_kwargs):
    """Stepping with a non-zero DM action perturbs the observation."""
    v5 = _build_v5(dm_kwargs, num_envs=2)
    A = v5.single_action_space.shape[0]
    n_seg = v5._n_seg_actions
    n_dm = v5._n_dm_acts
    assert A == n_seg + n_dm, (
        f"action space {A} != n_seg ({n_seg}) + n_dm ({n_dm})")

    # Zero-action step (DM stays at zero).
    obs_zero = v5.step(np.zeros((v5.num_envs, A), dtype=np.float32))[0].copy()

    # Random DM action, segments at zero. Use a moderate amplitude that
    # warps the PSF without saturating into pure noise.
    rng = np.random.default_rng(42)
    act = np.zeros((v5.num_envs, A), dtype=np.float32)
    act[:, n_seg:] = rng.uniform(-0.3, 0.3, (v5.num_envs, n_dm)).astype(np.float32)
    obs_dm = v5.step(act)[0].copy()

    diff = np.abs(obs_dm.astype(np.float64) - obs_zero.astype(np.float64))
    assert diff.max() > 10.0, (
        f"DM step did not perturb obs (max diff {diff.max():.3e} DN); "
        f"DM forward path may be inert")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
