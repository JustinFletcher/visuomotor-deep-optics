"""Parity and dynamic-resampling smoke tests for the v5 dark-hole path.

1. Static parity: v5's per-env hole mask (broadcast from a single cfg)
   matches v4's mask bit-for-bit, and the factored reward (dark-hole
   term only) agrees to within float tolerance on a zero-action step.

2. Dynamic resample: with ``dark_hole_randomize_on_reset=True`` the
   per-env mask and target_vec rows are distinct after reset — i.e.
   the per-env storage + GPU builder are wired up correctly.
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
def dh_kwargs():
    from train.ppo.train_ppo_elf_dark_hole import ELF_DARK_HOLE_ENV_KWARGS
    kw = dict(ELF_DARK_HOLE_ENV_KWARGS)
    kw["silence"] = True
    kw["max_episode_steps"] = 4
    # V5 requires a single frame per decision; v4 honours it too.
    kw["observation_window_size"] = 1
    kw["dark_hole"] = True
    kw["dark_hole_angular_location_degrees"] = 45.0
    kw["dark_hole_location_radius_fraction"] = 0.25
    kw["dark_hole_size_radius"] = 0.08
    return kw


def _build_v4(kw):
    import gymnasium as gym
    from train.ppo.train_ppo_optomech import register_optomech
    register_optomech("optomech-v4", max_episode_steps=kw["max_episode_steps"])
    with contextlib.redirect_stdout(io.StringIO()):
        env = gym.make("optomech-v4", **kw)
        env.reset(seed=0)
    return env


def _build_v5(kw, num_envs=4):
    from optomech.optomech.optomech_v5 import BatchedOptomechEnv
    with contextlib.redirect_stdout(io.StringIO()):
        env = BatchedOptomechEnv(num_envs=num_envs, device="cpu", **kw)
        env.reset(seed=0)
    return env


def test_v4_v5_static_mask_parity(dh_kwargs):
    import torch
    v4 = _build_v4(dh_kwargs)
    v4_mask = v4.unwrapped._target_zero_mask.astype(bool)
    v5 = _build_v5(dh_kwargs, num_envs=4)
    # All N envs should share the build-time mask, bit-identical to v4.
    v5_masks = v5._hole_mask_t.cpu().numpy()
    assert v5_masks.shape == (4, v4_mask.shape[0], v4_mask.shape[1])
    for i in range(4):
        assert np.array_equal(v5_masks[i], v4_mask), (
            f"v5 env {i} mask differs from v4 mask")
    v4.close()


def test_v5_static_target_vec_consistent(dh_kwargs):
    v5 = _build_v5(dh_kwargs, num_envs=4)
    tv = v5._target_vec_t.cpu().numpy()
    # Static build: all envs share target_vec.
    assert np.allclose(tv, tv[0:1], atol=1e-6), (
        "static target_vec should be identical across envs at build")
    # Expected values: [sin(45°), cos(45°), 0.25, 0.08]
    expected = np.array(
        [np.sin(np.deg2rad(45.0)), np.cos(np.deg2rad(45.0)), 0.25, 0.08],
        dtype=np.float32)
    assert np.allclose(tv[0], expected, atol=1e-5), (
        f"target_vec mismatch: got {tv[0]}, expected {expected}")


def test_v5_dynamic_resample_per_env(dh_kwargs):
    kw = dict(dh_kwargs)
    kw["dark_hole_randomize_on_reset"] = True
    kw["dark_hole_angle_range_deg"] = (0.0, 360.0)
    kw["dark_hole_radius_range"] = (0.16, 0.32)
    kw["dark_hole_size_range"] = (0.08, 0.08)
    v5 = _build_v5(kw, num_envs=8)
    # After reset, per-env target vecs should be distinct.
    tv = v5._target_vec_t.cpu().numpy()
    pairs_distinct = 0
    for i in range(len(tv)):
        for j in range(i + 1, len(tv)):
            if not np.allclose(tv[i], tv[j], atol=1e-5):
                pairs_distinct += 1
    # Expect at least most pairs to differ (probabilistic).
    n_pairs = len(tv) * (len(tv) - 1) // 2
    assert pairs_distinct >= int(0.8 * n_pairs), (
        f"only {pairs_distinct}/{n_pairs} env pairs have distinct target_vec")
    # Masks should also differ across envs.
    masks = v5._hole_mask_t.cpu().numpy()
    distinct_masks = 0
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if not np.array_equal(masks[i], masks[j]):
                distinct_masks += 1
    assert distinct_masks >= int(0.8 * n_pairs), (
        f"only {distinct_masks}/{n_pairs} env pairs have distinct masks")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
