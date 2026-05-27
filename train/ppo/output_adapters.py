"""Output adapters for composite policies with heterogeneous action spaces.

A composite policy can mix sub-policies that were trained against
different action representations -- e.g. a segment-PTT policy
(45 raw DOFs) and a Zernike-based DM policy (64 Zernike coefficients
projected to ~1225 DM actuators). At rollout time those sub-policies
all run inside ONE env whose action space is the union of every
phase's needs. Each phase's native output is then "adapted" -- placed
in the right slice of the env's combined action vector, projected if
needed, zeros elsewhere -- before the env consumes it.

This module is the small framework for that. ``policy_spec.py``
constructs one adapter per phase from the YAML spec; the composite
agent's ``apply_action_mask`` runs the adapter and returns the
env-facing action.

Two adapters are implemented:

- ``SliceAdapter`` -- passthrough into a contiguous slice of the env
  action. Used by segment-PTT phases: their native 45-DOF output is
  placed at the segment-PTT slice, rest zero.
- ``ZernikeToDMAdapter`` -- runs the same Zernike->DM projection that
  ZernikeDMVectorEnv applies at training time (so the policy sees
  identical wavefront-side semantics in v4 rollouts as it did in v5
  training), then places the resulting DM-action block at the DM
  slice, zeros elsewhere.

Both adapters take ``[B, native_dim]`` tensors and return
``[B, env_action_dim]`` tensors on the same device. The existing
per-phase bootstrap_mask_nontarget mechanism is applied to the
NATIVE action before the adapter runs, so the mask still zeros only
the segment DOFs it knows about.
"""
from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class OutputAdapter(ABC):
    """Map a phase's native action vector into the env's combined
    action vector.

    Subclasses must set ``self.env_action_dim`` so the composite
    agent and rollout can sanity-check shape compatibility."""

    env_action_dim: int

    @abstractmethod
    def __call__(self, native_action: torch.Tensor) -> torch.Tensor:
        """[B, native_dim] -> [B, env_action_dim]."""

    def __repr__(self) -> str:
        return (f"{type(self).__name__}"
                f"(env_action_dim={self.env_action_dim})")


# ---------------------------------------------------------------------------
# SliceAdapter
# ---------------------------------------------------------------------------

class SliceAdapter(OutputAdapter):
    """Passthrough into a contiguous slice ``[start, stop)`` of the
    env action vector. All other channels are zero. Used by phases
    whose native action representation IS the env's action
    representation at the chosen slot (e.g. a segment-PTT policy
    placed at the env's segment slice).

    Native action width must equal ``stop - start`` (or
    ``env_action_dim - start`` if ``stop`` is None).
    """

    def __init__(self, start: int, stop: Optional[int],
                 env_action_dim: int):
        self.start = int(start)
        self.stop = (int(env_action_dim) if stop is None else int(stop))
        self.env_action_dim = int(env_action_dim)
        if not (0 <= self.start < self.stop <= self.env_action_dim):
            raise ValueError(
                f"SliceAdapter: invalid slice [{self.start}, {self.stop}) "
                f"into env action vector of width {self.env_action_dim}")
        self.native_dim = self.stop - self.start

    def __call__(self, native_action: torch.Tensor) -> torch.Tensor:
        if native_action.shape[-1] != self.native_dim:
            raise ValueError(
                f"SliceAdapter expects native_action of width "
                f"{self.native_dim}, got {native_action.shape[-1]}")
        batch_shape = native_action.shape[:-1]
        full = torch.zeros(*batch_shape, self.env_action_dim,
                           dtype=native_action.dtype,
                           device=native_action.device)
        full[..., self.start:self.stop] = native_action
        return full

    def __repr__(self) -> str:
        return (f"SliceAdapter([{self.start}, {self.stop}) of "
                f"{self.env_action_dim})")


# ---------------------------------------------------------------------------
# ZernikeToDMAdapter
# ---------------------------------------------------------------------------

class ZernikeToDMAdapter(OutputAdapter):
    """Project Zernike coefficients [n_z] to DM actuator commands
    [n_dm], then place them at the env's DM slice. Zeros elsewhere.

    Build via ``ZernikeToDMAdapter.from_env(...)`` to derive the
    projection M, DM slice, and env action dim straight from a v4
    env (or a v5 env with the same shape attrs). The standalone
    constructor exists for tests that want to pre-compute M.

    Projection semantics match training-time ZernikeDMVectorEnv:
    ``a_dm = clamp(M @ a_z, -1, 1)`` with M built by
    ``build_zernike_to_dm_projection``.
    """

    def __init__(self, M: torch.Tensor, dm_slice: Tuple[int, int],
                 env_action_dim: int,
                 action_scale: float = 1.0):
        self.M = M                                  # [n_dm, n_z]
        self.n_dm, self.n_zernike = M.shape
        self.dm_start, self.dm_stop = dm_slice
        self.env_action_dim = int(env_action_dim)
        # Per-adapter action scaling. Applied to the projected DM
        # vector before clamping into [-1, 1]. Matches the
        # training-time env_action_scale that was applied uniformly
        # to the env's action vector by v5 -- since the env in a
        # heterogeneous composite cannot apply a different scale
        # per phase, we apply it inside the adapter so the segment
        # phases keep their original magnitude. The dm_atmos
        # checkpoint trained with env_action_scale=0.01 should pass
        # action_scale=0.01 here.
        self.action_scale = float(action_scale)
        if (self.dm_stop - self.dm_start) != self.n_dm:
            raise ValueError(
                f"ZernikeToDMAdapter: dm_slice "
                f"[{self.dm_start}, {self.dm_stop}) width "
                f"{self.dm_stop - self.dm_start} != n_dm {self.n_dm}")
        self.native_dim = self.n_zernike

    def __call__(self, native_action: torch.Tensor) -> torch.Tensor:
        if native_action.shape[-1] != self.n_zernike:
            raise ValueError(
                f"ZernikeToDMAdapter expects native_action of width "
                f"{self.n_zernike}, got {native_action.shape[-1]}")
        # native_action [B, n_z]; matmul with M.T -> [B, n_dm].
        dm = native_action @ self.M.t()
        if self.action_scale != 1.0:
            dm = dm * self.action_scale
        dm = torch.clamp(dm, -1.0, 1.0)
        batch_shape = native_action.shape[:-1]
        full = torch.zeros(*batch_shape, self.env_action_dim,
                           dtype=native_action.dtype,
                           device=native_action.device)
        full[..., self.dm_start:self.dm_stop] = dm
        return full

    # --- construction helper ---------------------------------------------

    @classmethod
    def from_env(cls, env, n_zernike: int,
                 dm_slice: Optional[Tuple[int, int]] = None,
                 env_action_dim: Optional[int] = None,
                 action_scale: float = 1.0,
                 skip_piston: bool = False,
                 normalize: str = "wavefront_rms",
                 regularize_rcond: float = 1e-3,
                 device: Optional[torch.device] = None,
                 silence: bool = True) -> "ZernikeToDMAdapter":
        """Build the adapter for ``env``. Works for v4 and v5.

        ``dm_slice`` and ``env_action_dim`` are auto-derived from the
        env when None: the DM block is assumed to live at the END of
        the action vector, immediately after any segment/tensioner
        DOFs, with width ``n_dm_actuators``."""
        from train.ppo.zernike_dm import build_zernike_to_dm_projection

        base = env.unwrapped if hasattr(env, "unwrapped") else env

        # Extract DM influence basis as [n_dm, H*W] numpy and the
        # binary aperture mask. v5 keeps both on env directly; v4
        # tucks the HCIPy DM under optical_system.
        dm_flat = None
        aperture_mask = None
        H = W = None
        n_dm = 0

        # v5 path: flat tensor pre-staged.
        dm_basis_flat_t = getattr(base, "_dm_basis_t_flat", None)
        if dm_basis_flat_t is not None:
            dm_flat = dm_basis_flat_t.detach().cpu().numpy()
            ap = getattr(base, "_aperture_t", None)
            if ap is not None:
                ap_np = np.abs(ap.detach().cpu().numpy())
                H = int(getattr(base, "_H", ap_np.shape[0]))
                W = int(getattr(base, "_W", ap_np.shape[1]))
                aperture_mask = (ap_np > 0).reshape(H, W)
            n_dm = int(getattr(base, "_n_dm_acts", dm_flat.shape[0]))

        # v4 path: HCIPy DeformableMirror on optical_system.
        if dm_flat is None:
            opt = getattr(base, "optical_system", None)
            if opt is None:
                raise ValueError(
                    "ZernikeToDMAdapter.from_env: env has no DM info "
                    "(neither v5 _dm_basis_t_flat nor v4 "
                    "optical_system.dm).")
            dm = getattr(opt, "dm", None)
            if dm is None:
                raise ValueError(
                    "ZernikeToDMAdapter.from_env: v4 optical_system "
                    "has no .dm. Set command_dm=True in env_kwargs.")
            # HCIPy ModeBasis -> [n_dm, H*W] numpy.
            inf_list = [np.asarray(m) for m in dm.influence_functions]
            n_dm = len(inf_list)
            # Each Field flattens to H*W; infer H, W from the aperture.
            ap_np = np.asarray(opt.aperture)
            num_px = int(getattr(opt, "_num_px"))
            H = W = num_px
            aperture_mask = (np.abs(ap_np).reshape(H, W) > 0)
            dm_flat = np.stack(
                [m.reshape(-1) for m in inf_list], axis=0)   # [n_dm, H*W]

        # Default env_action_dim and dm_slice -- the DM block sits at
        # the tail of the env action vector.
        if env_action_dim is None:
            try:
                env_action_dim = int(env.action_space.shape[0])
            except Exception as e:
                raise ValueError(
                    "ZernikeToDMAdapter.from_env: env_action_dim not "
                    "provided and env.action_space is unreadable: "
                    f"{e}") from e
        if dm_slice is None:
            dm_slice = (env_action_dim - n_dm, env_action_dim)

        # Build M and ship to torch on the requested device.
        M_np = build_zernike_to_dm_projection(
            dm_flat, aperture_mask, n_zernike,
            skip_piston=skip_piston, normalize=normalize,
            regularize_rcond=regularize_rcond, silence=silence)
        if device is None:
            # Place on the same device as a representative env tensor
            # if we can find one; otherwise CPU.
            for attr in ("_dm_basis_t_flat", "_aperture_t"):
                t = getattr(base, attr, None)
                if t is not None and hasattr(t, "device"):
                    device = t.device
                    break
            else:
                opt = getattr(base, "optical_system", None)
                device = (getattr(opt, "_torch_device", torch.device("cpu"))
                          if opt is not None else torch.device("cpu"))
        M = torch.tensor(M_np, dtype=torch.float32, device=device)
        return cls(M, dm_slice, env_action_dim,
                   action_scale=action_scale)

    def __repr__(self) -> str:
        return (f"ZernikeToDMAdapter(n_z={self.n_zernike}, "
                f"n_dm={self.n_dm}, dm_slice="
                f"[{self.dm_start}, {self.dm_stop}), "
                f"env_action_dim={self.env_action_dim})")


# ---------------------------------------------------------------------------
# SegmentZernikePassthroughAdapter
# ---------------------------------------------------------------------------

class SegmentZernikePassthroughAdapter(OutputAdapter):
    """Map a policy's combined ``[seg_PTT_45 | zernike_32]`` native action
    into the env's combined ``[seg_PTT_45 | dm_1225]`` action vector.

    This is the rollout-time mirror of training-time
    ``SegmentZernikeDMVectorEnv``: same Zernike->DM projection M, same
    DM-only ``action_scale`` cap, same clamp. The segment block passes
    through unchanged; the Zernike block is projected, scaled, clamped,
    and placed at the env's DM slot.

    Native input width:  n_seg_actions + n_zernike    (e.g. 45 + 32 = 77)
    Env-facing width:    n_seg_actions + n_dm         (e.g. 45 + 1225 = 1270)
    """

    def __init__(self, M: torch.Tensor,
                 n_seg_actions: int,
                 env_action_dim: int,
                 dm_action_scale: float = 1.0):
        self.M = M                                  # [n_dm, n_z]
        self.n_dm, self.n_zernike = M.shape
        self.n_seg_actions = int(n_seg_actions)
        self.env_action_dim = int(env_action_dim)
        self.dm_action_scale = float(dm_action_scale)
        if self.n_seg_actions + self.n_dm != self.env_action_dim:
            raise ValueError(
                f"SegmentZernikePassthroughAdapter: n_seg_actions "
                f"({self.n_seg_actions}) + n_dm ({self.n_dm}) "
                f"!= env_action_dim ({self.env_action_dim})")
        self.native_dim = self.n_seg_actions + self.n_zernike

    def __call__(self, native_action: torch.Tensor) -> torch.Tensor:
        if native_action.shape[-1] != self.native_dim:
            raise ValueError(
                f"SegmentZernikePassthroughAdapter expects "
                f"native_action of width {self.native_dim}, got "
                f"{native_action.shape[-1]}")
        seg = native_action[..., : self.n_seg_actions]
        zer = native_action[..., self.n_seg_actions:]
        dm = zer @ self.M.t()
        if self.dm_action_scale != 1.0:
            dm = dm * self.dm_action_scale
        dm = torch.clamp(dm, -1.0, 1.0)
        full = torch.cat([seg, dm], dim=-1)
        return full

    @classmethod
    def from_env(cls, env, n_zernike: int,
                 n_seg_actions: Optional[int] = None,
                 env_action_dim: Optional[int] = None,
                 dm_action_scale: float = 1.0,
                 skip_piston: bool = False,
                 normalize: str = "wavefront_rms",
                 regularize_rcond: float = 1e-3,
                 device: Optional[torch.device] = None,
                 silence: bool = True) -> "SegmentZernikePassthroughAdapter":
        """Build for ``env`` (v4 or v5). DM block sits at the tail of
        the env action vector; segment block (``n_seg_actions``)
        sits up front. Both are auto-derived from the env when not
        explicitly supplied."""
        # Reuse ZernikeToDMAdapter.from_env for the M-projection build
        # path -- it already handles both v4 and v5 envs. Then unpack
        # its M + slice and recombine into the passthrough shape.
        zd = ZernikeToDMAdapter.from_env(
            env, n_zernike=n_zernike,
            env_action_dim=env_action_dim,
            action_scale=1.0,           # scale applied here, not there
            skip_piston=skip_piston,
            normalize=normalize,
            regularize_rcond=regularize_rcond,
            device=device,
            silence=silence)
        if env_action_dim is None:
            env_action_dim = zd.env_action_dim
        if n_seg_actions is None:
            n_seg_actions = env_action_dim - zd.n_dm
        return cls(zd.M, n_seg_actions=n_seg_actions,
                   env_action_dim=env_action_dim,
                   dm_action_scale=dm_action_scale)

    def __repr__(self) -> str:
        return (f"SegmentZernikePassthroughAdapter(n_seg="
                f"{self.n_seg_actions}, n_z={self.n_zernike}, "
                f"n_dm={self.n_dm}, env_action_dim="
                f"{self.env_action_dim}, dm_scale="
                f"{self.dm_action_scale})")
