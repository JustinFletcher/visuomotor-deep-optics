"""Agent-side bilateral-symmetry wrapper for the DM dark-hole task.

Two transformations live here, both applied at the agent level so the
underlying ``BatchedOptomechEnv`` requires no special treatment:

1. **Action symmetrization.** The policy outputs commands for only one
   half of the DM actuators -- specifically, the half on the same side
   as the target dark hole, partitioned by the line through the origin
   perpendicular to the target's radial direction. The mirror half is
   filled in by reflecting controlled commands across that line; on-axis
   actuators are pinned to zero. The resulting full DM command is, by
   construction, bilaterally symmetric about the same axis.

2. **Observation blinding.** The bilateral mirror of the target dark
   hole (a circular region at angle theta+180 degrees, same radius and
   size) is masked out of the focal-plane observation. The policy never
   sees this region. At test time, the unmasked blind region is the
   only honest signal: any light that appears there reveals whether the
   policy genuinely shaped a dark hole rather than gaming the visible
   reward by pushing flux into a place it could not see.

The wrapper presents itself as a ``gymnasium.vector.VectorEnv`` so it
can drop into any training script that accepts a v5 batched env. The
action space dimension is reduced by ``n_dm_acts - n_half`` (where
``n_half = n_dm_acts // 2``), the rest determined by the symmetry.

The per-episode partition (which actuators are controlled, which are
mirrored, where the blind mask sits) is rebuilt on every ``reset()``
from the underlying env's per-env ``target_vec``, so dynamic-target
configurations work transparently. For static-target runs the partition
just doesn't change.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
from gymnasium import spaces
from gymnasium import vector


class BilateralDMVectorEnv(vector.VectorEnv):
    """VectorEnv wrapper that enforces bilateral DM symmetry and blinds
    the bilaterally-mirrored focal-plane region.

    Constructor arguments:

        env: a ``BatchedOptomechEnv`` built with ``command_dm=True``.
            Its action space must be ``[seg_actions ..., dm_actions ...]``
            (the v5 layout with DM appended). The wrapper is agnostic
            to whether the underlying env runs the DM in absolute or
            incremental control mode -- the symmetrization is purely
            on the action vector itself, not on the resulting DM state.

    Exposed action layout: ``[seg_actions ..., dm_half_actions ...]`` of
    dim ``n_seg_actions + n_half`` where ``n_half = n_dm_acts // 2``.
    """

    # Symmetry-axis modes:
    #   "per_target_radial" (default, original): per episode, axis is
    #       perpendicular to the target's radial direction. Controlled
    #       side = target side. Blind region is the diametric opposite
    #       of the target. Strict bilateral symmetry only for cardinal
    #       angles; off-cardinal targets fall back to nearest-neighbor
    #       mirror partners on the Cartesian actuator grid, so the
    #       symmetry is approximate.
    #   "fixed_vertical": axis is the y-axis (x=0) for every episode and
    #       every target. The 35x35 actuator grid is exactly bijective
    #       under (x, y) -> (-x, y), so symmetry is strict per-actuator.
    #       Blind region is the left/right mirror of the target across
    #       the vertical focal-plane axis.
    _SUPPORTED_MODES = ("per_target_radial", "fixed_vertical")

    def __init__(self, env, freeze_segments: bool = True,
                 mode: str = "per_target_radial"):
        if env._n_dm_acts == 0:
            raise ValueError(
                "BilateralDMVectorEnv requires the underlying env to be "
                "built with command_dm=True (n_dm_acts > 0)")
        if mode not in self._SUPPORTED_MODES:
            raise ValueError(
                f"BilateralDMVectorEnv mode {mode!r} not in "
                f"{self._SUPPORTED_MODES}")

        self._env = env
        self._N = env.num_envs
        self._dev = env.dev
        self._H = env._H
        self._W = env._W
        self._n_seg = env._n_seg_actions
        self._n_dm = env._n_dm_acts
        self._mode = mode
        # When freeze_segments is True the policy outputs only the DM
        # half-slice; the seg slice in the expanded action is zeroed.
        # This is the natural mode for the DM dark-hole task where
        # segments are held at zero relative motion as a simplification
        # (the segment-piston handoff is treated as a separate stage).
        self._freeze_segments = bool(freeze_segments)
        # Half the actuators are policy-controlled; the other half is
        # determined by mirror symmetry. With an odd actuator count
        # (e.g. 35x35 = 1225) the leftover one is on-axis and pinned.
        self._n_half = self._n_dm // 2

        # Actuator positions in pupil meters, shape [A, 2].
        # Row order matches _dm_basis_t.
        self._dm_xy = env._dm_actuator_xy_t.clone()        # [A, 2]
        # Pre-compute pairwise mirror lookups uses these.

        # --- Per-env partition state (populated by _rebuild_partition) ---
        # controlled_idx[n, k] = global DM actuator index of the k-th
        #     controlled actuator for env n.
        # mirror_partner_idx[n, k] = global DM actuator index of the
        #     actuator on the opposite side that mirrors controlled k.
        # mirror_partner_idx may have duplicates when the actuator grid
        # has no exact bijective partner under the chosen axis (off
        # cardinal angles); in that case multiple mirror slots receive
        # the same controlled command, which is the closest discrete
        # approximation of true bilateral symmetry on a Cartesian grid.
        self._controlled_idx = torch.zeros(
            self._N, self._n_half, dtype=torch.long, device=self._dev)
        self._mirror_partner_idx = torch.zeros(
            self._N, self._n_half, dtype=torch.long, device=self._dev)
        # blind_mask[n, h, w]: True where pixels are blinded from obs.
        self._blind_mask = torch.zeros(
            self._N, self._H, self._W, dtype=torch.bool, device=self._dev)

        # Cache the most recent target_vec we partitioned for so we can
        # detect when nothing has changed and skip recompute.
        self._last_target_vec = torch.full(
            (self._N, 4), float("nan"), dtype=torch.float32, device=self._dev)

        # Wrapped action space: drop n_dm - n_half DM actions, and
        # optionally drop the seg slice entirely when segments are
        # frozen at zero.
        action_dim = self._n_half + (0 if self._freeze_segments else self._n_seg)
        single_action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(action_dim,),
            dtype=np.float32)
        # Observation space inherits underlying env's shape and dtype.
        super().__init__(
            num_envs=self._N,
            observation_space=env.single_observation_space,
            action_space=single_action_space)

        # Default partition: build something sane for an arbitrary
        # zero target_vec so step/reset never see uninitialised state.
        # The first reset() will overwrite this with real per-env values.
        self._rebuild_partition(torch.zeros(self._N, 4, device=self._dev))

    # ------------------------------------------------------------------
    # Geometry: per-env partition + blind mask
    # ------------------------------------------------------------------

    def _rebuild_partition(self, target_vec: torch.Tensor) -> None:
        """Recompute controlled/mirror indices and the blind mask for
        every env whose target_vec differs from the cached one.

        target_vec: [N, 4] tensor on self._dev with columns
            [sin(theta), cos(theta), radius_frac, size_frac].
        """
        # Per-env mask of which envs need to be rebuilt.
        diff = (target_vec - self._last_target_vec).abs().sum(dim=1)
        needs = (diff > 1e-6) | torch.isnan(self._last_target_vec).any(dim=1)
        if not bool(needs.any().item()):
            return

        for n in torch.where(needs)[0].cpu().tolist():
            sin_t = float(target_vec[n, 0].item())
            cos_t = float(target_vec[n, 1].item())
            r_frac = float(target_vec[n, 2].item())
            s_frac = float(target_vec[n, 3].item())
            if self._mode == "fixed_vertical":
                # Symmetry axis is x=0 for every env. The partition
                # uses the same code path as per_target_radial but
                # forces the radial direction to (cos_t=1, sin_t=0),
                # which makes d = x and the topk pick the right (x>0)
                # half. Mirror partners are exact under (x,y)->(-x,y)
                # because the actuator grid is symmetric in x.
                self._build_partition_for_env(n, sin_t=0.0, cos_t=1.0)
                self._build_blind_mask_fixed_vertical(
                    n, sin_t, cos_t, r_frac, s_frac)
            else:
                self._build_partition_for_env(n, sin_t, cos_t)
                self._build_blind_mask_for_env(
                    n, sin_t, cos_t, r_frac, s_frac)

        self._last_target_vec = target_vec.detach().clone()

    def _build_partition_for_env(self, n: int, sin_t: float, cos_t: float):
        """Partition DM actuators for env n into (controlled, mirror).

        The symmetry axis is the line through the origin perpendicular
        to the target's radial direction (cos theta, sin theta). An
        actuator's signed distance along the radial direction,
        d = x*cos_t + y*sin_t, is positive on the target side and
        negative on the blind side.

        We pick the n_half actuators with the largest d as controlled.
        For each controlled actuator, the mirror partner is the actuator
        nearest its reflected position p' = p - 2 d (cos_t, sin_t) on
        the opposite side. Ties (or non-bijective mappings on off-axis
        rotations) are tolerated -- multiple mirror slots may copy the
        same controlled command. On-axis actuators (those that are not
        anyone's mirror partner and are not in controlled) are left at
        zero in expand_action.
        """
        xy = self._dm_xy                              # [A, 2]
        d = xy[:, 0] * cos_t + xy[:, 1] * sin_t       # [A] signed distance

        # Top n_half by signed distance -> controlled side.
        _, top_idx = torch.topk(d, k=self._n_half, largest=True, sorted=False)
        self._controlled_idx[n] = top_idx                              # [n_half]

        # For each controlled actuator k, compute its reflected position
        # p_k - 2 d_k (cos_t, sin_t), then find the nearest actuator on
        # the opposite side.
        p_ctrl = xy[top_idx]                          # [n_half, 2]
        d_ctrl = d[top_idx]                           # [n_half]
        refl = p_ctrl - 2.0 * d_ctrl.unsqueeze(1) * torch.tensor(
            [cos_t, sin_t], dtype=xy.dtype, device=xy.device)          # [n_half, 2]

        # Restrict candidate mirror set to actuators with d <= 0 (the
        # opposite side, including on-axis). With odd A there's exactly
        # one on-axis actuator and n_half on each side after the topk.
        opp_mask = (d <= 0.0)
        opp_idx = torch.where(opp_mask)[0]            # [n_opp]
        opp_xy = xy[opp_idx]                          # [n_opp, 2]

        # Pairwise distances [n_half, n_opp].
        dist = torch.cdist(refl.unsqueeze(0), opp_xy.unsqueeze(0))[0]
        nearest = torch.argmin(dist, dim=1)           # [n_half]
        self._mirror_partner_idx[n] = opp_idx[nearest]

    def _build_blind_mask_for_env(
            self, n: int, sin_t: float, cos_t: float,
            r_frac: float, s_frac: float):
        """Build the blind-region focal-plane mask for env n.

        The blind region is the bilateral mirror of the target dark hole
        across the symmetry axis. Since the dark hole sits on the radial
        line at angle theta and the axis is perpendicular to that line,
        reflecting the dark hole across the axis sends the centre to
        angle theta + 180 deg at the same radius. The size matches the
        target hole.

        Mirrors the construction in BatchedOptomechEnv._build_hole_mask
        so the mask uses identical pixel arithmetic.
        """
        H = self._H
        # Angle of the blind region centre, in radians.
        # If target = (r cos_t, r sin_t), blind centre = (-r cos_t, -r sin_t).
        # In pixel coords: cy_b = H/2 - r_px * sin_t; cx_b = H/2 - r_px * cos_t.
        # (The env uses cy = H/2 + sin*r_px, so flipping the sign
        # produces the diametrically opposite point.)
        loc_px = int(r_frac * H / 2)
        size_px = int(s_frac * H / 2)
        cx = int(H / 2 - loc_px * cos_t)
        cy = int(H / 2 - loc_px * sin_t)
        yy, xx = torch.meshgrid(
            torch.arange(H, device=self._dev),
            torch.arange(self._W, device=self._dev),
            indexing="ij")
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= size_px ** 2
        self._blind_mask[n] = mask

    def _build_blind_mask_fixed_vertical(
            self, n: int, sin_t: float, cos_t: float,
            r_frac: float, s_frac: float):
        """Blind region for fixed_vertical mode.

        The symmetry axis is x = 0 (vertical line through the focal-
        plane centre). The bilateral mirror of the target sends
        (x, y) -> (-x, y), so a target at angle theta maps to angle
        pi - theta at the same radius. Pixel arithmetic mirrors the
        cx component but preserves cy; same circle radius as the
        target hole.
        """
        H = self._H
        loc_px = int(r_frac * H / 2)
        size_px = int(s_frac * H / 2)
        cx = int(H / 2 - loc_px * cos_t)        # mirrored x: 2*ctr - x_target
        cy = int(H / 2 + loc_px * sin_t)        # y unchanged
        yy, xx = torch.meshgrid(
            torch.arange(H, device=self._dev),
            torch.arange(self._W, device=self._dev),
            indexing="ij")
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= size_px ** 2
        self._blind_mask[n] = mask

    # ------------------------------------------------------------------
    # Action expansion + obs masking
    # ------------------------------------------------------------------

    def expand_action(self, agent_action: np.ndarray) -> np.ndarray:
        """Map [N, n_seg + n_half] -> [N, n_seg + n_dm].

        Segment slice passes through unchanged. The DM slice places the
        agent's n_half values at the controlled actuator indices and
        copies them to the corresponding mirror partner indices. Slots
        that are neither controlled nor anyone's mirror partner stay at
        zero (this is the on-axis sliver).
        """
        a = torch.as_tensor(agent_action, dtype=torch.float32, device=self._dev)
        N = a.shape[0]
        if self._freeze_segments:
            seg = torch.zeros(
                N, self._n_seg, dtype=torch.float32, device=self._dev)
            half = a                                  # [N, n_half]
        else:
            seg = a[:, :self._n_seg]
            half = a[:, self._n_seg:]                 # [N, n_half]

        full_dm = torch.zeros(N, self._n_dm, dtype=torch.float32, device=self._dev)
        # Place controlled actions.
        full_dm.scatter_(1, self._controlled_idx, half)
        # Place mirror copies. scatter_ overwrites; this is fine because
        # _controlled_idx and _mirror_partner_idx are disjoint by
        # construction (mirror partners come from the d<=0 opposite
        # side, controlled comes from the largest-d top half).
        full_dm.scatter_(1, self._mirror_partner_idx, half)

        out = torch.cat([seg, full_dm], dim=1)
        return out.cpu().numpy()

    def mask_obs(self, obs: np.ndarray) -> np.ndarray:
        """Zero the blind-region pixels in every channel of the obs."""
        obs_t = torch.as_tensor(obs, device=self._dev)
        # obs shape: [N, C, H, W]; blind_mask shape: [N, H, W].
        m = self._blind_mask.unsqueeze(1)            # [N, 1, H, W]
        obs_t = torch.where(m, torch.zeros_like(obs_t), obs_t)
        return obs_t.cpu().numpy()

    # ------------------------------------------------------------------
    # VectorEnv interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        # Pull target_vec from infos (always present in v5 with dark_hole).
        tv_np = info.get("target_vec")
        if tv_np is None:
            tv_t = torch.zeros(self._N, 4, device=self._dev)
        else:
            tv_t = torch.as_tensor(
                tv_np, dtype=torch.float32, device=self._dev)
        self._rebuild_partition(tv_t)
        # The env emits the unmasked observation. Callers that feed the
        # observation to a policy should call mask_obs() on it first;
        # callers that only log/visualise should use it as-is so the
        # blind region is visible (it is the only honest test signal).
        return obs, info

    def step(self, agent_action):
        full_action = self.expand_action(agent_action)
        obs, rew, term, trunc, info = self._env.step(full_action)
        # Per-env target_vec may have changed for any env that auto-reset
        # (dynamic-target setups). Refresh partitions for those.
        tv_np = info.get("target_vec")
        if tv_np is not None:
            tv_t = torch.as_tensor(
                tv_np, dtype=torch.float32, device=self._dev)
            self._rebuild_partition(tv_t)
        # Same contract as reset(): unmasked obs goes to the caller;
        # mask_obs() is a separate explicit step right before the
        # policy forward.
        return obs, rew, term, trunc, info

    def close(self):
        return self._env.close()

    # Delegate attribute lookups (e.g. _target_vec_t, _hole_mask_t,
    # _last_raw_psf_t) to the wrapped env so callers can introspect
    # underlying physics tensors without per-field plumbing here.
    def __getattr__(self, item):
        # __getattr__ runs only when normal lookup fails. Guard against
        # infinite recursion by routing through __dict__ directly.
        env = self.__dict__.get("_env")
        if env is None:
            raise AttributeError(item)
        return getattr(env, item)


def n_half_from_n_dm(n_dm: int) -> int:
    """Convenience: action-dim for the DM half slice."""
    return n_dm // 2
