"""Zernike-coefficient policy wrapper for the v5 DM-control env.

This wrapper sits between the policy and a base v5 ``BatchedOptomechEnv``
so the agent emits low-dimensional Zernike coefficients instead of raw
per-actuator DM commands. A fixed projection matrix
``M [n_dm_acts, n_zernike]`` -- built once at construction time by
least-squares fitting each Zernike mode against the DM's influence
basis on the env's pupil -- maps the agent action to a full DM-action
vector that the env consumes natively. The env never sees a Zernike
action; it still receives ``[seg_actions..., dm_actions...]`` exactly
as before.

Motivation: plain PPO on 1225-dim absolute or incremental DM was
unable to learn even the simplest strehl-only sanity-check (runs
dm_strehl_only_full_1779084109 .. 1779089444). The action dim
dominates the policy gradient noise and the value-function bias
flows from a shared encoder into a mean-drift catastrophe within an
episode. Reducing the policy's effective dim to ~10-40 Zernike modes
matches the dimensionality where PPO is known to work and matches the
bench-deployment convention: outboard hardware will accept Zernike
coefficients and project them itself.

For bench transfer: the same ``M`` matrix (returned by
``ZernikeDMVectorEnv.projection_matrix()``) is what the operator-side
code should apply to the agent's emitted coefficients before sending
to the DM driver. The training rig and the eval rig both use this
wrapper, so the math is identical in simulation and on the bench.
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
from gymnasium import spaces, vector


# ---------------------------------------------------------------------------
# Noll-indexed Zernike polynomials
# ---------------------------------------------------------------------------

def _noll_nm_table(max_j: int) -> list[tuple[int, int]]:
    """Return ``[(n, m), ...]`` for Noll indices ``j=1..max_j``.

    Noll 1976 convention: within radial order ``n`` the modes are
    ordered as ``m=0`` (if ``n`` is even), then pairs ``±|m|`` for
    ``|m|`` increasing in steps of 2. The first sign within a pair
    depends on ``(n // 2) mod 2``: positive when even (``n in {0,1,4,5,
    8,9,...}``), negative when odd (``n in {2,3,6,7,...}``).
    """
    out: list[tuple[int, int]] = []
    j = 0
    n = 0
    while j < max_j:
        ms: list[int] = []
        if n % 2 == 0:
            ms.append(0)
        first_sign = +1 if (n // 2) % 2 == 0 else -1
        start_abs = 1 if n % 2 == 1 else 2
        for m_abs in range(start_abs, n + 1, 2):
            ms.append(first_sign * m_abs)
            ms.append(-first_sign * m_abs)
        for m in ms:
            j += 1
            out.append((n, m))
            if j >= max_j:
                break
        n += 1
    return out


def _radial_poly(n: int, m: int, r: np.ndarray) -> np.ndarray:
    """Standard Zernike radial polynomial ``R_n^{|m|}(r)``. Zero where
    ``(n - |m|)`` is odd; degree-(n) polynomial in ``r`` otherwise."""
    m = abs(m)
    if (n - m) % 2 != 0:
        return np.zeros_like(r)
    R = np.zeros_like(r)
    for k in range((n - m) // 2 + 1):
        num = (-1) ** k * math.factorial(n - k)
        den = (math.factorial(k)
               * math.factorial((n + m) // 2 - k)
               * math.factorial((n - m) // 2 - k))
        R = R + (num / den) * r ** (n - 2 * k)
    return R


def _zernike(n: int, m: int, r: np.ndarray, theta: np.ndarray,
             unit_disk_normalize: bool = True) -> np.ndarray:
    """Noll-normalized Zernike polynomial ``Z_n^m(r, theta)``.

    Unit-disk normalization: ``\\int Z^2 dA / (pi) = 1`` over the unit
    disk. With this convention an RMS of 1 on the disk equals a
    coefficient of 1.
    """
    R = _radial_poly(n, m, r)
    if m == 0:
        norm = math.sqrt(n + 1) if unit_disk_normalize else 1.0
        return norm * R
    norm = math.sqrt(2 * (n + 1)) if unit_disk_normalize else 1.0
    if m > 0:
        return norm * R * np.cos(m * theta)
    return norm * R * np.sin(abs(m) * theta)


def zernike_basis_on_grid(
        H: int, W: int, mask: np.ndarray,
        n_modes: int, skip_piston: bool = False) -> np.ndarray:
    """Evaluate the first ``n_modes`` Noll-indexed Zernikes on an HxW grid.

    Returns ``Z[H*W, n_modes]``. The aperture mask defines the unit
    disk: its bounding-box half-extent sets the unit-radius scale and
    its centroid sets the origin. Pixels outside the mask are zeroed.

    ``skip_piston=True`` drops Noll j=1 (uniform piston) from the basis;
    useful when the env already has a separate piston DOF and you don't
    want the Zernike wrapper to duplicate it.
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        raise ValueError("aperture mask is empty")
    cy = (ys.min() + ys.max()) / 2.0
    cx = (xs.min() + xs.max()) / 2.0
    R = max((ys.max() - ys.min()) / 2.0, (xs.max() - xs.min()) / 2.0)

    yy, xx = np.indices((H, W), dtype=np.float64)
    r = np.sqrt(((yy - cy) / R) ** 2 + ((xx - cx) / R) ** 2)
    theta = np.arctan2(yy - cy, xx - cx)

    table = _noll_nm_table(n_modes + (1 if skip_piston else 0))
    if skip_piston:
        table = table[1:]
    assert len(table) == n_modes

    Z = np.zeros((H * W, n_modes), dtype=np.float64)
    flat_mask = mask.reshape(-1)
    for k, (n, m) in enumerate(table):
        z = _zernike(n, m, r, theta, unit_disk_normalize=True)
        z = z * mask                                            # zero outside aperture
        Z[:, k] = z.reshape(-1)
        # Defensive: any NaN from r=0 in cos/sin branches gets zeroed.
        bad = ~np.isfinite(Z[:, k])
        if bad.any():
            Z[bad, k] = 0.0
    # Force exact zero outside mask to keep the LS solve clean.
    Z[~flat_mask, :] = 0.0
    return Z


# ---------------------------------------------------------------------------
# Standalone projection builder
# ---------------------------------------------------------------------------

def build_zernike_to_dm_projection(
        dm_influence_flat: np.ndarray,
        aperture_mask: np.ndarray,
        n_zernike: int,
        skip_piston: bool = False,
        normalize: str = "wavefront_rms",
        regularize_rcond: float = 1e-3,
        silence: bool = True) -> np.ndarray:
    """Build the Zernike-coefficient -> DM-actuator projection M.

    This is the same math the v5 ZernikeDMVectorEnv runs at training
    time, exposed as a free function so inference-time composites can
    project ZernikeDM-trained policies onto a DM action vector in a
    v4 env (or any env that exposes its DM influence basis + pupil
    mask).

    Arguments
    ---------
    dm_influence_flat : np.ndarray, shape [n_dm, H*W]
        DM influence basis flattened per actuator. Each row is one
        actuator's pupil-plane influence pattern.
    aperture_mask : np.ndarray, shape [H, W] (bool)
        Binary pupil mask. Defines the unit-disk extent used to
        evaluate the Zernike basis.
    n_zernike : int
        Number of Noll-indexed Zernike modes the policy expects to
        emit. Must match the checkpoint's policy_head shape.
    skip_piston, normalize, regularize_rcond
        Match the defaults used by ZernikeDMVectorEnv. ``normalize``
        is one of ``inf``, ``rms``, ``wavefront_rms``, ``none``.

    Returns
    -------
    M : np.ndarray, shape [n_dm, n_zernike], float32
        Action-space projection so that ``a_dm = clamp(M @ a_z,
        -1, 1)`` reproduces the same DM commands the policy
        emitted at training time.
    """
    H, W = aperture_mask.shape
    Z = zernike_basis_on_grid(H, W, aperture_mask, n_zernike,
                              skip_piston=skip_piston)        # [H*W, n_z]
    flat_mask = aperture_mask.reshape(-1)
    Phi_in = dm_influence_flat[:, flat_mask]                  # [n_dm, n_pix]
    Z_in = Z[flat_mask, :]                                    # [n_pix, n_z]
    n_pupil = int(flat_mask.sum())

    M, _resid, rank, svals = np.linalg.lstsq(
        Phi_in.T, Z_in, rcond=regularize_rcond)
    M = M.astype(np.float64)

    if not silence:
        print(f"[zernike-dm projection] LS rank={rank}/{Phi_in.shape[0]} "
              f"(rcond={regularize_rcond:g}, "
              f"sigma_max={svals[0]:.3g}, "
              f"sigma_min_kept={svals[min(rank, len(svals)) - 1]:.3g})")

    if normalize == "inf":
        norms = np.max(np.abs(M), axis=0)
    elif normalize == "rms":
        norms = np.sqrt((M * M).mean(axis=0))
    elif normalize == "wavefront_rms":
        wf = Phi_in.T @ M                                     # [n_pix, n_z]
        norms = np.sqrt((wf * wf).sum(axis=0) / max(n_pupil, 1))
    else:
        norms = np.ones(n_zernike, dtype=np.float64)
    norms = np.where(norms > 0, norms, 1.0)
    M = M / norms[None, :]

    # Cap so a unit Zernike action never drives any actuator beyond
    # ±1 (matches ZernikeDMVectorEnv's clamp-safety guarantee).
    max_per_col = np.max(np.abs(M), axis=0)
    cap = np.where(max_per_col > 1.0, max_per_col, 1.0)
    M = M / cap[None, :]

    return M.astype(np.float32)


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------

class ZernikeDMVectorEnv(vector.VectorEnv):
    """Vectorized wrapper that exposes Zernike coefficients to the policy.

    The wrapped env must be a v5 ``BatchedOptomechEnv`` with
    ``command_dm=True``. The wrapper:

    * Builds a projection ``M [n_dm_acts, n_zernike]`` once at init.
    * Overrides ``single_action_space`` to ``[-1, 1]^n_zernike``.
    * On ``step``, computes ``dm_action = clamp(a @ M.T, -1, 1)`` and
      passes a full env action vector (with zero seg slots when
      ``command_secondaries=False`` or with explicit zero seg slots
      when ``freeze_segments=True``) to the underlying env.

    Action-bound semantics: ``M`` is column-normalized so that
    ``a_zernike = 1`` produces a DM-action whose largest entry has
    magnitude exactly 1. Combined with the policy's
    ``scale_and_clamp_action``, this guarantees that any
    ``a_zernike in [-1, 1]^n_zernike`` produces a DM action in
    ``[-1, 1]^n_dm`` without saturation (modulo numerical roundoff).
    """

    _SUPPORTED_NORMS = ("inf", "rms", "wavefront_rms", "none")

    def __init__(self, env, n_zernike: int = 36,
                 skip_piston: bool = False,
                 freeze_segments: bool = True,
                 normalize: Literal[
                     "inf", "rms", "wavefront_rms", "none"] = "wavefront_rms",
                 regularize_rcond: float = 1e-3):
        if not hasattr(env, "_dm_basis_t_flat") or env._dm_basis_t_flat is None:
            raise ValueError(
                "ZernikeDMVectorEnv requires a v5 env with command_dm=True; "
                "the wrapped env has no DM influence basis.")
        if normalize not in self._SUPPORTED_NORMS:
            raise ValueError(
                f"normalize must be one of {self._SUPPORTED_NORMS}, "
                f"got {normalize!r}")

        self._env = env
        self._dev = env.dev
        self._N = env._num_envs
        self._n_dm = int(env._n_dm_acts)
        self._n_seg = int(env._num_apertures)
        self._n_dof_per_seg = int(env._n_dof_per_seg)
        self._n_seg_actions = int(env._n_seg_actions)
        self._n_zernike = int(n_zernike)
        self._skip_piston = bool(skip_piston)
        self._freeze_segments = bool(freeze_segments)
        self._normalize = normalize
        self._regularize_rcond = float(regularize_rcond)

        # Build the projection matrix (numpy first, then to GPU tensor).
        M_np = self._build_projection()
        self._M_t = torch.tensor(
            M_np, dtype=torch.float32, device=self._dev)         # [n_dm, n_z]

        # Action space exposed to the policy.
        single_action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self._n_zernike,),
            dtype=np.float32)
        super().__init__(
            num_envs=self._N,
            observation_space=env.single_observation_space,
            action_space=single_action_space)

        # Pre-allocate the zero seg-action block, used when the env
        # still expects seg actions (command_secondaries=True). With
        # command_secondaries=False the env's action layout is
        # DM-only and this block has zero size.
        self._zero_seg = torch.zeros(
            self._N, self._n_seg_actions,
            dtype=torch.float32, device=self._dev)

    # ------------------------------------------------------------------
    # Projection construction
    # ------------------------------------------------------------------

    def _build_projection(self) -> np.ndarray:
        """Solve ``Phi.T @ M = Z`` in *regularized* least squares and
        normalize the result so unit Zernike action produces a sensible
        wavefront amplitude.

        ``Phi``  : [n_dm, n_pix_in_pupil]   DM influence functions
        ``Z``    : [n_pix_in_pupil, n_z]    Zernike basis on the pupil
        ``M``    : [n_dm, n_z]              actuator coefficients per mode

        Regularization (``regularize_rcond``, default 1e-3) is critical:
        the gaussian-influence DM has heavily overlapping per-actuator
        bumps, so ``Phi_in.T`` is highly ill-conditioned. With the
        numpy default ``rcond`` (machine epsilon) the LS solution finds
        per-actuator amplitudes of order 1e6-1e7 that almost cancel
        through Phi, leaving the desired Zernike pattern as a tiny
        residual. After ``max|M|=1`` normalization the cancellation
        pattern is preserved but the wavefront amplitude collapses by
        the same 1e6 factor -- so the policy commanding large Zernike
        actions sees essentially no wavefront response. Truncating at
        ``rcond=1e-3`` drops the offending small singular values; the
        resulting M is smooth, distributed across actuators, and
        produces actual wavefronts.

        Normalization options:
          - ``"wavefront_rms"`` (default): scale each column so that
            unit Zernike action produces a DM-action vector whose
            wavefront has unit RMS on the pupil (in DM-action units;
            the env's stroke and action_scale convert this to a
            physical OPD). Caps any column that would otherwise drive
            an actuator beyond +/- 1 so the wrapper's downstream clamp
            never distorts the projection. This is the physically
            meaningful normalization for an RL policy.
          - ``"inf"``: legacy, max|M|=1 per column. Preserves pattern
            but couples wavefront amplitude to the LS regularization.
          - ``"rms"``: max RMS|M|=1 per column.
          - ``"none"``: no scaling.
        """
        env = self._env
        H, W = env._H, env._W

        # Binary pupil mask from the env's complex aperture tensor.
        aperture = np.asarray(env._aperture_t.cpu().numpy())
        mask = (np.abs(aperture) > 0).reshape(H, W)
        n_pupil = int(mask.sum())

        Z = zernike_basis_on_grid(H, W, mask, self._n_zernike,
                                  skip_piston=self._skip_piston)  # [H*W, n_z]

        flat_mask = mask.reshape(-1)
        Phi_full = np.asarray(env._dm_basis_t_flat.cpu().numpy())   # [n_dm, H*W]
        Phi_in = Phi_full[:, flat_mask]                             # [n_dm, n_pix]
        Z_in = Z[flat_mask, :]                                      # [n_pix, n_z]

        # Regularized LS. rcond truncates singular values smaller than
        # rcond * sigma_max so the gaussian-DM null space doesn't blow up.
        M, _resid, rank, svals = np.linalg.lstsq(
            Phi_in.T, Z_in, rcond=self._regularize_rcond)
        M = M.astype(np.float64)                                    # [n_dm, n_z]

        # Diagnostic so we can tell at-a-glance whether the regularizer
        # ate too many modes; printed only when we're inside a non-
        # silent env construction.
        if not getattr(env, "_silence", False):
            print(f"[ZernikeDMVectorEnv] LS rank={rank}/{Phi_in.shape[0]} "
                  f"(rcond={self._regularize_rcond:g}, "
                  f"sigma_max={svals[0]:.3g}, "
                  f"sigma_min_kept={svals[min(rank, len(svals)) - 1]:.3g})")

        # Apply column normalization. For "wavefront_rms" we measure
        # the actual wavefront RMS from the LS solution on the pupil
        # and rescale so it equals 1 in DM-action units.
        if self._normalize == "inf":
            norms = np.max(np.abs(M), axis=0)
        elif self._normalize == "rms":
            norms = np.sqrt((M * M).mean(axis=0))
        elif self._normalize == "wavefront_rms":
            # wavefront_k = Phi_in.T @ M[:, k] on the pupil [n_pupil].
            wf = Phi_in.T @ M                                       # [n_pix, n_z]
            norms = np.sqrt((wf * wf).sum(axis=0) / max(n_pupil, 1))
        else:
            norms = np.ones(self._n_zernike, dtype=np.float64)

        # Guard against degenerate modes (e.g., m != 0 evaluated on a
        # symmetric pupil that happens to project to zero on every
        # actuator, or modes outside Phi's representational reach) --
        # leave their column at zero rather than blow up.
        good = norms > 1e-12
        out = np.zeros_like(M)
        out[:, good] = M[:, good] / norms[good][None, :]

        # Belt-and-suspender clamp: ensure no actuator command exceeds
        # +/- 1 at unit Zernike action, regardless of normalization
        # choice. If any column does, rescale that whole column down so
        # the wrapper's downstream clamp() never silently distorts the
        # projected pattern (which would manifest as off-Zernike
        # speckle in the rendered DM OPD).
        col_peak = np.max(np.abs(out), axis=0)
        oversize = col_peak > 1.0
        if oversize.any():
            out[:, oversize] = out[:, oversize] / col_peak[oversize][None, :]

        return out.astype(np.float32)

    # ------------------------------------------------------------------
    # Public hooks for bench export / debugging
    # ------------------------------------------------------------------

    def projection_matrix(self) -> np.ndarray:
        """Return the [n_dm, n_zernike] projection as a CPU numpy array.

        Use this to export the same matrix for the operator-side bench
        code: ``dm_action_bench = M @ zernike_coeffs_from_agent``."""
        return self._M_t.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Action expansion
    # ------------------------------------------------------------------

    def expand_action(self, agent_action: np.ndarray) -> np.ndarray:
        """Map ``[N, n_zernike]`` -> ``[N, n_seg_actions + n_dm]``.

        Clamps the projected DM action to [-1, 1]; with inf-normalised
        ``M`` the clamp is essentially a no-op for ``|a_zernike| <= 1``
        but the safety belt covers numerical drift and stochastic
        samples that briefly leave the box.
        """
        a = torch.as_tensor(agent_action, dtype=torch.float32, device=self._dev)
        dm_action = (a @ self._M_t.T).clamp(-1.0, 1.0)            # [N, n_dm]
        if self._n_seg_actions > 0:
            full = torch.cat([self._zero_seg, dm_action], dim=1)
        else:
            full = dm_action
        return full.cpu().numpy()

    # ------------------------------------------------------------------
    # VectorEnv interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, agent_action):
        return self._env.step(self.expand_action(agent_action))

    def close(self):
        return self._env.close()

    # Delegate attribute lookups (e.g. _reference_fpi_max,
    # _hole_mask_t, _last_strehl) to the wrapped env so callers see a
    # transparent view of the underlying v5 state.
    def __getattr__(self, item):
        # __getattr__ is only invoked for missing attrs, so this won't
        # recurse on _env / wrapper internals.
        return getattr(self._env, item)


# ---------------------------------------------------------------------------
# Segment-passthrough + Zernike-DM wrapper
# ---------------------------------------------------------------------------

class SegmentZernikeDMVectorEnv(vector.VectorEnv):
    """Segment-passthrough variant of ZernikeDMVectorEnv.

    Exposes ``[n_seg_actions + n_zernike]`` to the policy. The first
    ``n_seg_actions`` channels pass through unchanged to the env's
    segment slots; the remaining ``n_zernike`` channels are projected
    via the same ``M [n_dm, n_zernike]`` matrix as
    ``ZernikeDMVectorEnv``, scaled by ``dm_action_scale``, and clamped
    to ``[-1, 1]`` before being passed to the env's DM slots.

    Why a separate class? ZernikeDMVectorEnv is DM-only (zeros the
    segment slots regardless of ``freeze_segments``). We need an
    env that lets a *single* policy emit segment PTT corrections
    AND DM corrections in the same step -- the segment_dm_agent
    transfer-learning setup (15 bootstrap phases, each fine-tuned
    to also drive 32 Zernike DM modes under a Kolmogorov atmosphere).

    ``dm_action_scale`` is applied to the DM block only; the env's
    own ``env_action_scale`` still applies uniformly to whatever this
    wrapper hands over (so segment magnitudes can be left at 1.0 while
    the DM block gets the per-step delta cap, e.g. 0.01, that the
    dm_atmos training pipeline used).
    """

    def __init__(self, env, n_zernike: int = 32,
                 dm_action_scale: float = 1.0,
                 skip_piston: bool = False,
                 normalize: Literal[
                     "inf", "rms", "wavefront_rms", "none"] = "wavefront_rms",
                 regularize_rcond: float = 1e-3):
        if not hasattr(env, "_dm_basis_t_flat") or env._dm_basis_t_flat is None:
            raise ValueError(
                "SegmentZernikeDMVectorEnv requires a v5 env with "
                "command_dm=True; the wrapped env has no DM influence "
                "basis.")
        if not getattr(env, "_n_seg_actions", 0):
            raise ValueError(
                "SegmentZernikeDMVectorEnv requires a v5 env with "
                "command_secondaries=True (n_seg_actions > 0); the "
                "wrapped env has no segment action slots.")

        self._env = env
        self._dev = env.dev
        self._N = env._num_envs
        self._n_dm = int(env._n_dm_acts)
        self._n_seg_actions = int(env._n_seg_actions)
        self._n_zernike = int(n_zernike)
        self._skip_piston = bool(skip_piston)
        self._normalize = normalize
        self._regularize_rcond = float(regularize_rcond)
        self._dm_action_scale = float(dm_action_scale)

        # Build the projection matrix using the standalone helper
        # (same math as ZernikeDMVectorEnv._build_projection).
        H, W = env._H, env._W
        aperture = np.asarray(env._aperture_t.cpu().numpy())
        mask = (np.abs(aperture) > 0).reshape(H, W)
        dm_flat = np.asarray(env._dm_basis_t_flat.cpu().numpy())
        M_np = build_zernike_to_dm_projection(
            dm_flat, mask, n_zernike,
            skip_piston=skip_piston,
            normalize=normalize,
            regularize_rcond=regularize_rcond,
            silence=getattr(env, "_silence", False))
        self._M_t = torch.tensor(
            M_np, dtype=torch.float32, device=self._dev)        # [n_dm, n_z]

        # Action space exposed to the policy: [segments | zernikes].
        single_action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self._n_seg_actions + self._n_zernike,),
            dtype=np.float32)
        super().__init__(
            num_envs=self._N,
            observation_space=env.single_observation_space,
            action_space=single_action_space)

    def projection_matrix(self) -> np.ndarray:
        """[n_dm, n_zernike] projection (CPU numpy). For bench export."""
        return self._M_t.detach().cpu().numpy()

    def expand_action(self, agent_action: np.ndarray) -> np.ndarray:
        """Map ``[N, n_seg + n_z]`` -> ``[N, n_seg + n_dm]`` env action."""
        a = torch.as_tensor(agent_action, dtype=torch.float32, device=self._dev)
        seg = a[:, : self._n_seg_actions]
        zer = a[:, self._n_seg_actions:]
        dm = zer @ self._M_t.T                                  # [N, n_dm]
        if self._dm_action_scale != 1.0:
            dm = dm * self._dm_action_scale
        dm = dm.clamp(-1.0, 1.0)
        full = torch.cat([seg, dm], dim=1)
        return full.cpu().numpy()

    def reset(self, *, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, agent_action):
        return self._env.step(self.expand_action(agent_action))

    def close(self):
        return self._env.close()

    def __getattr__(self, item):
        return getattr(self._env, item)
