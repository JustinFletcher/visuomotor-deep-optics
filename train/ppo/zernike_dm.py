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

    _SUPPORTED_NORMS = ("inf", "rms", "none")

    def __init__(self, env, n_zernike: int = 36,
                 skip_piston: bool = False,
                 freeze_segments: bool = True,
                 normalize: Literal["inf", "rms", "none"] = "inf"):
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
        """Solve ``Phi.T @ M = Z`` in least squares and column-normalize.

        ``Phi``  : [n_dm, n_pix_in_pupil]   DM influence functions
        ``Z``    : [n_pix_in_pupil, n_z]    Zernike basis on the pupil
        ``M``    : [n_dm, n_z]              actuator coefficients per mode
        """
        env = self._env
        H, W = env._H, env._W

        # Binary pupil mask from the env's complex aperture tensor.
        aperture = np.asarray(env._aperture_t.cpu().numpy())
        mask = (np.abs(aperture) > 0).reshape(H, W)

        Z = zernike_basis_on_grid(H, W, mask, self._n_zernike,
                                  skip_piston=self._skip_piston)  # [H*W, n_z]

        flat_mask = mask.reshape(-1)
        Phi_full = np.asarray(env._dm_basis_t_flat.cpu().numpy())   # [n_dm, H*W]
        Phi_in = Phi_full[:, flat_mask]                             # [n_dm, n_pix]
        Z_in = Z[flat_mask, :]                                      # [n_pix, n_z]

        # Solve Phi_in.T @ M = Z_in for M.
        # lstsq returns (solution, residuals, rank, sv).
        M, *_ = np.linalg.lstsq(Phi_in.T, Z_in, rcond=None)
        M = M.astype(np.float64)                                    # [n_dm, n_z]

        if self._normalize == "inf":
            norms = np.max(np.abs(M), axis=0)
        elif self._normalize == "rms":
            norms = np.sqrt((M * M).mean(axis=0))
        else:
            norms = np.ones(self._n_zernike, dtype=np.float64)
        # Guard against degenerate modes (e.g., m != 0 evaluated on a
        # symmetric pupil that happens to project to zero on every
        # actuator) -- leave their column at zero rather than blow up.
        good = norms > 1e-12
        out = np.zeros_like(M)
        out[:, good] = M[:, good] / norms[good][None, :]
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
