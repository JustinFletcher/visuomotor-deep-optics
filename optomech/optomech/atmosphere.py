"""GPU-native multi-layer atmospheric turbulence for v5.

A separate state element from the DM / segments: ``self._atm`` lives on
``BatchedOptomechEnv`` and contributes an additive OPD to the
wavefront *before* MFT propagation. It is **not** folded into the DM
or segment baselines because the agent's job is to *correct* it, not
to find a fixed-point offset.

Model
-----
For each layer ``l = 0..L-1`` and each env ``n = 0..N-1`` we sample a
real-valued phase screen with von Karman (= Kolmogorov + finite outer
scale) power spectral density:

    Phi_phi(f) = 0.023 * r0[l]**(-5/3) * (f**2 + 1/L0[l]**2)**(-11/6)

evaluated on the env's pupil grid. The total atmospheric OPD on the
pupil is the sum of all layers, converted from radians (at the
reference wavelength) to meters:

    OPD_atm(x) = (lambda_ref / 2*pi) * sum_l phase_l(x)

Per-layer ``r0`` is derived from a total ``r0_total`` and a turbulence
weighting (Cn^2 profile) via the standard ``r0_l = r0_total * w_l**(-3/5)``
identity. Defaults are tuned to Mt. Teide-typical conditions (median
r0_500nm ~ 12 cm, L0 ~ 25 m, 3 layers with ground-heavy weighting).

Runtime
-------
Static mode (default): screen sampled once per episode in
``reset(env_mask=...)``. Per-step cost is then **one tensor read**
of the cached screen — no FFTs, no allocations. Static is the only
mode plumbed today; ``step(dt)`` is a no-op.

Future evolving mode: each layer will be pre-sampled at a larger
spatial extent at construction time. ``step(dt)`` will advance a per-
layer offset (vx*dt, vy*dt) and return a view into the pre-sampled
slab via torch.roll or affine grid_sample. The current static API was
shaped to drop into that loop without refactoring callers.

GPU
---
Everything is a torch tensor on the env's device. The FFT-based screen
generator uses ``torch.fft.ifft2``. With N=64 envs and L=3 layers at
256x256 this is a few hundred microseconds per reset on H100 -- well
below per-step optical-pipeline cost.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Teide-typical preset (Mt. Teide observatory, Tenerife)
# ---------------------------------------------------------------------------
#
# Three-layer simplification of the canonical Mt. Teide Cn^2 profile.
# Ground-layer-heavy weighting (~50%) with one mid-altitude and one
# high-altitude layer. Wind speeds are placeholders for the future
# time-evolving mode -- ignored when static=True.
#
TEIDE_DEFAULT = dict(
    r0_total_m=0.12,              # 12 cm at 500 nm: typical median seeing
    L0_m=25.0,                    # 25 m outer scale (single value applied
                                  # to all layers; can be overridden)
    layer_heights_m=(0.0, 4000.0, 10000.0),
    layer_weights=(0.50, 0.30, 0.20),  # Cn^2 fraction by layer
    layer_wind_speeds_mps=(8.0, 14.0, 28.0),
    layer_wind_directions_rad=(0.0, 1.57, 3.14),
    wavelength_ref_m=500e-9,
    static=True,
)


@dataclass
class AtmosphereSpec:
    """Resolved per-layer parameters in canonical form."""
    r0_per_layer_m: tuple[float, ...]
    L0_per_layer_m: tuple[float, ...]
    heights_m: tuple[float, ...]
    wind_speeds_mps: tuple[float, ...]
    wind_directions_rad: tuple[float, ...]
    wavelength_ref_m: float
    static: bool


def _resolve_spec(cfg: dict) -> AtmosphereSpec:
    """Take a user cfg block (possibly missing keys) and fill from
    TEIDE_DEFAULT. Compute per-layer r0 from total r0 + Cn^2 weighting.
    """
    out = dict(TEIDE_DEFAULT)
    out.update({k: v for k, v in cfg.items() if v is not None})

    weights = tuple(float(w) for w in out["layer_weights"])
    total_w = sum(weights)
    if total_w <= 0:
        raise ValueError("atmosphere layer_weights must sum to a positive "
                         f"value, got {weights}")
    weights = tuple(w / total_w for w in weights)

    r0_total = float(out["r0_total_m"])
    # r0_total ** (-5/3) = sum_l (r0_l ** (-5/3)) = sum_l w_l * (r0_total ** (-5/3))
    # => r0_l ** (-5/3) = w_l * r0_total ** (-5/3)
    # => r0_l = r0_total * w_l ** (-3/5)
    r0_layers = tuple(r0_total * (w ** (-3.0 / 5.0)) for w in weights)

    # L0 may be a scalar (applied to all layers) or a per-layer tuple.
    L0_raw = out["L0_m"]
    if hasattr(L0_raw, "__len__"):
        L0_layers = tuple(float(x) for x in L0_raw)
        if len(L0_layers) != len(weights):
            raise ValueError(
                f"atmosphere L0_m length {len(L0_layers)} does not match "
                f"layer_weights length {len(weights)}")
    else:
        L0_layers = (float(L0_raw),) * len(weights)

    heights = tuple(float(h) for h in out["layer_heights_m"])
    speeds = tuple(float(s) for s in out["layer_wind_speeds_mps"])
    dirs = tuple(float(d) for d in out["layer_wind_directions_rad"])
    if len(heights) != len(weights) or len(speeds) != len(weights) or len(dirs) != len(weights):
        raise ValueError(
            "atmosphere heights/wind_speeds/wind_directions must all have "
            f"the same length as layer_weights ({len(weights)})")

    return AtmosphereSpec(
        r0_per_layer_m=r0_layers,
        L0_per_layer_m=L0_layers,
        heights_m=heights,
        wind_speeds_mps=speeds,
        wind_directions_rad=dirs,
        wavelength_ref_m=float(out["wavelength_ref_m"]),
        static=bool(out["static"]),
    )


# ---------------------------------------------------------------------------
# Phase-screen generation
# ---------------------------------------------------------------------------

def _von_karman_psd(H: int, W: int, dx_m: float, r0_m: float,
                    L0_m: float, device: torch.device) -> torch.Tensor:
    """Von Karman OPD PSD on the [H, W] FFT grid.

    Returns the *amplitude* coefficients sqrt(PSD * df^2) ready to
    multiply the unit-variance Hermitian noise. df = 1 / (N * dx) is
    the spatial-frequency increment of the FFT.
    """
    df_y = 1.0 / (H * dx_m)
    df_x = 1.0 / (W * dx_m)
    fy = torch.fft.fftfreq(H, d=dx_m, device=device).unsqueeze(1)    # [H, 1]
    fx = torch.fft.fftfreq(W, d=dx_m, device=device).unsqueeze(0)    # [1, W]
    f2 = fy * fy + fx * fx                                           # [H, W]
    # Von Karman OPD PSD in m^2 / (cycle/m)^2 -- standard form from
    # Roddier 1981 with the (lambda_ref / 2pi)^2 cancellation when we
    # convert phase -> OPD. We keep the constant 0.023 explicitly:
    #    Phi_OPD(f) = 0.023 * r0^(-5/3) * (f^2 + 1/L0^2)^(-11/6)
    # (this is the *phase* PSD measured in rad^2/(cycle/m)^2; the
    # final OPD scaling is applied via wavelength_ref/2pi below.)
    psd = 0.023 * (r0_m ** (-5.0 / 3.0)) * (f2 + 1.0 / (L0_m * L0_m)) ** (-11.0 / 6.0)
    # Kill the DC bin (constant piston is irrelevant for imaging and
    # we want zero-mean screens anyway).
    psd[0, 0] = 0.0
    # Amplitude per Fourier bin: sqrt(psd * df_y * df_x).
    return torch.sqrt(psd * df_y * df_x)                             # [H, W]


def _sample_screen_batch(amp: torch.Tensor, N: int, generator,
                         device: torch.device) -> torch.Tensor:
    """Sample N independent real phase screens given precomputed PSD amp."""
    H, W = amp.shape
    # Complex Gaussian noise, IFFT, take real part. Output variance
    # equals integrated PSD; the FFT normalization (1/(H*W)) interacts
    # with the dx^2 weighting in amp so the result is in radians of
    # phase at the reference wavelength.
    re = torch.randn(N, H, W, device=device, generator=generator)
    im = torch.randn(N, H, W, device=device, generator=generator)
    spectrum = torch.complex(re, im) * amp.unsqueeze(0)
    # Multiply by H*W to undo torch.fft's 1/N normalization: integrated
    # power should equal sum(|amp|^2) * H * W = total variance.
    screen = torch.fft.ifft2(spectrum).real * (H * W)
    return screen                                                    # [N, H, W]


# ---------------------------------------------------------------------------
# Atmosphere object
# ---------------------------------------------------------------------------

class KolmogorovAtmosphere:
    """Multi-layer Kolmogorov/von Karman atmosphere on the env's pupil.

    Currently static-only: the screen is sampled at ``reset()`` and held
    constant for the rest of the episode. ``step(dt)`` is a no-op. The
    object owns the OPD tensor; the env reads it via ``opd_meters()``.
    """

    def __init__(self, num_envs: int, H: int, W: int, dx_m: float,
                 device: torch.device, cfg: dict,
                 seed: Optional[int] = None,
                 silence: bool = False):
        self.spec = _resolve_spec(cfg)
        self.N = int(num_envs)
        self.H = int(H)
        self.W = int(W)
        self.dx_m = float(dx_m)
        self.device = device
        self.n_layers = len(self.spec.r0_per_layer_m)

        # Dedicated CUDA generator so atmosphere randomness is reproducible
        # independent of other RNG (DM noise, init perturbations, ...).
        self._gen = torch.Generator(device=device)
        if seed is not None:
            self._gen.manual_seed(int(seed))

        # Precompute per-layer PSD amplitudes. These are independent of
        # which env is being sampled, so they're shared across N.
        self._amp_per_layer = [
            _von_karman_psd(self.H, self.W, self.dx_m,
                            r0, L0, device)
            for r0, L0 in zip(self.spec.r0_per_layer_m,
                              self.spec.L0_per_layer_m)]

        # OPD storage: [N, H, W] in meters. Filled by reset().
        self._opd_m = torch.zeros(self.N, self.H, self.W,
                                  dtype=torch.float32, device=device)
        # Phase-to-OPD scale: phase (radians at lambda_ref) -> OPD (meters)
        # OPD = phase * lambda_ref / (2 pi). We apply this once when
        # summing layers into the cached OPD tensor.
        self._phase_to_opd_m = (self.spec.wavelength_ref_m
                                / (2.0 * math.pi))

        if not silence:
            print(f"[atmosphere] {self.n_layers} layer(s), "
                  f"r0_total={float(cfg.get('r0_total_m', TEIDE_DEFAULT['r0_total_m'])):.3f} m "
                  f"@ lambda_ref={self.spec.wavelength_ref_m*1e9:.0f} nm, "
                  f"L0={self.spec.L0_per_layer_m[0]:.1f} m, "
                  f"static={self.spec.static}")
            for li, (r0, h, ws) in enumerate(zip(
                    self.spec.r0_per_layer_m,
                    self.spec.heights_m,
                    self.spec.wind_speeds_mps)):
                print(f"[atmosphere]   layer {li}: r0={r0:.3f} m, "
                      f"h={h:.0f} m, wind={ws:.1f} m/s")

    # ------------------------------------------------------------------
    # Public API consumed by v5
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reset(self, env_mask: Optional[torch.Tensor] = None) -> None:
        """Sample a fresh atmosphere realization for the masked envs.

        env_mask: bool [N] tensor. None means all envs.
        """
        if env_mask is None:
            env_mask = torch.ones(self.N, dtype=torch.bool, device=self.device)
        n_reset = int(env_mask.sum().item())
        if n_reset == 0:
            return

        # Accumulate layers' phase contributions on the masked envs.
        new_opd = torch.zeros(n_reset, self.H, self.W,
                              dtype=torch.float32, device=self.device)
        for amp in self._amp_per_layer:
            phase = _sample_screen_batch(amp, n_reset, self._gen, self.device)
            new_opd = new_opd + phase * self._phase_to_opd_m
        self._opd_m[env_mask] = new_opd

    @torch.no_grad()
    def step(self, dt_s: float) -> None:
        """Advance the atmosphere by dt_s seconds.

        Static mode: no-op. The hook is here so v5 / training loops can
        call it unconditionally now, and a later non-static
        implementation will Just Work.
        """
        if self.spec.static:
            return
        # Future: shift per-layer screens by wind*dt, then re-aggregate
        # into self._opd_m. Pre-sampled wide-screen + torch.roll is the
        # cheapest implementation.
        raise NotImplementedError(
            "Time-evolving atmosphere not yet implemented; set "
            "atmosphere.static=True for now.")

    def opd_meters(self) -> torch.Tensor:
        """Current atmospheric OPD on the pupil, [N, H, W] in meters.

        Tensor is owned by the atmosphere; do not modify in place.
        """
        return self._opd_m

    # ------------------------------------------------------------------
    # Convenience for diagnostics / sweep figures
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "opd_m": self._opd_m.detach().cpu().clone(),
            "spec": self.spec,
        }
