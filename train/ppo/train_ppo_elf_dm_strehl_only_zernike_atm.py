"""Zernike-coefficient strehl-only with KolmogorovAtmosphere disturbance.

Same control surface as ``train_ppo_elf_dm_strehl_only_zernike.py``
(Zernike-coefficient action wrapper over the full DM, plain Strehl
reward), but the disturbance the agent has to correct is a static
multi-layer Kolmogorov atmosphere instead of per-actuator white noise.

The motivating diagnostic was that per-actuator-iid init noise is
white in spatial frequency, but Zernike modes pile spectral energy at
the pupil rim and have power-law falloff. So a Zernike-action agent
*cannot* fit the disturbance regardless of basis size -- there's no
truncation of Z that captures white-noise variance efficiently. A
Kolmogorov screen is low-frequency-dominant by construction: ~87% of
its variance lives in the first 32 Noll modes, ~96% in the first 128
(measured on a fixed realization in /tmp/kolmogorov_zernike_coefs.py).
This is the basis-matched setting where PPO has a fighting chance.

Eval intentionally renders FIGURES + FILMSTRIPS each interval (one
episode) so we can visually inspect the policy's behaviour each
checkpoint. eval_fast=False, eval_episodes=1.
"""
import argparse
import sys

from train.ppo.train_ppo_optomech import run_main
from train.ppo.train_ppo_elf_dm_strehl_only_zernike import (
    ENV_KWARGS as ZERNIKE_ENV_KWARGS,
    LOCAL_CONFIG as BASE_LOCAL_CONFIG,
    HPC_CONFIG as BASE_HPC_CONFIG,
)


# ----------------------------------------------------------------------
# Env kwargs
# ----------------------------------------------------------------------
ENV_KWARGS = dict(ZERNIKE_ENV_KWARGS)

# Remove the per-actuator IID DM init noise; the atmosphere is the
# disturbance now. Leaving init_dm_micron_std non-zero would just
# inject Zernike-unfit white noise on top of the Zernike-friendly
# atmosphere, conflating two disturbance types.
ENV_KWARGS["init_dm_micron_std"] = 0.0

# Atmosphere: Mt. Teide-typical defaults from atmosphere.TEIDE_DEFAULT
# with r0 explicitly settable here so it's visible in the run config.
# Static-per-episode for now; evolving mode is plumbed but not
# implemented (see optomech/optomech/atmosphere.py).
ENV_KWARGS["atmosphere"] = dict(
    r0_total_m=0.12,                                  # 12 cm @ 500 nm
    L0_m=25.0,
    static=True,
)


# ----------------------------------------------------------------------
# PPO config: same as the parent + eval-image overrides
# ----------------------------------------------------------------------
def _patch(cfg):
    cfg = dict(cfg)
    cfg["env_kwargs"] = ENV_KWARGS
    # Re-enable image rendering on the in-training eval so we can
    # peek at the rollout filmstrip + summary figure each interval.
    # Single episode keeps wall-clock manageable on HPC.
    cfg["eval_fast"] = False
    cfg["eval_episodes"] = 1
    cfg["no_eval"] = False
    return cfg


LOCAL_CONFIG = _patch(BASE_LOCAL_CONFIG)
HPC_CONFIG = _patch(BASE_HPC_CONFIG)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--r0", type=float, default=None,
        help="Override the atmosphere r0_total_m (meters at 500 nm). "
             "Smaller r0 = worse seeing. Teide range ~0.05 to 0.25.")
    pre_parser.add_argument(
        "--L0", type=float, default=None,
        help="Override the von Karman outer scale L0 (meters).")
    pre_parser.add_argument(
        "--seed", type=int, default=None,
        help="Override the PPO seed (default: from config).")
    pre_parser.add_argument(
        "--n-zernike", type=int, default=None,
        help="Override the Zernike basis size (default 24 inherited).")
    pre_args, remaining = pre_parser.parse_known_args()

    for cfg in (LOCAL_CONFIG, HPC_CONFIG):
        cfg["env_kwargs"] = dict(cfg["env_kwargs"])
        cfg["env_kwargs"]["atmosphere"] = dict(cfg["env_kwargs"]["atmosphere"])
        if pre_args.r0 is not None:
            cfg["env_kwargs"]["atmosphere"]["r0_total_m"] = float(pre_args.r0)
        if pre_args.L0 is not None:
            cfg["env_kwargs"]["atmosphere"]["L0_m"] = float(pre_args.L0)
        if pre_args.seed is not None:
            cfg["seed"] = int(pre_args.seed)
        if pre_args.n_zernike is not None:
            cfg["zernike_n_modes"] = int(pre_args.n_zernike)

    sys.argv = [sys.argv[0]] + remaining
    run_main(LOCAL_CONFIG, HPC_CONFIG)
