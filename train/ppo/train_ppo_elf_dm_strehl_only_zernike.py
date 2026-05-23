"""Zernike-coefficient strehl-only sanity-check experiment.

Same env as ``train_ppo_elf_dm_strehl_only_full.py`` (full DM, no
bilateral wrapper, no dark hole, plain strehl reward) but the policy
operates in a low-dimensional Zernike-coefficient action space instead
of 1225 raw DM actuators. A ``ZernikeDMVectorEnv`` wrapper holds a
fixed [n_dm, n_zernike] projection matrix and projects the policy
output to the env's native action vector each step. The env never
sees a Zernike action; it still receives a full DM-action vector. The
same projection can be exported (via ``wrapper.projection_matrix()``)
to operator-side bench code so that on-orbit / on-bench deployment is
mathematically identical to simulation.

Premise: the run series 1779084109 / 1779085555 / 1779089444 showed
plain PPO can't learn closed-loop DM control on 1225 raw action dims
with the recurrent shared-encoder architecture in use here -- the
policy mean drifts within an episode regardless of sigma envelope,
action penalty, or entropy settings. Reducing the policy dim to
24-36 Zernike modes brings the task into a regime where vanilla PPO
typically converges, while preserving the env's existing DM control
surface.
"""
import argparse
import sys

from train.ppo.train_ppo_optomech import run_main
from train.ppo.train_ppo_elf_dm_strehl_only_full import (
    ENV_KWARGS as FULL_ENV_KWARGS,
    LOCAL_CONFIG as BASE_LOCAL_CONFIG,
    HPC_CONFIG as BASE_HPC_CONFIG,
)


# ----------------------------------------------------------------------
# Env kwargs: identical to the raw-actuator full-DM run.
# ----------------------------------------------------------------------
ENV_KWARGS = dict(FULL_ENV_KWARGS)


# ----------------------------------------------------------------------
# PPO config
# ----------------------------------------------------------------------
def _patch(cfg):
    cfg = dict(cfg)
    cfg["env_kwargs"] = ENV_KWARGS
    # Engage the Zernike wrapper. n_zernike picks the policy dim;
    # 24 covers through 5th radial order (defocus + astig + coma +
    # trefoil + spherical + 2nd-astig + ...) which spans the modes
    # an init_dm_micron_std=0.05 random-per-actuator perturbation
    # projects onto with non-negligible amplitude.
    cfg["zernike_dm"] = True
    cfg["zernike_n_modes"] = 24
    # Keep piston in the basis: it's the only mode that can correct a
    # uniform DM offset, and our random init has a nonzero mean draw
    # per realisation.
    cfg["zernike_skip_piston"] = False
    cfg["zernike_freeze_segments"] = True
    cfg["zernike_normalize"] = "inf"
    return cfg


LOCAL_CONFIG = _patch(BASE_LOCAL_CONFIG)
HPC_CONFIG = _patch(BASE_HPC_CONFIG)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--init-dm-micron-std", type=float, default=None,
        help="Override per-actuator std (microns) of the random init "
             "DM perturbation.")
    pre_parser.add_argument(
        "--seed", type=int, default=None,
        help="Override the PPO seed (default: from config).")
    pre_parser.add_argument(
        "--n-zernike", type=int, default=None,
        help="Override the Zernike basis size (default 24).")
    pre_args, remaining = pre_parser.parse_known_args()

    for cfg in (LOCAL_CONFIG, HPC_CONFIG):
        cfg["env_kwargs"] = dict(cfg["env_kwargs"])
        if pre_args.init_dm_micron_std is not None:
            cfg["env_kwargs"]["init_dm_micron_std"] = float(
                pre_args.init_dm_micron_std)
        if pre_args.seed is not None:
            cfg["seed"] = int(pre_args.seed)
        if pre_args.n_zernike is not None:
            cfg["zernike_n_modes"] = int(pre_args.n_zernike)

    sys.argv = [sys.argv[0]] + remaining
    run_main(LOCAL_CONFIG, HPC_CONFIG)
