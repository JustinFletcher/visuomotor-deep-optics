"""PPO training: dark-hole shaping with the curriculum, but no
bilateral wrapper at all.

Companion debug variant of
``train_ppo_elf_dm_dark_hole_curriculum.py``. Same env (absolute DM,
50 nm symmetric init perturbation, log-contrast curriculum on top of
centered-Strehl) and same schedule (1M warmup, 10M ramp to weight
0.2), but:

  * ``bilateral_dm`` is OFF. The policy sees the full action space
    (15 segment DOFs + 1225 DM actuators = 1240 dims) and the
    full unmasked focal-plane observation. No symmetric expansion
    of actions, no blind-region mask on the obs.

  * Entropy regulation is tuned up. ``ent_coef`` rises from 1e-6 to
    1e-5 (10x), and ``log_std_max`` is relaxed from -2.0 to -1.5
    (per-dim sigma cap rises from 0.135 to 0.223). The bilateral
    track was converging onto a brittle low-entropy pattern just
    before the KL explosion, so this gives the policy more room
    to keep its sample distribution broad through the curriculum
    transition.

Segments are NOT explicitly frozen here because the wrapper is off;
the env's standard segment-piston path is active. With the init
perturbation only on the DM and zero init perturbation on segments,
the optimum still has segments at zero, and the policy is expected
to learn that. The extra 15 segment DOFs add a small amount of
exploration noise but should not be a fundamental obstacle.

The point of this variant is to isolate whether the curriculum
failure was caused by (a) the wrapper/symmetry/mask itself or
(b) the PPO + log-contrast setup at this action dimensionality.
If this variant succeeds and the bilateral one does not, (a) is
the issue; if both fail, (b) is, and we need to look further
upstream.
"""
import argparse
import sys

from train.ppo.train_ppo_optomech import run_main
from train.ppo.train_ppo_elf_dm_dark_hole_curriculum import (
    ENV_KWARGS as BASE_ENV_KWARGS,
    LOG_CONTRAST_CURRICULUM,
    LOCAL_CONFIG as BASE_LOCAL_CONFIG,
    HPC_CONFIG as BASE_HPC_CONFIG,
)


# ----------------------------------------------------------------------
# Env kwargs (inherit dark-hole + DM + reward composition from the
# bilateral curriculum script; we only touch wrapper-related toggles
# at the PPO-config level below).
# ----------------------------------------------------------------------
ENV_KWARGS = dict(BASE_ENV_KWARGS)


# ----------------------------------------------------------------------
# PPO config patch
# ----------------------------------------------------------------------
def _patch(cfg):
    cfg = dict(cfg)
    cfg["env_kwargs"] = ENV_KWARGS
    cfg["target_dim"] = 4
    # Wrapper OFF: no symmetric action expansion, no blind mask on obs.
    cfg["bilateral_dm"] = False
    # bilateral_freeze_segments has no effect when bilateral_dm is False,
    # but stripping it removes config noise.
    cfg.pop("bilateral_freeze_segments", None)
    cfg.pop("bilateral_dm_mode", None)

    # Entropy regulation tuned up: previous run converged on a brittle
    # low-entropy pattern (H=-790 at the KL-explosion step). 10x larger
    # entropy bonus + a more permissive log_std_max keep the action
    # distribution wider through the curriculum transition.
    cfg["ent_coef"] = 1e-5
    cfg["log_std_max"] = -1.5     # per-dim sigma cap exp(-1.5) ~ 0.223
    cfg["log_std_min"] = -5.0     # floor unchanged
    cfg["init_log_std"] = -2.5    # init unchanged

    cfg["reward_weight_curriculum"] = LOG_CONTRAST_CURRICULUM
    return cfg


LOCAL_CONFIG = _patch(BASE_LOCAL_CONFIG)
HPC_CONFIG = _patch(BASE_HPC_CONFIG)


if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--dark-hole-angle", type=float, default=None)
    pre_parser.add_argument("--dark-hole-radius-frac", type=float, default=None)
    pre_parser.add_argument("--dark-hole-size", type=float, default=None)
    pre_parser.add_argument("--curriculum-warmup", type=int, default=None)
    pre_parser.add_argument("--curriculum-anneal", type=int, default=None)
    pre_parser.add_argument("--ent-coef", type=float, default=None)
    pre_parser.add_argument("--log-std-max", type=float, default=None)
    pre_parser.add_argument("--seed", type=int, default=None)
    pre_args, remaining = pre_parser.parse_known_args()

    for cfg in (LOCAL_CONFIG, HPC_CONFIG):
        cfg["env_kwargs"] = dict(cfg["env_kwargs"])
        cfg["reward_weight_curriculum"] = dict(cfg["reward_weight_curriculum"])
        if pre_args.dark_hole_angle is not None:
            cfg["env_kwargs"]["dark_hole_angular_location_degrees"] = float(
                pre_args.dark_hole_angle)
        if pre_args.dark_hole_radius_frac is not None:
            cfg["env_kwargs"]["dark_hole_location_radius_fraction"] = float(
                pre_args.dark_hole_radius_frac)
        if pre_args.dark_hole_size is not None:
            cfg["env_kwargs"]["dark_hole_size_radius"] = float(
                pre_args.dark_hole_size)
        if pre_args.curriculum_warmup is not None:
            cfg["reward_weight_curriculum"]["warmup_timesteps"] = int(
                pre_args.curriculum_warmup)
        if pre_args.curriculum_anneal is not None:
            cfg["reward_weight_curriculum"]["anneal_timesteps"] = int(
                pre_args.curriculum_anneal)
        if pre_args.ent_coef is not None:
            cfg["ent_coef"] = float(pre_args.ent_coef)
        if pre_args.log_std_max is not None:
            cfg["log_std_max"] = float(pre_args.log_std_max)
        if pre_args.seed is not None:
            cfg["seed"] = int(pre_args.seed)

    sys.argv = [sys.argv[0]] + remaining
    run_main(LOCAL_CONFIG, HPC_CONFIG)
