"""
PPO training: ELF piston-only dark-hole shaping, DYNAMIC target-aware.

The dark-hole geometry is resampled from a uniform envelope at the
start of every episode. The policy's LSTM input includes the 4-vector
[sin(θ), cos(θ), radius_frac, size_frac] encoding the current target,
so one trained policy generalises across the grid instead of needing
a separate checkpoint per hole.

Sampling envelope (matches the static-grid launcher's two rings):
    angle:       uniform on [0, 360) degrees
    radius_frac: uniform on [0.16, 0.32]
    size_frac:   fixed at 0.08 (matches the tangent-rings layout)

V5 now stores a per-env hole mask tensor and resamples on reset, so
HPC training runs on the full batched GPU env without falling back to
v4.

Usage:
    python train/ppo/train_ppo_elf_dark_hole_dynamic.py --seed 7
    python train/ppo/train_ppo_elf_dark_hole_dynamic.py --hpc --seed 7
"""
import argparse
import sys

from train.ppo.train_ppo_optomech import run_main
from train.ppo.train_ppo_elf_dark_hole import (
    ELF_DARK_HOLE_ENV_KWARGS,
    LOCAL_CONFIG as BASE_LOCAL_CONFIG,
    HPC_CONFIG as BASE_HPC_CONFIG,
)


# Build the env-kwargs copy with dynamic-reset flags baked in. Keep the
# placeholder geometry fields — they are overwritten on every reset.
DYNAMIC_ENV_KWARGS = dict(ELF_DARK_HOLE_ENV_KWARGS)
DYNAMIC_ENV_KWARGS.update({
    "dark_hole": True,
    "dark_hole_randomize_on_reset": True,
    "dark_hole_angle_range_deg": (0.0, 360.0),
    "dark_hole_radius_range": (0.16, 0.32),
    "dark_hole_size_range": (0.08, 0.08),
})

LOCAL_CONFIG = dict(BASE_LOCAL_CONFIG)
LOCAL_CONFIG["env_kwargs"] = DYNAMIC_ENV_KWARGS
LOCAL_CONFIG["target_dim"] = 4

HPC_CONFIG = dict(BASE_HPC_CONFIG)
HPC_CONFIG["env_kwargs"] = DYNAMIC_ENV_KWARGS
HPC_CONFIG["target_dim"] = 4


if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--seed", type=int, default=None,
        help="Override the PPO seed (default: from config = 1).")
    pre_args, remaining = pre_parser.parse_known_args()

    for cfg in (LOCAL_CONFIG, HPC_CONFIG):
        cfg["env_kwargs"] = dict(cfg["env_kwargs"])
        if pre_args.seed is not None:
            cfg["seed"] = int(pre_args.seed)

    sys.argv = [sys.argv[0]] + remaining
    run_main(LOCAL_CONFIG, HPC_CONFIG)
