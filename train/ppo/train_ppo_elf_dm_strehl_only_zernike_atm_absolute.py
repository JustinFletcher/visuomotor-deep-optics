"""Transfer-learn an atmospheric Zernike-action policy to ABSOLUTE DM control.

Same env, same reward, same action representation (Zernike via
ZernikeDMVectorEnv), same architecture as
``train_ppo_elf_dm_strehl_only_zernike_atm.py`` -- the only
differences are two env_kwargs flipped in ``_patch``:

  - ``dm_incremental_control``  True  -> False  (absolute DM control)
  - ``env_action_scale``        0.01  -> 1.0    (full DM stroke per
                                                  unit action)

Action semantics shift: under incremental control the agent's action
was a per-step DM DELTA (~0.015 µm/step at env_action_scale=0.01),
which accumulates across the episode; under absolute control the same
action is the FULL DM STATE for that step (~1.5 µm/unit-action at
env_action_scale=1.0), no integration. The same policy weights
therefore produce very different physical commands, but the
architecture is identical so weights load 1-to-1 (no
init_from_skip_prefixes -- the policy_head, log_std, action_low/high,
and lstm.weight_ih_l* all match shape because n_zernike is unchanged).

PPO fine-tuning adapts the policy to the new action-meaning via
reward feedback. Typically the agent learns to emit much smaller
Zernike coefficients than it did under incremental control (target
~0.1 µm OPD / 1.5 µm = 0.07 action magnitude instead of the
~0.5-saturated values it was producing as accumulator deltas).

Usage (typically via launch_zernike_atm_transfer.py with
``--train-script train/ppo/train_ppo_elf_dm_strehl_only_zernike_atm_
absolute.py`` rather than by hand):

    poetry run python \\
        train/ppo/train_ppo_elf_dm_strehl_only_zernike_atm_absolute.py \\
        --hpc \\
        --source-checkpoint atmos_models/r0_25cm_seed.../checkpoints/latest.pt \\
        --r0 0.25 \\
        --run-dir atmos_finetuning/absolute_n24/
"""
import argparse
import sys

from train.ppo.train_ppo_optomech import run_main
from train.ppo.train_ppo_elf_dm_strehl_only_zernike_atm import (
    ENV_KWARGS as BASE_ENV_KWARGS,
    LOCAL_CONFIG as BASE_LOCAL_CONFIG,
    HPC_CONFIG as BASE_HPC_CONFIG,
)


ENV_KWARGS = dict(BASE_ENV_KWARGS)

# Switch the env's DM control surface from incremental to absolute.
# Each step now writes the full DM state from a * env_action_scale *
# stroke_limit instead of accumulating deltas.
ENV_KWARGS["dm_incremental_control"] = False

# Lift env_action_scale from 0.01 (incremental-friendly per-step
# delta cap) to 1.0 (natural absolute-control range: action 1.0 ->
# full DM stroke). This is the per-DOF multiplier applied BEFORE
# the env's stroke clip; with 1.0 the agent's [-1, 1] range maps
# directly to [-stroke, +stroke] of DM displacement.
ENV_KWARGS["env_action_scale"] = 1.0


def _patch(cfg):
    """Overlay absolute-control env on top of the parent atm config.

    Note: NO init_from_skip_prefixes here. n_zernike is unchanged, so
    the source checkpoint's action-head and LSTM input weights match
    shape exactly and transfer verbatim. PPO adapts the action-meaning
    via reward feedback alone.
    """
    cfg = dict(cfg)
    cfg["env_kwargs"] = ENV_KWARGS
    return cfg


LOCAL_CONFIG = _patch(BASE_LOCAL_CONFIG)
HPC_CONFIG = _patch(BASE_HPC_CONFIG)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
if __name__ == "__main__":
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--source-checkpoint", type=str, default=None,
        help="Alias for --init-from for readability. If both are "
             "given --source-checkpoint wins.")
    pre.add_argument(
        "--r0", type=float, default=None,
        help="Override atmosphere r0_total_m at runtime.")
    pre.add_argument(
        "--seed", type=int, default=None)
    # --n-zernike intentionally NOT exposed here: this experiment is
    # specifically the same-architecture transfer (incremental ->
    # absolute) on the source's existing basis size. Use the
    # basis-change transfer script for n_zernike sweeps.
    pre_args, remaining = pre.parse_known_args()

    if pre_args.source_checkpoint:
        cleaned = []
        skip_next = False
        for tok in remaining:
            if skip_next:
                skip_next = False
                continue
            if tok == "--init-from":
                skip_next = True
                continue
            if tok.startswith("--init-from="):
                continue
            cleaned.append(tok)
        remaining = cleaned + ["--init-from", pre_args.source_checkpoint]

    for cfg in (LOCAL_CONFIG, HPC_CONFIG):
        cfg["env_kwargs"] = dict(cfg["env_kwargs"])
        cfg["env_kwargs"]["atmosphere"] = dict(
            cfg["env_kwargs"]["atmosphere"])
        if pre_args.r0 is not None:
            cfg["env_kwargs"]["atmosphere"]["r0_total_m"] = float(pre_args.r0)
        if pre_args.seed is not None:
            cfg["seed"] = int(pre_args.seed)

    sys.argv = [sys.argv[0]] + remaining
    run_main(LOCAL_CONFIG, HPC_CONFIG)
