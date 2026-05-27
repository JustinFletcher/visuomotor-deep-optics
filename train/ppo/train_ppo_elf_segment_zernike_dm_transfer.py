"""Transfer-learn each bootstrap phase from segment-PTT-only to
segment-PTT + Zernike-DM correction under a random Kolmogorov atmosphere.

Starts from a phase_NN.pt checkpoint of the existing bootstrap agent
(45-DOF policy head, segment-piston-tip-tilt only) and re-trains it
in an env that:

  * still has the same 15-segment PTT action surface
    (command_secondaries=True, max_piston/tip/tilt unchanged),
  * additionally enables the DM (command_dm=True,
    dm_incremental_control=True so DM commands are deltas, like the
    dm_atmos training family), and
  * applies a fresh Kolmogorov atmosphere realisation at every reset
    with r0 sampled uniformly from a configurable range (default
    10-25 cm at 500 nm).

The env is wrapped with ``SegmentZernikeDMVectorEnv``, which exposes
``[seg_PTT_45 | zernike_N]`` to the policy: the first 45 channels pass
through to the env's segment slots; the remaining N channels are
projected via M [n_dm, N] to DM actuator commands and scaled by
``dm_action_scale`` before clamping. The agent's policy head grows
from 45 to 45 + N -- the new N channels are re-initialised from
scratch because their slot in the source state-dict has the wrong
shape (or doesn't exist at all). The segment policy weights, the
encoder, the LSTM hidden-hidden weights, and the value head are all
transferred verbatim.

Re-initialised on transfer (shape depends on the new wider action_dim):
  * ``policy_head.*``      -- final Linear out = 45 + n_zernike
  * ``log_std``            -- shape [45 + n_zernike]
  * ``action_low``         -- buffer, shape [45 + n_zernike]
  * ``action_high``        -- buffer, shape [45 + n_zernike]
  * ``lstm.weight_ih_l*``  -- prior_action is concatenated into the
                              LSTM input

Typical command (per-phase, single GPU):

    poetry run python \\
        train/ppo/train_ppo_elf_segment_zernike_dm_transfer.py \\
        --hpc \\
        --source-checkpoint agents/agent_..._/checkpoints/phase_03.pt \\
        --phased-count 3 \\
        --n-zernike 32 \\
        --r0-range 0.10 0.25 \\
        --run-dir agent_finetuning/segment_dm_v2/phase_03/

A launcher script (launch_segment_zernike_dm_transfer.py) fans this
out across 15 phases / multiple nodes.
"""
from __future__ import annotations

import argparse
import sys

from train.ppo.train_ppo_optomech import run_main
from train.ppo.train_ppo_elf_bootstrap import (
    ELF_BOOTSTRAP_ENV_KWARGS,
    LOCAL_CONFIG as BASE_LOCAL_CONFIG,
    HPC_CONFIG as BASE_HPC_CONFIG,
)


# ----------------------------------------------------------------------
# Env kwargs: ELF bootstrap + DM + atmosphere
# ----------------------------------------------------------------------
ENV_KWARGS = dict(ELF_BOOTSTRAP_ENV_KWARGS)

# Keep segments as before (45 DOF, max_piston/tip/tilt from bootstrap).
# Add DM commanding in incremental-delta mode so segment phases (which
# don't get added here -- single policy now controls everything) and
# the dm head can coexist without each step wiping the other's state.
ENV_KWARGS["command_dm"] = True
ENV_KWARGS["dm_incremental_control"] = True

# Atmosphere: random r0 per reset (default 10-25 cm at 500 nm).
# Static-per-episode for now; the spec already accepts a range and the
# atmosphere module rolls a fresh r0 + screen at each reset call.
ENV_KWARGS["atmosphere"] = dict(
    r0_total_m_range=[0.10, 0.25],
    L0_m=25.0,
    static=True,
)


# ----------------------------------------------------------------------
# State-dict keys whose shape depends on action_dim. The source
# bootstrap policy has action_dim=45; the new policy is 45+n_zernike,
# so these keys are filtered out at load time and re-initialised.
# The rest (encoder, LSTM hidden-hidden, value head) transfer verbatim.
# ----------------------------------------------------------------------
_TRANSFER_SKIP_PREFIXES = (
    "policy_head",
    "log_std",
    "action_low",
    "action_high",
    "lstm.weight_ih_l",
)


def _patch(cfg):
    """Overlay transfer-learning bits on top of the bootstrap config.

    Engages:
      * the SegmentZernikeDMVectorEnv wrapper at training and eval,
      * init_from_skip_prefixes so the action-dim-changed parameters
        get re-initialised instead of raising a state_dict shape error,
      * a DM-only action_scale knob (default 0.01) that matches the
        dm_atmos training family's per-step delta cap.
    """
    cfg = dict(cfg)
    cfg["env_kwargs"] = ENV_KWARGS
    cfg["segment_zernike_dm"] = True
    cfg["zernike_n_modes"] = 32
    cfg["zernike_dm_action_scale"] = 0.01
    cfg["zernike_normalize"] = "wavefront_rms"
    cfg["init_from_skip_prefixes"] = list(_TRANSFER_SKIP_PREFIXES)
    # Effectively indefinite per Fletcher's standing preference:
    # runs are bounded by SLURM --time, not by step count. 10B steps
    # at typical fine-tune SPS is weeks of wall clock, so the loop
    # never finishes on its own; the master cancels / SLURM kills it
    # eventually.
    cfg["total_timesteps"] = 10_000_000_000
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
        help="Alias for --init-from. Source bootstrap phase checkpoint, "
             "e.g. agents/agent_..._/checkpoints/phase_NN.pt.")
    pre.add_argument(
        "--phased-count", type=int, default=None,
        help="Bootstrap phase index (0..14). Sets env_kwargs."
             "bootstrap_phased_count so the env reproduces phase N's "
             "training-time start state: segs 0..N-1 aligned, seg N "
             "perturbed, segs N+1..14 off-axis.")
    pre.add_argument(
        "--n-zernike", type=int, default=None,
        help="Number of Noll-indexed Zernike modes for the DM head "
             "(default 32). Larger means more representational power "
             "for the DM correction; bench-deployment side has to "
             "consume the same M projection.")
    pre.add_argument(
        "--dm-action-scale", type=float, default=None,
        help="Per-step DM delta cap as a fraction of stroke. Default "
             "0.01 matches the zernike-atm-transfer training family.")
    pre.add_argument(
        "--r0-range", type=float, nargs=2, default=None,
        metavar=("LOW", "HIGH"),
        help="Per-episode r0 sampling range in meters at 500 nm "
             "(default 0.10 0.25). Each reset draws u ~ U(LOW, HIGH) "
             "and rebuilds the atmosphere amps.")
    pre.add_argument(
        "--seed", type=int, default=None)
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
        if pre_args.phased_count is not None:
            cfg["env_kwargs"]["bootstrap_phased_count"] = int(
                pre_args.phased_count)
        if pre_args.n_zernike is not None:
            cfg["zernike_n_modes"] = int(pre_args.n_zernike)
        if pre_args.dm_action_scale is not None:
            cfg["zernike_dm_action_scale"] = float(
                pre_args.dm_action_scale)
        if pre_args.r0_range is not None:
            cfg["env_kwargs"]["atmosphere"]["r0_total_m_range"] = [
                float(pre_args.r0_range[0]),
                float(pre_args.r0_range[1])]
        if pre_args.seed is not None:
            cfg["seed"] = int(pre_args.seed)

    sys.argv = [sys.argv[0]] + remaining
    run_main(LOCAL_CONFIG, HPC_CONFIG)
