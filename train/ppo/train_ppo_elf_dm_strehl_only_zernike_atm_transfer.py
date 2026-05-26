"""Transfer-learn an atmospheric DM correction policy across Zernike basis sizes.

Same env + reward as ``train_ppo_elf_dm_strehl_only_zernike_atm.py`` --
DM-only action via the ZernikeDMVectorEnv wrapper, static Kolmogorov
atmosphere, plain Strehl reward, full eval instrumentation. The
single difference is the warm start: ``--init-from <checkpoint>``
loads a source policy trained at one ``n_zernike`` and re-initialises
the action-dim-dependent parameters so the new policy can adapt to a
different ``n_zernike``.

Re-initialised on transfer (shape changes with action_dim):
  - ``policy_head.*``      -- final Linear out = action_dim
  - ``log_std``            -- per-dim policy noise, shape [action_dim]
  - ``action_low``         -- buffer, shape [action_dim]
  - ``action_high``        -- buffer, shape [action_dim]
  - ``lstm.weight_ih_l*``  -- prior_action is concatenated into the
                              LSTM input, so input-hidden weight
                              shape depends on action_dim

Transferred verbatim (action_dim-independent):
  - encoder (CNN + projection MLP) -- the main visual representation
  - LSTM hidden-hidden weights (recurrent dynamics)
  - LSTM biases
  - value_head (scalar output)
  - target_vec projection (unused here, target_dim=0 anyway)

Usage (typically via launch_zernike_atm_transfer.py rather than by
hand):

    poetry run python \\
        train/ppo/train_ppo_elf_dm_strehl_only_zernike_atm_transfer.py \\
        --hpc \\
        --init-from atmos_models/r0_25cm_seed.../checkpoints/latest.pt \\
        --n-zernike 64 \\
        --run-dir agent_finetuning/zernike_transfer_.../n64
"""
import argparse
import sys

from train.ppo.train_ppo_optomech import run_main
from train.ppo.train_ppo_elf_dm_strehl_only_zernike_atm import (
    ENV_KWARGS as BASE_ENV_KWARGS,
    LOCAL_CONFIG as BASE_LOCAL_CONFIG,
    HPC_CONFIG as BASE_HPC_CONFIG,
)


# Same env as the parent atmosphere script. We make our own dict so
# downstream patches don't leak into the parent's globals.
ENV_KWARGS = dict(BASE_ENV_KWARGS)


# These are the state-dict prefixes that change shape when action_dim
# (= n_zernike for this experiment) changes. ``init_from`` in
# train_ppo_optomech.run_main loads with strict=False whenever this
# list is non-empty, so any other keys in the source whose names
# happen to NOT match are also tolerated. The fresh-init schedule
# uses the model class's own initialisers (LSTM xavier-ish from
# torch, policy_head's `layer_init(std=0.01)` for the action head,
# init_log_std for the log_std parameter).
_TRANSFER_SKIP_PREFIXES = (
    "policy_head",          # final action-mean head
    "log_std",              # per-dim policy noise (shape [action_dim])
    "action_low",           # buffer (shape [action_dim])
    "action_high",          # buffer (shape [action_dim])
    "lstm.weight_ih_l",     # LSTM input weights (input dim depends on
                            # action_dim via prior_action concatenation)
)


def _patch(cfg):
    """Overlay transfer-learning bits on top of the parent atm config.

    All other knobs (learning_rate, total_timesteps, eval_*, zernike_*
    plumbing) are inherited from the parent so this experiment is a
    pure superset of the atmosphere zernike training run.
    """
    cfg = dict(cfg)
    cfg["env_kwargs"] = ENV_KWARGS
    # Engaging init_from_skip_prefixes flips run_main's init_from path
    # to strict=False + filtered load. Source weights for unmatched
    # keys get re-initialised by the model constructor.
    cfg["init_from_skip_prefixes"] = list(_TRANSFER_SKIP_PREFIXES)
    return cfg


LOCAL_CONFIG = _patch(BASE_LOCAL_CONFIG)
HPC_CONFIG = _patch(BASE_HPC_CONFIG)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
if __name__ == "__main__":
    pre = argparse.ArgumentParser(add_help=False)
    # --init-from is forwarded to run_main verbatim (it owns that flag
    # in train_ppo_optomech). We don't intercept it here; we just want
    # to make it clear in --help that this script needs one.
    pre.add_argument(
        "--source-checkpoint", type=str, default=None,
        help="Alias for --init-from for readability. If both are "
             "given --source-checkpoint wins.")
    pre.add_argument(
        "--r0", type=float, default=None,
        help="Override atmosphere r0_total_m at runtime (forwarded to "
             "the parent script's logic via env_kwargs.atmosphere).")
    pre.add_argument(
        "--n-zernike", type=int, default=None,
        help="Target Zernike basis size for the policy. The source "
             "checkpoint can have any n_zernike; transfer re-inits "
             "the action-dim-dependent parameters.")
    pre.add_argument(
        "--seed", type=int, default=None)
    pre_args, remaining = pre.parse_known_args()

    # --source-checkpoint -> --init-from rewrite.
    if pre_args.source_checkpoint:
        # If the user also passed --init-from in remaining, drop it so
        # --source-checkpoint wins.
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
        if pre_args.n_zernike is not None:
            cfg["zernike_n_modes"] = int(pre_args.n_zernike)

    sys.argv = [sys.argv[0]] + remaining
    run_main(LOCAL_CONFIG, HPC_CONFIG)
