"""Per-phase fine-tune worker for the ELF PTT bootstrap composite agent.

Called by ``finetune_agent_master.py`` either as a local subprocess
(on the master's compute node) or via sbatch on a separate node. One
invocation = one phase of fine-tuning, starting from the source
agent's ``phase_NN.pt`` checkpoint and writing into the master-owned
``phases/phase_NN/`` output dir.

Reads a recipe YAML (env_kwarg + ppo overrides) and overlays it on
top of ``ELF_BOOTSTRAP_ENV_KWARGS`` / the bootstrap PPO configs, then
hands off to ``train_ppo_optomech.run_main`` exactly the way
``train_ppo_elf_bootstrap.py`` does.

Two load modes, mutually exclusive:
  - ``--source-checkpoint <ckpt>`` (default): weights-only init, fresh
    optimizer + step counter. Standard fine-tune-from-source pattern.
  - ``--resume-from <ckpt>``: full resume -- model + optimizer +
    global_step. Used by the master's ``--resume`` mode to continue a
    fine-tune that hit total_timesteps (or SLURM wall-clock) without
    re-initialising state. The resume checkpoint usually lives under
    this same phase's prior ``ppo_optomech_*/checkpoints/latest.pt``.

Writes a ``status.json`` sentinel in the output dir on success/failure
so the master can poll for completion without parsing TB.

The source agent is NEVER written to; everything lands under the
master-supplied --output-dir.

Usage (typically not invoked directly):
    poetry run python train/ppo/finetune_agent_worker.py \\
        --source-checkpoint agents/<src>/checkpoints/phase_00.pt \\
        --phase 0 \\
        --recipe train/ppo/finetune_recipes/piston_1mm.yaml \\
        --output-dir agent_finetuning/.../phases/phase_00/ \\
        --hpc
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import yaml

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from train.ppo.train_ppo_optomech import run_main
from train.ppo.train_ppo_elf_bootstrap import (
    ELF_BOOTSTRAP_ENV_KWARGS,
    LOCAL_CONFIG as BASE_LOCAL_CONFIG,
    HPC_CONFIG as BASE_HPC_CONFIG,
)


def _write_status(output_dir: str, status: str, **extra) -> None:
    """Atomic-ish status sentinel for the master to poll."""
    os.makedirs(output_dir, exist_ok=True)
    payload = {"status": status, "ts": time.time(), **extra}
    tmp = os.path.join(output_dir, "status.json.tmp")
    final = os.path.join(output_dir, "status.json")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, final)


def _patch_config(base_cfg: dict, env_kwargs: dict,
                  ppo_overrides: dict, source_checkpoint: str,
                  resume_checkpoint: str | None = None) -> dict:
    cfg = dict(base_cfg)
    cfg["env_kwargs"] = env_kwargs
    # Resume wins over init-from: full resume restores model +
    # optimizer + global_step, so we should NOT also re-initialise
    # weights from the original source. Mutually exclusive.
    if resume_checkpoint:
        cfg["resume_from"] = resume_checkpoint
        cfg.pop("init_from", None)
    else:
        cfg["init_from"] = source_checkpoint
        cfg.pop("resume_from", None)
    # Apply recipe-level PPO overrides (LR, total_timesteps, eval flags...).
    for k, v in (ppo_overrides or {}).items():
        cfg[k] = v
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune one phase of a bootstrap composite agent.",
        add_help=False)
    parser.add_argument("--source-checkpoint", required=True,
                        help="Path to the source agent's phase_NN.pt. "
                             "Used as the init-from checkpoint when "
                             "--resume-from is NOT set. Always kept "
                             "around for status.json provenance.")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to a previous PPO checkpoint to "
                             "fully resume from (model + optimizer + "
                             "global_step). Overrides --source-"
                             "checkpoint as the actual weight source. "
                             "Used by the master's --resume mode to "
                             "continue training from this phase's own "
                             "prior ppo_optomech_*/checkpoints/"
                             "latest.pt.")
    parser.add_argument("--phase", type=int, required=True,
                        help="0-based phase index (sets bootstrap_phased_count).")
    parser.add_argument("--recipe", required=True,
                        help="YAML recipe (env_kwarg + ppo overrides).")
    parser.add_argument("--output-dir", required=True,
                        help="Per-phase output dir (master-owned).")
    parser.add_argument("-h", "--help", action="help")
    cli, remaining = parser.parse_known_args()

    os.makedirs(cli.output_dir, exist_ok=True)
    _write_status(cli.output_dir, "starting",
                  source_checkpoint=cli.source_checkpoint,
                  resume_from=cli.resume_from,
                  phase=cli.phase,
                  recipe=cli.recipe)

    # ----------------------------------------------------------------
    # Load recipe and build configs
    # ----------------------------------------------------------------
    with open(cli.recipe) as f:
        recipe = yaml.safe_load(f)

    env_kwargs = dict(ELF_BOOTSTRAP_ENV_KWARGS)
    env_kwargs["bootstrap_phased_count"] = int(cli.phase)
    env_kwargs.update(recipe.get("env_kwarg_overrides", {}))

    ppo_overrides = recipe.get("ppo_overrides", {})
    local_cfg = _patch_config(BASE_LOCAL_CONFIG, env_kwargs,
                              ppo_overrides, cli.source_checkpoint,
                              resume_checkpoint=cli.resume_from)
    hpc_cfg = _patch_config(BASE_HPC_CONFIG, env_kwargs,
                            ppo_overrides, cli.source_checkpoint,
                            resume_checkpoint=cli.resume_from)

    # ----------------------------------------------------------------
    # Hand off to run_main. We forward all remaining CLI flags (notably
    # --hpc, --seed, --no-eval) and inject --run-dir plus EITHER
    # --resume-from (full resume) OR --init-from (weights-only).
    # run_main does its own CLI parse, so we have to expose the chosen
    # flag on its arg list too.
    # ----------------------------------------------------------------
    extra_args = ["--run-dir", cli.output_dir]
    if cli.resume_from:
        extra_args += ["--resume-from", cli.resume_from]
    else:
        extra_args += ["--init-from", cli.source_checkpoint]
    sys.argv = [sys.argv[0]] + remaining + extra_args

    _write_status(cli.output_dir, "running",
                  source_checkpoint=cli.source_checkpoint,
                  resume_from=cli.resume_from,
                  phase=cli.phase)

    try:
        run_main(local_cfg, hpc_cfg)
    except SystemExit as e:
        # run_main may call sys.exit; treat 0 as success.
        if e.code not in (None, 0):
            _write_status(cli.output_dir, "failed",
                          phase=cli.phase,
                          error=f"SystemExit({e.code})")
            raise
    except Exception as e:
        _write_status(cli.output_dir, "failed",
                      phase=cli.phase,
                      error=str(e),
                      traceback=traceback.format_exc())
        raise

    # ----------------------------------------------------------------
    # On success, locate the trained checkpoint(s). run_main writes
    # under <output-dir>/ppo_optomech_<seed>_<ts>/checkpoints/. Find
    # the latest run dir's best.pt (the artifact the packer will pull
    # into the final agent bundle).
    # ----------------------------------------------------------------
    run_dirs = sorted(Path(cli.output_dir).glob("ppo_optomech_*"))
    best_ckpt = None
    if run_dirs:
        ck_dir = run_dirs[-1] / "checkpoints"
        for candidate in ("best.pt", "latest.pt"):
            p = ck_dir / candidate
            if p.is_file():
                best_ckpt = str(p)
                break

    _write_status(cli.output_dir, "completed",
                  phase=cli.phase,
                  best_checkpoint=best_ckpt or "",
                  run_dir=str(run_dirs[-1]) if run_dirs else "")


if __name__ == "__main__":
    main()
