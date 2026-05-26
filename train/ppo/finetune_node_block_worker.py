"""Node-block worker: run K parallel per-phase fine-tunes on one node.

Submitted as a single SLURM job by the fine-tune master against a
specific node (--nodelist + --gres=gpu:K). Inside the job, this
script reads a manifest file describing K phases to fine-tune,
spawns K parallel ``finetune_agent_worker.py`` subprocesses each
pinned to one local GPU via CUDA_VISIBLE_DEVICES, waits for all
to finish, and writes its own block-level status sentinel.

Rationale: the target cluster does not reliably oversubscribe
nodes, so submitting K independent --gres=gpu:1 jobs to the same
node may queue them sequentially instead of running them
concurrently. Submitting one --gres=gpu:K job that internally
fans out gives us the parallelism without depending on partition
config.

Manifest schema (JSON file passed via --manifest):
  [
    {"phase": 0,
     "source_checkpoint": "...",
     "output_dir": "...",
     "resume_checkpoint": "..." (optional; if present the inner worker
                                  is run in full-resume mode instead of
                                  init-from-source)},
    ...
  ]

Each entry's per-phase status.json is written by the inner
finetune_agent_worker.py exactly as in the single-phase case, so
the master can poll those files unchanged. The block-level
sentinel ``block_status.json`` lands in ``--block-dir``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _write_block_status(block_dir: str, status: str, **extra) -> None:
    os.makedirs(block_dir, exist_ok=True)
    payload = {"status": status, "ts": time.time(), **extra}
    tmp = os.path.join(block_dir, "block_status.json.tmp")
    final = os.path.join(block_dir, "block_status.json")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, final)


def main():
    parser = argparse.ArgumentParser(
        description="Run K parallel per-phase fine-tunes on one node.")
    parser.add_argument("--manifest", required=True,
                        help="JSON file listing phases to run in parallel.")
    parser.add_argument("--recipe", required=True,
                        help="Recipe YAML forwarded to every sub-worker.")
    parser.add_argument("--block-dir", required=True,
                        help="Per-block log + sentinel dir.")
    args = parser.parse_args()

    os.makedirs(args.block_dir, exist_ok=True)
    _write_block_status(args.block_dir, "starting",
                        manifest=args.manifest)

    with open(args.manifest) as f:
        entries = json.load(f)
    if not entries:
        _write_block_status(args.block_dir, "completed",
                            note="empty manifest")
        return 0

    # ----------------------------------------------------------------
    # Spawn one inner worker per entry, pinned to a unique local GPU.
    # Each entry's status.json is written by finetune_agent_worker.py
    # into its own output_dir, so the master can poll them directly.
    # ----------------------------------------------------------------
    procs = []
    for gpu_idx, entry in enumerate(entries):
        phase = int(entry["phase"])
        src = entry["source_checkpoint"]
        outdir = entry["output_dir"]
        resume_ckpt = entry.get("resume_checkpoint") or None
        os.makedirs(outdir, exist_ok=True)

        out_log = open(os.path.join(outdir, "slurm_inner.out"), "w")
        err_log = open(os.path.join(outdir, "slurm_inner.err"), "w")
        cmd = [
            sys.executable, "-u",
            os.path.join(_REPO_ROOT, "train", "ppo",
                         "finetune_agent_worker.py"),
            "--source-checkpoint", src,
            "--phase", str(phase),
            "--recipe", args.recipe,
            "--output-dir", outdir,
            "--hpc",
        ]
        if resume_ckpt:
            # Full resume mode: model + optimizer + global_step from
            # the prior run's latest.pt. Source checkpoint stays in
            # the command line for provenance only.
            cmd += ["--resume-from", resume_ckpt]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        try:
            proc = subprocess.Popen(
                cmd, cwd=_REPO_ROOT,
                stdout=out_log, stderr=err_log,
                env=env)
        except Exception as e:
            _write_block_status(args.block_dir, "failed",
                                error=f"spawn failed for phase {phase}: {e}",
                                traceback=traceback.format_exc())
            # Kill any spawned siblings before giving up.
            for prior_proc, _ in procs:
                try:
                    prior_proc.terminate()
                except Exception:
                    pass
            return 1
        procs.append((proc, phase))

    _write_block_status(args.block_dir, "running",
                        n_phases=len(procs),
                        phases=[p for _, p in procs])

    # Wait for all to complete. We don't bail on the first failure --
    # let siblings finish so their per-phase status sentinels land and
    # the master can act on each independently.
    rcodes = []
    for proc, phase in procs:
        rc = proc.wait()
        rcodes.append((phase, rc))

    all_ok = all(rc == 0 for _, rc in rcodes)
    _write_block_status(
        args.block_dir,
        "completed" if all_ok else "failed",
        per_phase_returncodes=dict(rcodes))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
