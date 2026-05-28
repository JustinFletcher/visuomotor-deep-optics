#!/usr/bin/env python
"""Autotrainer: iteratively close out the missing phases in a winners agent.

Loop:
  1. Read --winners-agent's manifest to find phases_missing.
  2. For each missing phase, in order:
     a. Launch a seed-sweep of N runs (one per node, fresh seeds) for
        this single phase. Each run is capped at --max-steps-per-run
        (default 50M) via --total-timesteps so it self-terminates if
        it never converges.
     b. Loop:
          * sleep --eval-interval-s
          * For each currently-active run on this phase, peek at its
            latest checkpoint and run a fast headless eval (no GIFs).
          * If any run's median Strehl >= --strehl-threshold, declare
            it the winner: copy its best.pt into the winners agent,
            update the manifest, scancel all other runs for this
            phase, and break the inner loop.
          * Otherwise, replace any finished/cancelled/timed-out runs
            with fresh seeds so the node budget stays full.
       Continue until a winner is found.
  3. Move on to the next missing phase. Once all are present, exit.

Designed to run on a login node as a long-running orchestrator.
Survives loss of connectivity to individual workers (we re-check
squeue + checkpoint state every tick, never trust in-memory state
that hasn't been confirmed from disk).

Usage::

    poetry run python train/ppo/autotrain_segment_dm_phases.py \\
        --winners-agent /p/work/fletch/agents/segment_dm_agent_v2_winners \\
        --source-agent agents/agent_20260419T211137Z_e3b7 \\
        --max-nodes 6 \\
        --max-steps-per-run 50000000 \\
        --eval-interval-s 600 \\
        --strehl-threshold 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml  # noqa: F401

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from train.ppo.launch_static_dark_hole import (
    HPC_CODE_DIR, HPC_WORKDIR, SLURM_ACCOUNT, SLURM_PARTITION, SLURM_TIME)
from train.ppo.finetune_sinfo import (
    query_sinfo_nodes, select_best_nodes)


_TRAIN_SCRIPT = "train/ppo/train_ppo_elf_segment_zernike_dm_transfer.py"
_MAX_SEED = 2**31 - 1
_NUM_PHASES = 15


# --------------------------------------------------------------------------
# Manifest helpers
# --------------------------------------------------------------------------

def _load_manifest(winners_dir: Path) -> dict:
    """Read the winners-agent manifest (created by pack_winner_phases)."""
    p = winners_dir / "manifest.json"
    if not p.is_file():
        raise FileNotFoundError(
            f"winners-agent manifest not found: {p}. Run "
            f"pack_winner_phases.py first to seed the dir.")
    with open(p) as f:
        return json.load(f)


def _save_manifest(winners_dir: Path, manifest: dict) -> None:
    with open(winners_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def _missing_phases(winners_dir: Path) -> list[int]:
    m = _load_manifest(winners_dir)
    return sorted(int(x) for x in m.get("phases_missing", []))


def _record_winner(winners_dir: Path, phase: int, src_ckpt: Path,
                   seed: int, source_run: Path) -> None:
    """Copy the winning checkpoint into the winners agent + update
    manifest with provenance. Idempotent: overwrites whatever was
    there for this phase."""
    ck_dir = winners_dir / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    dst = ck_dir / f"phase_{phase:02d}.pt"
    shutil.copyfile(src_ckpt, dst)

    manifest = _load_manifest(winners_dir)
    present = set(int(x) for x in manifest.get("phases_present", []))
    missing = set(int(x) for x in manifest.get("phases_missing", []))
    present.add(phase)
    missing.discard(phase)
    manifest["phases_present"] = sorted(present)
    manifest["phases_missing"] = sorted(missing)
    manifest.setdefault("pack_history", []).append({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "autotrainer",
        "phase_winners": [{
            "phase": phase,
            "seed": int(seed),
            "source_checkpoint": str(src_ckpt),
            "source_run": str(source_run),
            "dst": str(dst),
        }],
    })
    _save_manifest(winners_dir, manifest)


# --------------------------------------------------------------------------
# Sbatch submission (per-seed, one job per node)
# --------------------------------------------------------------------------

def _sweep_sbatch(node_name: str, phase: int, seed: int,
                  src_ckpt: str, out_dir: str, log_dir: str,
                  extra_args: str, slurm_time: str,
                  total_timesteps: int) -> str:
    job_name = f"auto-p{phase:02d}-s{seed % 100000:05d}"
    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --time={slurm_time}
        #SBATCH --account={SLURM_ACCOUNT}
        #SBATCH --partition={SLURM_PARTITION}
        #SBATCH --nodes=1
        #SBATCH --nodelist={node_name}
        #SBATCH --gres=gpu:1
        #SBATCH --output={log_dir}/sbatch-%j.out
        #SBATCH --error={log_dir}/sbatch-%j.err

        export PATH=$HOME/local/bin:$HOME/.local/bin:$PATH
        export LD_LIBRARY_PATH=$HOME/local/lib:$HOME/local/lib64:${{LD_LIBRARY_PATH:-}}

        mkdir -p {log_dir}
        cd {HPC_CODE_DIR}
        poetry run python -u {_TRAIN_SCRIPT} \\
            --hpc \\
            --phased-count {phase} \\
            --source-checkpoint {src_ckpt} \\
            --seed {seed} \\
            --total-timesteps {total_timesteps} \\
            --run-dir {out_dir} \\
            {extra_args}
    """)


def _submit_seed(node_name: str, phase: int, src_ckpt: str,
                 phase_root: Path, extra_args: str,
                 slurm_time: str, total_timesteps: int) -> Optional[dict]:
    """Submit one seed run pinned to one node. Returns a slot dict
    or None on submission failure."""
    seed = secrets.randbelow(_MAX_SEED)
    out_dir = phase_root / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    script = _sweep_sbatch(
        node_name=node_name, phase=phase, seed=seed,
        src_ckpt=str(src_ckpt), out_dir=str(out_dir),
        log_dir=str(log_dir), extra_args=extra_args,
        slurm_time=slurm_time, total_timesteps=total_timesteps)
    res = subprocess.run(
        ["sbatch", "--parsable"],
        input=script, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  [submit] sbatch FAILED on {node_name}: "
              f"{res.stderr.strip()}")
        return None
    job_id = res.stdout.strip()
    print(f"  [submit] phase {phase:02d} seed {seed} on {node_name} "
          f"-> job {job_id}")
    return {
        "seed": seed,
        "node": node_name,
        "out_dir": out_dir,
        "job_id": job_id,
        "proc": None,
        "runner": "sbatch",
        "submitted_at": time.time(),
    }


def _submit_seed_local(phase: int, src_ckpt: str,
                       phase_root: Path, extra_args: str,
                       total_timesteps: int,
                       gpu_idx: int = 0) -> Optional[dict]:
    """Spawn a worker as a local subprocess on the master's own node,
    pinned to a single local GPU via CUDA_VISIBLE_DEVICES. Uses the
    GPU the master's sbatch was already allocated -- otherwise idle
    -- so we don't lose a node slot to the orchestrator. Mirrors
    finetune_agent_master._start_local_phase."""
    seed = secrets.randbelow(_MAX_SEED)
    out_dir = phase_root / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_log = open(log_dir / "local.out", "w")
    err_log = open(log_dir / "local.err", "w")
    cmd = [
        sys.executable, "-u",
        os.path.join(_REPO_ROOT, _TRAIN_SCRIPT),
        "--hpc",
        "--phased-count", str(phase),
        "--source-checkpoint", str(src_ckpt),
        "--seed", str(seed),
        "--total-timesteps", str(total_timesteps),
        "--run-dir", str(out_dir),
    ] + shlex.split(extra_args)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    try:
        proc = subprocess.Popen(
            cmd, cwd=HPC_CODE_DIR,
            stdout=out_log, stderr=err_log, env=env)
    except Exception as e:
        print(f"  [submit-local] spawn FAILED: {e}")
        return None
    node_name = os.environ.get("HOSTNAME") or "localhost"
    print(f"  [submit-local] phase {phase:02d} seed {seed} on "
          f"{node_name} (GPU {gpu_idx}) -> pid {proc.pid}")
    return {
        "seed": seed,
        "node": node_name,
        "out_dir": out_dir,
        "job_id": None,
        "proc": proc,
        "runner": "local",
        "submitted_at": time.time(),
    }


def _pick_nodes(n: int, exclude: set[str]) -> list:
    """Sinfo for n idle GPU nodes, skipping anything in ``exclude``."""
    try:
        all_nodes = query_sinfo_nodes(partition=SLURM_PARTITION)
    except Exception as e:
        print(f"  [sinfo] query failed: {e}")
        return []
    cands = select_best_nodes(
        all_nodes, in_use=exclude, min_gpus=1,
        allow_mix=False, exclude_self=False)
    return cands[:n]


# --------------------------------------------------------------------------
# Job liveness check
# --------------------------------------------------------------------------

def _slot_is_alive(slot: dict) -> bool:
    """Liveness check that branches on runner type."""
    if slot.get("runner") == "local":
        proc = slot.get("proc")
        if proc is None:
            return False
        return proc.poll() is None
    job_id = slot.get("job_id")
    if not job_id:
        return False
    try:
        out = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%T"],
            capture_output=True, text=True, timeout=20)
    except Exception:
        return False
    if out.returncode != 0:
        return False
    st = out.stdout.strip()
    if not st:
        return False
    return st.upper() not in {
        "COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY"}


def _slot_cancel(slot: dict) -> None:
    """Cancellation that branches on runner type."""
    if slot.get("runner") == "local":
        proc = slot.get("proc")
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        return
    job_id = slot.get("job_id")
    if job_id:
        subprocess.run(["scancel", job_id], capture_output=True, text=True)


# --------------------------------------------------------------------------
# Checkpoint discovery + eval
# --------------------------------------------------------------------------

def _find_latest_checkpoint(run_dir: Path) -> Optional[Path]:
    """Find the most-recently-modified latest.pt under
    <run_dir>/ppo_optomech_*/checkpoints/."""
    cands = []
    for rd in run_dir.glob("ppo_optomech_*"):
        ck = rd / "checkpoints" / "latest.pt"
        if ck.is_file():
            cands.append(ck)
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _find_best_checkpoint(run_dir: Path) -> Optional[Path]:
    """Find the most-recently-modified best.pt under
    <run_dir>/ppo_optomech_*/checkpoints/."""
    cands = []
    for rd in run_dir.glob("ppo_optomech_*"):
        ck = rd / "checkpoints" / "best.pt"
        if ck.is_file():
            cands.append(ck)
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _eval_checkpoint(ckpt_path: Path, phase: int, args) -> Optional[float]:
    """Headless per-phase eval: median final Strehl over K episodes.

    Returns None on env build / load failure (transient -- caller
    treats as "no data yet" and tries again next tick).
    """
    import gymnasium as gym
    from train.ppo.train_ppo_optomech import register_optomech
    from train.ppo.rollout_elf_bootstrap_ptt import ROLLOUT_ENV_KWARGS
    from train.ppo.rollout import (
        load_agent, run_single_episode,
    )
    from train.ppo.policy_spec import (
        _build_bootstrap_action_mask,
    )
    from train.ppo.output_adapters import SegmentZernikePassthroughAdapter
    from train.ppo.ppo_models import PPOActorWrapper
    from train.ppo.agents import (
        CompositeAgent, NeverTrigger, Phase, SingleModelAgent,
    )

    try:
        env_kwargs = dict(ROLLOUT_ENV_KWARGS)
        env_kwargs.update({
            "command_secondaries": True,
            "command_dm": True,
            "dm_incremental_control": True,
            "atmosphere": {
                "r0_total_m_range": [args.r0_range[0], args.r0_range[1]],
                "L0_m": 25.0,
                "static": True},
            "bootstrap_phased_count": phase,
            "silence": True,
        })
        register_optomech("optomech-v4",
                          max_episode_steps=args.eval_steps)
        env = gym.make("optomech-v4", **env_kwargs)
        env.reset(seed=int(time.time()) % 2**31)

        # Load model at native action_dim 45 + n_zernike.
        native_dim = 45 + args.n_zernike
        model, ckpt_cfg, _orm = load_agent(
            str(ckpt_path), env, device="cpu",
            action_dim_override=native_dim)
        wrapper = PPOActorWrapper(model)
        single = SingleModelAgent(wrapper, "cpu")
        adapter = SegmentZernikePassthroughAdapter.from_env(
            env, n_zernike=args.n_zernike,
            n_seg_actions=45,
            dm_action_scale=args.dm_action_scale,
            device=torch.device("cpu"),
            silence=True)

        ck_env = dict((ckpt_cfg or {}).get("env_kwargs", {}) or {})
        ck_env["bootstrap_phased_count"] = phase
        cfg_for_mask = dict(ckpt_cfg or {})
        cfg_for_mask["env_kwargs"] = ck_env
        mask = _build_bootstrap_action_mask(
            cfg_for_mask, native_dim, adapter=adapter)

        ph = Phase(agent=single, trigger=NeverTrigger(),
                   name=f"phase-{phase:02d}",
                   action_mask=mask,
                   bootstrap_phased_count=phase,
                   output_adapter=adapter)
        composite = CompositeAgent([ph], device="cpu")

        base = env.unwrapped
        obs_ref_max = float(getattr(
            getattr(base, "optical_system", None),
            "_reference_fpi_max", 1.0))

        finals = []
        for i in range(args.eval_episodes):
            env.reset(seed=int(time.time()) % 2**31 + i)
            ep = run_single_episode(
                composite, env, seed=int(time.time()) % 2**31 + i,
                obs_ref_max=obs_ref_max, device="cpu")
            finals.append(float(ep["strehls"][-1]))
        env.close()
        return float(np.median(finals))
    except Exception as e:
        print(f"  [eval] error on {ckpt_path}: {e}")
        return None


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Autotrainer: close out missing phases in a winners "
                    "agent via parallel seed sweeps + headless eval.")
    p.add_argument("--winners-agent", required=True,
                   help="Writable agent dir produced by "
                        "pack_winner_phases.py. The autotrainer fills "
                        "in phases_missing one at a time.")
    p.add_argument("--source-agent", required=True,
                   help="Source bootstrap agent (read-only). Used for "
                        "per-phase --source-checkpoint init.")
    p.add_argument("--max-nodes", type=int, default=6,
                   help="Number of seeds run in parallel per phase. "
                        "Each gets its own node + 1 GPU.")
    p.add_argument("--max-steps-per-run", type=int, default=50_000_000,
                   help="Per-seed total_timesteps cap. Default 50M; "
                        "runs that hit this self-terminate and are "
                        "replaced with a fresh seed by the loop.")
    p.add_argument("--eval-interval-s", type=int, default=600,
                   help="Seconds between eval ticks (default 600 = "
                        "10 min). Lower = faster reaction to winners "
                        "but more eval overhead.")
    p.add_argument("--eval-episodes", type=int, default=3,
                   help="Episodes per headless eval (default 3).")
    p.add_argument("--eval-steps", type=int, default=50,
                   help="Steps per eval episode (default 50).")
    p.add_argument("--strehl-threshold", type=float, default=0.5,
                   help="Median final Strehl >= this declares a "
                        "winner.")
    p.add_argument("--phases", type=str, default=None, metavar="LIST",
                   help="Override the winners-agent's phases_missing "
                        "with an explicit comma-separated list.")
    p.add_argument("--n-zernike", type=int, default=32)
    p.add_argument("--dm-action-scale", type=float, default=0.01)
    p.add_argument("--r0-range", type=float, nargs=2,
                   default=[0.10, 0.25],
                   metavar=("LOW", "HIGH"))
    p.add_argument("--slurm-time", type=str, default=SLURM_TIME)
    p.add_argument("--output-root", type=str, default=None,
                   help="Per-run sweep output root. Default: "
                        "$HPC_WORKDIR/segment_dm_finetuning/autotrain_"
                        "<src>_<ts>/")
    args = p.parse_args()

    winners_dir = Path(args.winners_agent).resolve()
    source_dir = Path(args.source_agent).resolve()
    if not winners_dir.is_dir():
        print(f"ERROR: --winners-agent not a dir: {winners_dir}")
        sys.exit(1)
    if not source_dir.is_dir():
        print(f"ERROR: --source-agent not a dir: {source_dir}")
        sys.exit(1)

    # Resolve phase list.
    if args.phases:
        todo = [int(x) for x in args.phases.split(",") if x.strip()]
    else:
        todo = _missing_phases(winners_dir)
    print(f"=== Autotrainer ===")
    print(f"Winners dir:        {winners_dir}")
    print(f"Source dir:         {source_dir}")
    print(f"Phases to train:    {todo}")
    print(f"Max nodes (seeds):  {args.max_nodes}")
    print(f"Max steps per run:  {args.max_steps_per_run:,}")
    print(f"Eval interval:      {args.eval_interval_s}s")
    print(f"Strehl threshold:   {args.strehl_threshold}")
    print()

    # Output root for all sweeps.
    if args.output_root:
        sweep_root = Path(args.output_root)
    else:
        ts = int(time.time())
        sweep_root = Path(HPC_WORKDIR) / "segment_dm_finetuning" / (
            f"autotrain_{source_dir.name}_{ts}")
    sweep_root.mkdir(parents=True, exist_ok=True)
    print(f"Sweep root:         {sweep_root}\n")

    extra_args = (
        f"--n-zernike {args.n_zernike} "
        f"--dm-action-scale {args.dm_action_scale} "
        f"--r0-range {args.r0_range[0]} {args.r0_range[1]}")

    # --------------------------------------------------------------
    # Per-phase loop
    # --------------------------------------------------------------
    for phase in todo:
        src_ckpt = source_dir / "checkpoints" / f"phase_{phase:02d}.pt"
        if not src_ckpt.is_file():
            print(f"\nSKIP phase {phase}: source checkpoint missing "
                  f"({src_ckpt})")
            continue

        phase_root = sweep_root / f"phase_{phase:02d}"
        phase_root.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 60}")
        print(f"  PHASE {phase:02d}")
        print(f"{'=' * 60}")

        active: dict[int, dict] = {}    # seed -> slot
        nodes_in_use = set()

        # Initial fill. The master node always counts as one of the
        # max_nodes slots: it runs one worker as a local subprocess
        # on its already-allocated GPU (CUDA_VISIBLE_DEVICES=0) so we
        # don't lose a node slot to the orchestrator. The remaining
        # max_nodes-1 slots go to sbatch workers on other nodes.
        master_node = (os.environ.get("SLURMD_NODENAME")
                       or os.environ.get("HOSTNAME")
                       or "localhost")
        # Reserve master's node so sinfo doesn't try to submit a
        # second sbatch onto it (which the cluster would reject).
        nodes_in_use.add(master_node)

        # Local worker on the master node.
        if args.max_nodes >= 1:
            slot = _submit_seed_local(
                phase, str(src_ckpt), phase_root, extra_args,
                args.max_steps_per_run, gpu_idx=0)
            if slot:
                slot["node"] = master_node
                active[slot["seed"]] = slot

        # Remote workers on the other max_nodes-1 nodes.
        n_remote = max(0, args.max_nodes - 1)
        if n_remote > 0:
            nodes = _pick_nodes(n_remote, exclude=nodes_in_use)
            if len(nodes) < n_remote:
                print(f"  [sinfo] WARNING: only {len(nodes)}/{n_remote} "
                      f"idle remote nodes available "
                      f"(+1 local on {master_node}).")
            for n in nodes:
                slot = _submit_seed(
                    n.name, phase, str(src_ckpt), phase_root,
                    extra_args, args.slurm_time,
                    args.max_steps_per_run)
                if slot:
                    active[slot["seed"]] = slot
                    nodes_in_use.add(n.name)

        # Eval loop.
        winner = None
        while not winner:
            try:
                time.sleep(args.eval_interval_s)
            except KeyboardInterrupt:
                print(f"\nInterrupted -- cancelling phase {phase} runs.")
                for sl in active.values():
                    _slot_cancel(sl)
                sys.exit(130)

            # Sync liveness: branch on runner type (local subprocess
            # vs sbatch).
            dead = [s for s, sl in active.items()
                    if not _slot_is_alive(sl)]
            for s in dead:
                sl = active.pop(s)
                # Only free a remote node slot; the master's local
                # subprocess can be re-spawned on the same node.
                if sl["runner"] != "local":
                    nodes_in_use.discard(sl["node"])
                ident = (f"pid {sl['proc'].pid}" if sl.get("proc")
                         else f"job {sl.get('job_id')}")
                print(f"  [reap] phase {phase} seed {s} on "
                      f"{sl['node']} ({ident}) gone")

            # Eval each live slot.
            print(f"\n  [tick] phase {phase}: {len(active)} alive")
            for seed, sl in list(active.items()):
                ck = (_find_best_checkpoint(sl["out_dir"])
                      or _find_latest_checkpoint(sl["out_dir"]))
                if ck is None:
                    print(f"    seed {seed}: no checkpoint yet")
                    continue
                m_strehl = _eval_checkpoint(ck, phase, args)
                if m_strehl is None:
                    continue
                tag = "WIN" if m_strehl >= args.strehl_threshold else "..."
                print(f"    seed {seed} ({sl['node']}): "
                      f"median final S = {m_strehl:.4f}  [{tag}]")
                if m_strehl >= args.strehl_threshold:
                    winner = (seed, sl, ck, m_strehl)
                    break

            if winner:
                seed, sl, ck, m = winner
                print(f"\n  *** WINNER: phase {phase} seed {seed} "
                      f"strehl={m:.4f} ***")
                best = _find_best_checkpoint(sl["out_dir"]) or ck
                _record_winner(winners_dir, phase, best, seed,
                               sl["out_dir"])
                # Cancel all other runs for this phase.
                for sd, other in active.items():
                    if sd != seed:
                        _slot_cancel(other)
                        ident = (f"pid {other['proc'].pid}"
                                 if other.get("proc")
                                 else f"job {other.get('job_id')}")
                        print(f"    cancelled seed {sd} ({ident})")
                break

            # Replace dead slots with fresh seeds. The local slot
            # gets refilled in-place on the master node; remote
            # slots get fresh sinfo picks.
            has_local = any(sl["runner"] == "local"
                            for sl in active.values())
            if not has_local:
                slot = _submit_seed_local(
                    phase, str(src_ckpt), phase_root, extra_args,
                    args.max_steps_per_run, gpu_idx=0)
                if slot:
                    slot["node"] = master_node
                    active[slot["seed"]] = slot

            shortfall = (args.max_nodes
                         - sum(1 for _ in active.values()))
            if shortfall > 0:
                fresh = _pick_nodes(shortfall, exclude=nodes_in_use)
                for n in fresh:
                    slot = _submit_seed(
                        n.name, phase, str(src_ckpt), phase_root,
                        extra_args, args.slurm_time,
                        args.max_steps_per_run)
                    if slot:
                        active[slot["seed"]] = slot
                        nodes_in_use.add(n.name)

    print("\n=== Autotrainer done. ===")
    final = _load_manifest(winners_dir)
    print(f"Phases present: {final.get('phases_present')}")
    print(f"Phases missing: {final.get('phases_missing')}")


if __name__ == "__main__":
    main()
