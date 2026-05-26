"""Master orchestrator for fine-tuning every phase of a composite agent.

Designed to run on a compute node (submitted by
``launch_finetune_agent.py``). The master:

1. Reads a source agent (``agents/<name>/{manifest.json, composed.yaml,
   checkpoints/phase_NN.pt}``) -- READ-ONLY -- and a recipe YAML.
2. Builds an output dir mirror under ``agent_finetuning/<source>_
   <recipe>_<ts>/`` with per-phase ``phases/phase_NN/`` subdirs.
3. Maintains a job pool of (1 local subprocess + N sbatch workers).
   Default N = 5, so 6 concurrent fine-tune jobs at peak. Defaults
   are configurable.
4. As phases complete (status.json sentinel + squeue check), it
   schedules the next pending phase into the freed slot. Continues
   until all 15 phases are done.
5. Packs the final agent: pulls each phase's best.pt into
   ``agent/checkpoints/phase_NN.pt``, writes a new ``composed.yaml``
   (mirrors source spec but pointing at the new checkpoints) and
   ``manifest.json`` (source + recipe + per-phase provenance).

Source agent is NEVER written to.

Pacing note: today the scheduler is plain FIFO -- phases get a single
training budget (per the recipe's ``ppo_overrides.total_timesteps``)
and the master fills any free slot from the pending queue. With 15
phases and 6 slots the late phases land later by an order-of-rounds
factor. A future enhancement could chunk each phase into K rounds of
``total_timesteps / K`` and round-robin them so all phases progress
together; the master / worker contract via ``init_from`` already
supports this.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# --------------------------------------------------------------------------
# SLURM constants (mirrored from launch_static_dark_hole.py so the
# master doesn't have to import a launcher script).
# --------------------------------------------------------------------------
SLURM_ACCOUNT = "MHPCC38870258"
SLURM_PARTITION = "standard"
SLURM_GRES = "gpu"
SLURM_TIME_DEFAULT = "72:00:00"
# Per cluster policy ("jobs running on filesystems other than /p/work
# / $WORKDIR will be terminated"), the working tree used by SLURM jobs
# must live on the work filesystem. Resolved at launcher-import time
# from $WORKDIR with a conservative fallback. Override with
# VMDO_HPC_WORKDIR if your clone lives elsewhere on the work fs.
HPC_WORKDIR = os.environ.get(
    "VMDO_HPC_WORKDIR",
    os.path.join(os.environ.get("WORKDIR", "/p/work/fletch"),
                 "visuomotor-deep-optics"))


_PHASE_LABEL_RE = re.compile(r"phase[-_]?(\d{2})")


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------

@dataclass
class PhaseJob:
    phase: int
    source_ckpt: str
    output_dir: str
    status: str = "pending"           # pending, running, completed, failed
    runner: Optional[str] = None      # "local" | "slurm"
    slurm_job_id: Optional[str] = None
    slurm_node: Optional[str] = None  # node-block target, if known
    block_dir: Optional[str] = None   # per-block log + manifest dir
    proc: Optional[subprocess.Popen] = None
    local_gpu_idx: Optional[int] = None   # which CUDA_VISIBLE_DEVICES slot
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    best_ckpt: Optional[str] = None
    attempts: int = 0
    last_error: Optional[str] = None
    # Full-resume checkpoint: when set, the worker uses --resume-from
    # (model + optimizer + global_step) instead of --init-from (weights
    # only). Populated by --resume mode after scanning the phase dir
    # for the most recent ppo_optomech_*/checkpoints/latest.pt.
    resume_ckpt: Optional[str] = None


@dataclass
class MasterState:
    output_root: str
    source_agent: dict                # parsed manifest.json
    source_agent_dir: str
    recipe_path: str
    recipe: dict
    phases: list[PhaseJob] = field(default_factory=list)
    # Total nodes the run is allowed to use, INCLUDING the master's
    # own node. Default 4 = 1 master + 3 sbatch node-blocks. With
    # 4-GPU nodes, that's 16 concurrent fine-tunes -- enough to run
    # every phase of a 15-phase agent in parallel and still leave
    # nodes for other work.
    max_nodes: int = 4
    gpus_per_node: int = 1            # local GPU slots on the master's node
    poll_interval_s: float = 30.0
    max_retries: int = 1
    slurm_time: str = SLURM_TIME_DEFAULT
    # Resume mode: when True, _build_phase_jobs scans each phase dir
    # for the most recent ppo_optomech_*/checkpoints/latest.pt and
    # populates phase.resume_ckpt. The worker then runs in full-
    # resume mode (--resume-from) instead of init-from-source. Stale
    # status.json files from the prior run are cleared so the master
    # doesn't immediately mark phases "completed" before the new
    # worker has had a chance to overwrite the sentinel.
    resume: bool = False


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _log(state: MasterState, msg: str) -> None:
    """Append to log.txt and stdout."""
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_path = os.path.join(state.output_root, "log.txt")
    with open(log_path, "a") as f:
        f.write(line + "\n")


def _read_source_agent(agent_dir: str) -> dict:
    """Parse the source agent's manifest. Validates the bundle is read-only
    relative to where we're going (output dir cannot equal agent dir)."""
    manifest_path = os.path.join(agent_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Source agent manifest not found: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    # Sanity: must have phases list with checkpoints.
    if "phases" not in manifest:
        raise ValueError(f"Source manifest missing 'phases': {manifest_path}")
    return manifest


def _find_latest_phase_checkpoint(output_dir: str) -> Optional[str]:
    """Scan <output_dir>/ppo_optomech_*/checkpoints/ for the most
    recently modified ``latest.pt``. Returns its absolute path or None
    if no prior run dir exists. Used by --resume mode to pick the
    checkpoint to hand to the worker."""
    candidates = []
    for run_dir in Path(output_dir).glob("ppo_optomech_*"):
        ck = run_dir / "checkpoints" / "latest.pt"
        if ck.is_file():
            candidates.append(ck)
    if not candidates:
        return None
    # Pick the most recently modified -- this is the "freshest" prior
    # run, which is what we want to continue from. Comparing by mtime
    # is robust to wall-clock skew in the seed/ts component of the
    # run dir name.
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0].resolve())


def _clear_phase_status(output_dir: str) -> None:
    """Rename any pre-existing status.json out of the way before a
    resume so the master doesn't read a stale 'completed' sentinel
    before the new worker has had a chance to overwrite it."""
    path = os.path.join(output_dir, "status.json")
    if os.path.isfile(path):
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        os.rename(path, os.path.join(output_dir,
                                     f"status.prev_{stamp}.json"))


def _build_phase_jobs(state: MasterState) -> list[PhaseJob]:
    """One PhaseJob per source-agent phase. Source checkpoint comes
    from the manifest's bundle_path (resolved relative to the agent
    dir). Output dir lives under <output_root>/phases/phase_NN/.

    In --resume mode, additionally:
      - Scan the phase dir for the most recent latest.pt and attach
        it as resume_ckpt (worker will full-resume from it).
      - Clear stale status.json so the master doesn't short-circuit.
      - Phases without any prior checkpoint fall back to init-from-
        source as a starting case (e.g. the resume was triggered
        before phase N ever started)."""
    out = []
    for ph in state.source_agent["phases"]:
        idx = int(ph["phase"])
        src = os.path.join(state.source_agent_dir, ph["bundle_path"])
        if not os.path.isfile(src):
            raise FileNotFoundError(
                f"Source checkpoint missing for phase {idx}: {src}")
        out_dir = os.path.join(state.output_root,
                               "phases", f"phase_{idx:02d}")
        os.makedirs(out_dir, exist_ok=True)
        job = PhaseJob(phase=idx, source_ckpt=src, output_dir=out_dir)
        if state.resume:
            _clear_phase_status(out_dir)
            resume_ckpt = _find_latest_phase_checkpoint(out_dir)
            if resume_ckpt:
                job.resume_ckpt = resume_ckpt
        out.append(job)
    return out


# --------------------------------------------------------------------------
# Job scheduling
# --------------------------------------------------------------------------

def _read_status(output_dir: str) -> Optional[dict]:
    path = os.path.join(output_dir, "status.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _is_slurm_job_alive(job_id: str) -> bool:
    """True if the SLURM job is still in the queue (PD/R/CG/...)."""
    try:
        out = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%T"],
            capture_output=True, text=True, timeout=20)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    if out.returncode != 0:
        return False
    state_str = out.stdout.strip()
    if not state_str:
        return False
    # Anything other than COMPLETED / CANCELLED / FAILED / TIMEOUT means
    # the job still exists in SLURM's view.
    return state_str.upper() not in {"COMPLETED", "CANCELLED",
                                     "FAILED", "TIMEOUT", "OUT_OF_MEMORY"}


def _slurm_node_block_script(state: MasterState, jobs: list[PhaseJob],
                             run_id: str, node_name: str,
                             gpu_count: int, block_dir: str,
                             manifest_path: str) -> str:
    """Build an sbatch script for one *node-block* worker: a single
    SLURM job pinned to one node, requesting --gres=gpu:<gpu_count>,
    that internally fans out one finetune_agent_worker.py subprocess
    per assigned phase (each on its own local GPU).
    """
    phases_label = "-".join(f"p{j.phase:02d}" for j in jobs)
    name = f"ft-{run_id[-8:]}-{node_name[-6:]}-{phases_label}"
    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={name}
        #SBATCH --time={state.slurm_time}
        #SBATCH --account={SLURM_ACCOUNT}
        #SBATCH --partition={SLURM_PARTITION}
        #SBATCH --nodes=1
        #SBATCH --nodelist={node_name}
        #SBATCH --gres=gpu:{gpu_count}
        #SBATCH --output={block_dir}/slurm-%j.out
        #SBATCH --error={block_dir}/slurm-%j.err

        export PATH=$HOME/local/bin:$HOME/.local/bin:$PATH
        export LD_LIBRARY_PATH=$HOME/local/lib:$HOME/local/lib64:${{LD_LIBRARY_PATH:-}}

        cd {HPC_WORKDIR}
        poetry run python -u train/ppo/finetune_node_block_worker.py \\
            --manifest {manifest_path} \\
            --recipe {state.recipe_path} \\
            --block-dir {block_dir}
    """)


def _submit_slurm_node_block(state: MasterState, jobs: list[PhaseJob],
                             run_id: str, node_name: str,
                             gpu_count: int) -> None:
    """Submit one node-block sbatch job covering ``jobs``. Writes the
    per-phase manifest and assigns the same slurm_job_id to every
    PhaseJob in the block so we can track them as a unit."""
    if not jobs:
        return
    # Block dir under blocks/<run_id>/<idx>; idx based on submission
    # order. Cheap to construct; we just need it unique per block.
    blocks_root = os.path.join(state.output_root, "blocks")
    os.makedirs(blocks_root, exist_ok=True)
    existing = sorted(p.name for p in Path(blocks_root).iterdir()
                      if p.is_dir())
    next_idx = len(existing)
    block_label = (
        f"block_{next_idx:04d}_{node_name}_"
        + "-".join(f"p{j.phase:02d}" for j in jobs))
    block_dir = os.path.join(blocks_root, block_label)
    os.makedirs(block_dir, exist_ok=True)

    manifest = [
        {"phase": j.phase,
         "source_checkpoint": j.source_ckpt,
         "output_dir": j.output_dir,
         # Optional: when set, the inner per-phase worker runs in
         # --resume-from mode (full model + optimizer + step resume)
         # instead of --init-from. None means "fresh init from source".
         "resume_checkpoint": j.resume_ckpt}
        for j in jobs]
    manifest_path = os.path.join(block_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    script = _slurm_node_block_script(
        state, jobs, run_id,
        node_name=node_name, gpu_count=gpu_count,
        block_dir=block_dir, manifest_path=manifest_path)
    result = subprocess.run(
        ["sbatch", "--parsable"],
        input=script, capture_output=True, text=True)
    if result.returncode != 0:
        err = result.stderr.strip()
        for j in jobs:
            j.status = "failed"
            j.last_error = f"sbatch failed (block): {err}"
        _log(state, f"  block {block_label}: sbatch FAILED -- {err}")
        return
    job_id = result.stdout.strip()
    for j in jobs:
        j.runner = "slurm"
        j.status = "running"
        j.slurm_job_id = job_id
        j.slurm_node = node_name
        j.block_dir = block_dir
        j.started_at = time.time()
        j.attempts += 1
    _log(state, f"  block {block_label}: sbatch job {job_id} submitted "
                f"on {node_name} (--gres=gpu:{gpu_count})")


def _submit_slurm_phase(state: MasterState, job: PhaseJob,
                        run_id: str) -> None:
    script = _slurm_script_for_phase(state, job, run_id)
    result = subprocess.run(
        ["sbatch", "--parsable"],
        input=script, capture_output=True, text=True)
    if result.returncode != 0:
        job.status = "failed"
        job.last_error = f"sbatch failed: {result.stderr.strip()}"
        _log(state, f"  phase {job.phase:02d}: sbatch FAILED -- {job.last_error}")
        return
    job.slurm_job_id = result.stdout.strip()
    job.runner = "slurm"
    job.status = "running"
    job.started_at = time.time()
    job.attempts += 1
    _log(state, f"  phase {job.phase:02d}: sbatch job {job.slurm_job_id} submitted")


def _start_local_phase(state: MasterState, job: PhaseJob,
                       gpu_idx: int) -> None:
    """Spawn the worker as a subprocess on the master's own node,
    pinned to a specific local GPU via CUDA_VISIBLE_DEVICES.

    stdout/stderr go to local.out / local.err in the phase dir;
    when running multiple local workers (one per GPU) the per-phase
    log files keep each worker's output separated.
    """
    out_log = open(os.path.join(job.output_dir, "local.out"), "w")
    err_log = open(os.path.join(job.output_dir, "local.err"), "w")
    cmd = [
        sys.executable, "-u",
        os.path.join(_REPO_ROOT, "train", "ppo",
                     "finetune_agent_worker.py"),
        "--source-checkpoint", job.source_ckpt,
        "--phase", str(job.phase),
        "--recipe", state.recipe_path,
        "--output-dir", job.output_dir,
        "--hpc",
    ]
    if job.resume_ckpt:
        # Full resume: model + optimizer + global_step from the prior
        # run's latest.pt. Source checkpoint stays in the cmd line for
        # status.json provenance only.
        cmd += ["--resume-from", job.resume_ckpt]
    # Pin the subprocess to one GPU so multiple local workers on the
    # same node don't fight for the same device. CUDA_VISIBLE_DEVICES
    # is the standard mechanism; the worker process sees only its
    # assigned GPU and indexes it as 0 internally.
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    proc = subprocess.Popen(
        cmd, cwd=_REPO_ROOT,
        stdout=out_log, stderr=err_log,
        env=env)
    job.proc = proc
    job.runner = "local"
    job.local_gpu_idx = gpu_idx
    job.status = "running"
    job.started_at = time.time()
    job.attempts += 1
    _log(state, f"  phase {job.phase:02d}: local subprocess pid={proc.pid} "
                f"on GPU {gpu_idx} started")


def _check_running(state: MasterState) -> None:
    """For each running job, see if it has completed (by status file +
    process state) and update accordingly."""
    for job in state.phases:
        if job.status != "running":
            continue
        st = _read_status(job.output_dir)

        if job.runner == "local":
            # Local subprocess: completion = proc.returncode is set.
            rc = job.proc.poll() if job.proc else None
            if rc is None:
                continue  # still running
            # Process exited.
            if st and st.get("status") == "completed":
                job.status = "completed"
                job.best_ckpt = st.get("best_checkpoint") or None
                job.finished_at = time.time()
                _log(state, f"  phase {job.phase:02d}: local DONE "
                     f"(rc={rc}, best={job.best_ckpt})")
            else:
                # Process gone but status not "completed" => failure
                err = (st.get("error") if st else
                       f"exit code {rc} without status sentinel")
                job.last_error = err
                job.status = "failed"
                job.finished_at = time.time()
                _log(state, f"  phase {job.phase:02d}: local FAILED -- {err}")

        elif job.runner == "slurm":
            # Prefer the sentinel: workers write it on success/failure.
            if st and st.get("status") in ("completed", "failed"):
                if st["status"] == "completed":
                    job.status = "completed"
                    job.best_ckpt = st.get("best_checkpoint") or None
                    job.finished_at = time.time()
                    _log(state, f"  phase {job.phase:02d}: slurm "
                         f"{job.slurm_job_id} DONE (best={job.best_ckpt})")
                else:
                    job.status = "failed"
                    job.last_error = st.get("error", "worker reported failed")
                    job.finished_at = time.time()
                    _log(state, f"  phase {job.phase:02d}: slurm "
                         f"{job.slurm_job_id} FAILED -- {job.last_error}")
            else:
                # No sentinel yet. If squeue says the job is gone, the
                # worker died before writing -- treat as failed.
                if not _is_slurm_job_alive(job.slurm_job_id):
                    job.status = "failed"
                    job.last_error = (
                        f"slurm job {job.slurm_job_id} exited without "
                        f"writing status.json (cancelled/OOM/timeout?)")
                    job.finished_at = time.time()
                    _log(state, f"  phase {job.phase:02d}: slurm "
                         f"{job.slurm_job_id} GONE without sentinel")


def _schedule_pending(state: MasterState, run_id: str) -> None:
    """Fill open slots:
      - up to gpus_per_node local-subprocess slots on the master's node,
        each pinned to one GPU via CUDA_VISIBLE_DEVICES;
      - up to max_concurrent_slurm sbatch *node-block* jobs, each
        targeting one explicit node via --nodelist and packing as many
        pending phases as that node has GPUs.

    Slots are picked greedily in pending-job order. Local slots are
    preferred over sbatch slots because the local node already has the
    GPUs allocated (no queue wait).

    The sbatch path picks "best" nodes via finetune_sinfo: idle, most
    GPUs first. Nodes already in use by an in-flight block are
    skipped so we don't pile two blocks onto the same node. If sinfo
    yields nothing usable we fall back to submitting a 1-GPU
    single-phase sbatch (lets SLURM choose) so the run still makes
    progress, just without node-pinning.
    """
    from train.ppo.finetune_sinfo import (
        query_sinfo_nodes, select_best_nodes)

    # --- Local slots ---
    local_busy = [False] * max(state.gpus_per_node, 1)
    for j in state.phases:
        if (j.status == "running" and j.runner == "local"
                and j.local_gpu_idx is not None
                and 0 <= j.local_gpu_idx < len(local_busy)):
            local_busy[j.local_gpu_idx] = True

    # Pull pending in phase order so the natural sweep through phases
    # 0..14 happens in order, regardless of how we pack them.
    pending = [j for j in state.phases if j.status == "pending"]

    # Fill local slots first.
    while pending:
        free_local = next(
            (i for i, busy in enumerate(local_busy) if not busy), None)
        if free_local is None:
            break
        job = pending.pop(0)
        _start_local_phase(state, job, gpu_idx=free_local)
        local_busy[free_local] = True

    # --- SLURM node-block slots ---
    # Budget is on TOTAL NODES, not jobs. The master's own node counts
    # for 1; each in-flight node-block counts for 1 (it pins exactly
    # one node via --nodelist). Unpinned fallback jobs (no slurm_node)
    # also count for 1 each conservatively.
    nodes_in_use = {j.slurm_node for j in state.phases
                    if j.status == "running" and j.runner == "slurm"
                    and j.slurm_node}
    unpinned_jobs = len({j.slurm_job_id for j in state.phases
                         if j.status == "running" and j.runner == "slurm"
                         and not j.slurm_node and j.slurm_job_id})
    # +1 for the master's own node.
    nodes_used_count = 1 + len(nodes_in_use) + unpinned_jobs
    nodes_budget_remaining = max(0, state.max_nodes - nodes_used_count)

    if not pending or nodes_budget_remaining <= 0:
        return

    # Query sinfo ONCE per scheduling tick.
    try:
        all_nodes = query_sinfo_nodes(partition=SLURM_PARTITION)
    except Exception as e:
        _log(state, f"  sinfo query failed: {e}")
        all_nodes = []

    candidates = select_best_nodes(
        all_nodes,
        in_use=nodes_in_use,
        min_gpus=1,
        allow_mix=False,
        exclude_self=True,
    )

    # Walk best-to-worst nodes, packing up to min(node_gpu_count,
    # pending) phases per node. Stop when the node budget is full.
    for node in candidates:
        if not pending or nodes_budget_remaining <= 0:
            break
        take = min(node.gpu_count, len(pending))
        if take <= 0:
            continue
        batch = pending[:take]
        pending = pending[take:]
        _submit_slurm_node_block(
            state, batch, run_id,
            node_name=node.name, gpu_count=take)
        nodes_budget_remaining -= 1

    # Fallback: if sinfo gave nothing usable and there's pending work
    # plus node budget, submit single-phase 1-GPU sbatch jobs and let
    # SLURM pick the node. Each counts for 1 node against the budget.
    if pending and nodes_budget_remaining > 0 and not candidates:
        _log(state, f"  no sinfo candidates; falling back to "
                    f"unpinned single-phase sbatch")
        for job in list(pending):
            if nodes_budget_remaining <= 0:
                break
            _submit_slurm_phase(state, job, run_id)
            nodes_budget_remaining -= 1
            pending.remove(job)


def _retry_failed_if_eligible(state: MasterState) -> None:
    """Reset failed jobs back to pending if they have retries left.
    Conservative: only retry once by default."""
    for job in state.phases:
        if job.status != "failed":
            continue
        if job.attempts <= state.max_retries:
            _log(state, f"  phase {job.phase:02d}: retrying "
                 f"(attempt {job.attempts + 1}/{state.max_retries + 1})")
            job.status = "pending"
            job.runner = None
            job.slurm_job_id = None
            job.slurm_node = None
            job.block_dir = None
            job.local_gpu_idx = None
            job.proc = None
            job.last_error = None


def _status_summary(state: MasterState) -> str:
    counts = {}
    for j in state.phases:
        counts[j.status] = counts.get(j.status, 0) + 1
    return "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))


# --------------------------------------------------------------------------
# Packing the final agent
# --------------------------------------------------------------------------

def _pack_agent(state: MasterState) -> str:
    """Bundle the completed phases into a final agent dir mirroring the
    source. Returns the agent dir path."""
    agent_dir = os.path.join(state.output_root, "agent")
    ck_dir = os.path.join(agent_dir, "checkpoints")
    os.makedirs(ck_dir, exist_ok=True)

    new_manifest = dict(state.source_agent)  # shallow copy preserves order
    new_manifest = {
        "agent_name": os.path.basename(state.output_root),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_agent": state.source_agent.get("agent_name"),
        "source_agent_dir": state.source_agent_dir,
        "recipe": state.recipe.get("name", "unnamed"),
        "recipe_path": state.recipe_path,
        "recipe_description": state.recipe.get("description"),
        "recipe_env_kwarg_overrides": state.recipe.get("env_kwarg_overrides", {}),
        "recipe_ppo_overrides": state.recipe.get("ppo_overrides", {}),
        "trigger": state.source_agent.get("trigger"),
        "num_phases": state.source_agent.get("num_phases"),
        "phases": [],
    }

    for src_phase, job in zip(state.source_agent["phases"], state.phases):
        idx = int(src_phase["phase"])
        dst = os.path.join(ck_dir, f"phase_{idx:02d}.pt")
        if not job.best_ckpt or not os.path.isfile(job.best_ckpt):
            raise RuntimeError(
                f"Cannot pack phase {idx}: best_ckpt missing "
                f"({job.best_ckpt!r})")
        shutil.copyfile(job.best_ckpt, dst)
        new_manifest["phases"].append({
            "phase": idx,
            "name": src_phase["name"],
            "source_phase_checkpoint": job.source_ckpt,
            "finetune_run_dir": job.output_dir,
            "finetune_best_checkpoint": job.best_ckpt,
            "bundle_path": f"checkpoints/phase_{idx:02d}.pt",
            "size_bytes": os.path.getsize(dst),
        })

    # Mirror the source composed.yaml structure but point at our
    # bundled paths. Keep the same per-phase step trigger.
    src_spec_path = os.path.join(state.source_agent_dir, "composed.yaml")
    if os.path.isfile(src_spec_path):
        with open(src_spec_path) as f:
            spec = yaml.safe_load(f)
        # Rewrite checkpoint paths to be relative to the new agent dir.
        if spec.get("type") == "composite":
            for i, ph in enumerate(spec["phases"]):
                ph["checkpoint"] = f"checkpoints/phase_{i:02d}.pt"
    else:
        # Fall back to building from scratch.
        spec = {
            "type": "composite",
            "phases": [
                {
                    "name": p["name"],
                    "checkpoint": p["bundle_path"],
                    "until": state.source_agent.get("trigger") or {"step": 16},
                } for p in new_manifest["phases"]],
        }
    with open(os.path.join(agent_dir, "composed.yaml"), "w") as f:
        yaml.dump(spec, f, default_flow_style=False, sort_keys=False)

    with open(os.path.join(agent_dir, "manifest.json"), "w") as f:
        json.dump(new_manifest, f, indent=2)

    # README pointing back to where this came from.
    readme = textwrap.dedent(f"""\
        # Fine-tuned agent: {new_manifest['agent_name']}

        Built by `train/ppo/finetune_agent_master.py` from:
          source agent: {state.source_agent_dir}
          recipe:       {state.recipe_path}
        on {new_manifest['created_utc']}.

        Per-phase fine-tune run dirs live under the parent
          {state.output_root}/phases/

        The source agent was not modified. To roll this agent out:
          poetry run python train/ppo/rollout_elf_bootstrap_ptt.py \\
              --policy-spec {agent_dir}/composed.yaml
        """)
    with open(os.path.join(agent_dir, "README.md"), "w") as f:
        f.write(readme)

    return agent_dir


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def run_master(args) -> int:
    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    source_agent = _read_source_agent(args.source_agent)
    with open(args.recipe) as f:
        recipe = yaml.safe_load(f)

    # Resolve local GPU count: explicit --gpus-per-node wins, otherwise
    # auto-detect on the compute node we're running on.
    gpus_per_node = args.gpus_per_node
    if gpus_per_node is None or gpus_per_node < 0:
        try:
            import torch
            gpus_per_node = max(1, torch.cuda.device_count())
        except Exception:
            gpus_per_node = 1

    state = MasterState(
        output_root=output_root,
        source_agent=source_agent,
        source_agent_dir=os.path.abspath(args.source_agent),
        recipe_path=os.path.abspath(args.recipe),
        recipe=recipe,
        max_nodes=args.max_nodes,
        gpus_per_node=gpus_per_node,
        poll_interval_s=args.poll_interval_s,
        max_retries=args.max_retries,
        slurm_time=args.slurm_time,
        resume=bool(args.resume),
    )

    # Save the resolved config alongside the runs.
    cfg_path = os.path.join(output_root, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump({
            "source_agent_dir": state.source_agent_dir,
            "source_agent_name": source_agent.get("agent_name"),
            "recipe_path": state.recipe_path,
            "recipe_name": recipe.get("name"),
            "max_nodes": state.max_nodes,
            "gpus_per_node": state.gpus_per_node,
            "max_retries": state.max_retries,
            "slurm_time": state.slurm_time,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                         time.gmtime()),
        }, f, default_flow_style=False, sort_keys=False)

    _log(state, f"Master run dir:     {output_root}")
    _log(state, f"Source agent:       {state.source_agent_dir}")
    _log(state, f"Recipe:             {state.recipe_path} "
                f"({recipe.get('name', '?')})")
    _log(state, f"Node budget:        {state.max_nodes} total "
                f"(1 master + up to {state.max_nodes - 1} sbatch blocks)")
    _log(state, f"Master GPUs:        {state.gpus_per_node} "
                f"(one local worker per GPU)")
    _log(state, f"Resume mode:        {state.resume}")

    state.phases = _build_phase_jobs(state)
    _log(state, f"Built {len(state.phases)} phase jobs from source agent")
    if state.resume:
        n_resume = sum(1 for j in state.phases if j.resume_ckpt)
        n_fresh = len(state.phases) - n_resume
        _log(state, f"  Resuming {n_resume} phase(s) from prior latest.pt; "
                    f"{n_fresh} starting fresh (no prior checkpoint)")
        for j in state.phases:
            if j.resume_ckpt:
                _log(state, f"    phase {j.phase:02d}: resume_from "
                            f"{j.resume_ckpt}")
            else:
                _log(state, f"    phase {j.phase:02d}: init_from "
                            f"{j.source_ckpt} (no prior latest.pt)")

    run_id = os.path.basename(output_root)

    try:
        while True:
            _check_running(state)
            _retry_failed_if_eligible(state)
            _schedule_pending(state, run_id)

            done = all(j.status in ("completed",) for j in state.phases)
            unrecoverable = [j for j in state.phases
                             if j.status == "failed"
                             and j.attempts > state.max_retries]
            if done:
                _log(state, "All phases completed.")
                break
            if unrecoverable:
                _log(state, f"FATAL: {len(unrecoverable)} phase(s) failed "
                            f"beyond retry limit -- aborting")
                for j in unrecoverable:
                    _log(state, f"  phase {j.phase:02d}: {j.last_error}")
                return 1

            _log(state, f"Status: {_status_summary(state)}")
            time.sleep(state.poll_interval_s)
    except KeyboardInterrupt:
        _log(state, "Interrupted -- not packing; leaving phase runs in place")
        return 130

    # Pack the final agent.
    try:
        agent_dir = _pack_agent(state)
        _log(state, f"Packed fine-tuned agent: {agent_dir}")
    except Exception as e:
        _log(state, f"Pack FAILED: {e}\n{traceback.format_exc()}")
        return 2

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate fine-tuning of every phase of a composite "
                    "ELF PTT bootstrap agent.")
    parser.add_argument("--source-agent", required=True,
                        help="Path to source agent dir "
                             "(agents/agent_*).")
    parser.add_argument("--recipe", required=True,
                        help="Fine-tune recipe YAML.")
    parser.add_argument("--output-root", required=True,
                        help="New dir under agent_finetuning/...; "
                             "must NOT be inside --source-agent.")
    parser.add_argument("--max-nodes", type=int, default=4,
                        help="Total number of nodes the run is allowed "
                             "to use, INCLUDING the master's own node. "
                             "Default 4 (1 master + up to 3 sbatch node-"
                             "block workers). With 4-GPU nodes that's "
                             "16 concurrent fine-tunes -- enough to run "
                             "all 15 phases of a bootstrap agent in "
                             "parallel and still keep nodes free.")
    parser.add_argument("--gpus-per-node", type=int, default=None,
                        help="Number of local GPU slots on the master's "
                             "compute node; one local worker is pinned to "
                             "each via CUDA_VISIBLE_DEVICES. Default: "
                             "auto-detect via torch.cuda.device_count(). "
                             "Master sbatch must have requested at least "
                             "this many GPUs (--gres=gpu:N via launcher).")
    parser.add_argument("--poll-interval-s", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=1,
                        help="Per-phase retry budget on worker failure "
                             "(default 1 = one retry).")
    parser.add_argument("--slurm-time", type=str, default=SLURM_TIME_DEFAULT,
                        help=f"SLURM --time for worker jobs "
                             f"(default {SLURM_TIME_DEFAULT}).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an existing fine-tune job rooted at "
                             "--output-root. For each phase, scans "
                             "<output-root>/phases/phase_NN/ppo_optomech_"
                             "*/checkpoints/latest.pt and continues "
                             "training from the most recent one (full "
                             "model + optimizer + global_step resume). "
                             "Phases with no prior checkpoint fall back "
                             "to init-from-source. Stale status.json "
                             "files are moved aside.")
    args = parser.parse_args()

    # Read-only invariant: refuse to clobber the source agent.
    source_abs = os.path.abspath(args.source_agent)
    out_abs = os.path.abspath(args.output_root)
    if out_abs.startswith(source_abs):
        print(f"ERROR: --output-root ({out_abs}) cannot be inside "
              f"--source-agent ({source_abs}). The source agent is "
              f"READ-ONLY.")
        sys.exit(1)

    sys.exit(run_master(args))


if __name__ == "__main__":
    main()
