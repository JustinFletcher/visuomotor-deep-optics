"""SLURM node introspection + selection for the agent fine-tuner.

The fine-tune master and launcher both want the same thing: query
``sinfo`` for currently-available compute nodes, sort by GPU count
(biggest first), and pick the next-best one that's not already in
use. Lives in one module so both layers behave identically.

The cluster this targets does not reliably oversubscribe nodes
(multiple --gres=gpu:1 jobs on a 4-GPU node may not run
concurrently), so the strategy is to:

  - explicitly --nodelist each sbatch invocation,
  - request --gres=gpu:N where N matches the picked node's GPU
    count, and
  - have the body of that sbatch job fan out K=N worker
    subprocesses internally (see finetune_node_block_worker.py).

This way the master can saturate a multi-GPU node with one SLURM job
instead of relying on the partition's OverSubscribe setting.
"""
from __future__ import annotations

import re
import subprocess
import socket
from dataclasses import dataclass
from typing import Iterable, Optional


# Matches `gpu:4`, `gpu:tesla:4`, `gpu:a100-40gb:8(IDX:0-7)`, etc.
_GRES_GPU_RE = re.compile(r"gpu:(?:[^:,()]+:)?(\d+)")

# Which SLURM states do we treat as "GPUs definitely free on this node"?
# - idle:      all resources free
# - mix:       node is partially used (we lose -- SLURM may queue our job
#              but it might also schedule it next; cheaper to skip)
# - allocated: in use, skip
# - down/drain/inval/maint: skip
# We default to ``idle`` only to keep things predictable. Set
# allow_mix=True to also include "mix" nodes.
_DEFAULT_OK_STATES = frozenset({"idle", "idle*"})
_MIX_STATES = frozenset({"mix", "mix*", "mixed", "mixed*"})


@dataclass(frozen=True)
class NodeInfo:
    name: str
    state: str
    gpu_count: int
    partition: Optional[str] = None


def _parse_gres(gres: str) -> int:
    """Return the total GPU count across all gres tokens in ``gres``.
    Handles (null) / empty / multi-resource strings."""
    if not gres or gres in ("(null)", "n/a", "None"):
        return 0
    total = 0
    for m in _GRES_GPU_RE.finditer(gres):
        try:
            total += int(m.group(1))
        except (TypeError, ValueError):
            continue
    return total


def query_sinfo_nodes(partition: Optional[str] = None,
                      timeout_s: float = 30.0,
                      sinfo_bin: str = "sinfo") -> list[NodeInfo]:
    """Run ``sinfo`` once and parse the result.

    Output format ``%N|%t|%G|%R``: node name, compact state, gres,
    partition. ``-N`` so each node appears once.

    Empty list on any failure -- callers should fall back gracefully
    (e.g. let SLURM pick a node the old way).
    """
    cmd = [sinfo_bin, "-h", "-N", "-o", "%N|%t|%G|%R"]
    if partition:
        cmd += ["-p", partition]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    out: list[NodeInfo] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 4:
            continue
        name, state, gres, part = parts[0], parts[1], parts[2], parts[3]
        # When --partition is unset and a node appears in multiple
        # partitions, sinfo lists it once per partition; keep them all
        # so callers can filter later.
        out.append(NodeInfo(
            name=name.strip(),
            state=state.strip().lower(),
            gpu_count=_parse_gres(gres.strip()),
            partition=part.strip() or None,
        ))
    return out


def select_best_nodes(nodes: Iterable[NodeInfo],
                      in_use: Iterable[str] = (),
                      min_gpus: int = 1,
                      allow_mix: bool = False,
                      exclude_self: bool = True) -> list[NodeInfo]:
    """Filter + sort: idle (or mix when allowed), gpu_count >= min_gpus,
    not already in use. Sorted by gpu_count descending, then name.

    ``in_use`` is a list of node names the caller is already targeting
    (or running on) and wants to skip. ``exclude_self`` adds the
    current hostname to the exclude set automatically (useful when
    the master is picking sub-worker nodes and shouldn't pick its
    own).
    """
    in_use_set = {n for n in in_use}
    if exclude_self:
        try:
            in_use_set.add(socket.gethostname().split(".")[0])
        except Exception:
            pass

    ok_states = set(_DEFAULT_OK_STATES)
    if allow_mix:
        ok_states |= _MIX_STATES

    out = []
    for n in nodes:
        if n.state not in ok_states:
            continue
        if n.gpu_count < min_gpus:
            continue
        if n.name in in_use_set:
            continue
        out.append(n)
    # Dedup by name (a node in multiple partitions yields multiple
    # entries; keep the one with the highest gpu_count, then any).
    by_name: dict[str, NodeInfo] = {}
    for n in out:
        existing = by_name.get(n.name)
        if existing is None or n.gpu_count > existing.gpu_count:
            by_name[n.name] = n
    deduped = list(by_name.values())
    deduped.sort(key=lambda n: (-n.gpu_count, n.name))
    return deduped


def pick_one(nodes: Iterable[NodeInfo],
             in_use: Iterable[str] = (),
             min_gpus: int = 1,
             allow_mix: bool = False,
             exclude_self: bool = True) -> Optional[NodeInfo]:
    """Convenience: ``select_best_nodes(...)[0]`` or ``None``."""
    best = select_best_nodes(
        nodes, in_use=in_use, min_gpus=min_gpus,
        allow_mix=allow_mix, exclude_self=exclude_self)
    return best[0] if best else None
