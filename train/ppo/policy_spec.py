"""
Policy specification loader.

Parses a YAML file describing a single or composite policy and returns
a ready-to-run BaseAgent.

Spec format
-----------

Single model (equivalent to passing --checkpoint):

    type: single
    checkpoint: runs/.../best.pt

Stepwise composite:

    type: composite
    phases:
      - checkpoint: runs/.../model_a.pt
        until:
          step: 32
      - checkpoint: runs/.../model_b.pt
        # last phase has no 'until' -> runs for remainder

Metric-threshold composite:

    type: composite
    phases:
      - checkpoint: runs/.../coarse.pt
        until:
          metric_above:
            strehl: 0.8
      - checkpoint: runs/.../fine.pt

Episode-fraction composite:

    type: composite
    max_episode_steps: 256        # required for fraction triggers
    phases:
      - checkpoint: runs/.../early.pt
        until:
          episode_fraction: 0.25
      - checkpoint: runs/.../late.pt
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from train.ppo.agents import (
    BaseAgent,
    CompositeAgent,
    CompositeHidden,
    EpisodeFractionTrigger,
    MetricThresholdTrigger,
    NeverTrigger,
    Phase,
    SingleModelAgent,
    StepTrigger,
)
from train.ppo.ppo_models import PPOActorWrapper


def _resolve_path(path: str, spec_dir: str) -> str:
    """Resolve a checkpoint path relative to the spec file's directory."""
    if os.path.isabs(path):
        return path
    # Try relative to spec file first, then repo root
    candidate = os.path.join(spec_dir, path)
    if os.path.exists(candidate):
        return candidate
    candidate = os.path.join(_REPO_ROOT, path)
    if os.path.exists(candidate):
        return candidate
    return path  # return as-is; load_agent will give a clear error


def _load_single_model(checkpoint_path: str, env, device: str,
                       action_dim_override: int | None = None):
    """Load a single checkpoint into a SingleModelAgent.

    Returns (SingleModelAgent, obs_ref_max, config). The config dict is
    the one stored in the checkpoint — callers can inspect it to build
    per-phase action masks for bootstrap composite rollouts.

    ``action_dim_override``: when set, the model is built at this
    explicit action width instead of the live env's action_dim. Used
    by composite loading so each phase's head shape comes from its
    own checkpoint, not the (potentially-wider, combined) env action
    space.
    """
    from train.ppo.rollout import load_agent
    agent, config, obs_ref_max = load_agent(
        checkpoint_path, env, device,
        action_dim_override=action_dim_override)
    wrapper = PPOActorWrapper(agent)
    return SingleModelAgent(wrapper, device), obs_ref_max, config


# ---------------------------------------------------------------------------
# Output-adapter construction from spec
# ---------------------------------------------------------------------------

def _build_output_adapter(adapter_spec: Optional[dict], env, device):
    """Translate a per-phase ``adapter:`` spec block into an
    OutputAdapter instance. Returns None when adapter_spec is None
    (legacy homogeneous-composite path)."""
    if adapter_spec is None:
        return None
    from train.ppo.output_adapters import (
        SliceAdapter, ZernikeToDMAdapter,
        SegmentZernikePassthroughAdapter)

    a_type = adapter_spec.get("type")
    env_action_dim = int(env.action_space.shape[0])

    if a_type == "slice":
        ts = adapter_spec["target_slice"]
        if not (isinstance(ts, (list, tuple)) and len(ts) == 2):
            raise ValueError(
                "slice adapter requires target_slice: [start, stop|null]")
        start = int(ts[0])
        stop = (None if ts[1] is None else int(ts[1]))
        return SliceAdapter(start, stop, env_action_dim)

    if a_type == "zernike_to_dm":
        n_modes = int(adapter_spec["n_modes"])
        dm_slice = adapter_spec.get("dm_slice")
        if dm_slice is not None:
            if not (isinstance(dm_slice, (list, tuple)) and len(dm_slice) == 2):
                raise ValueError(
                    "zernike_to_dm adapter dm_slice must be "
                    "[start, stop]")
            dm_slice = (int(dm_slice[0]), int(dm_slice[1]))
        return ZernikeToDMAdapter.from_env(
            env, n_zernike=n_modes,
            dm_slice=dm_slice,
            env_action_dim=env_action_dim,
            action_scale=float(adapter_spec.get("action_scale", 1.0)),
            skip_piston=bool(adapter_spec.get("skip_piston", False)),
            normalize=str(adapter_spec.get("normalize", "wavefront_rms")),
            regularize_rcond=float(
                adapter_spec.get("regularize_rcond", 1e-3)),
            device=device,
            silence=False)

    if a_type == "segment_zernike_passthrough":
        n_modes = int(adapter_spec["n_zernike"])
        n_seg = adapter_spec.get("n_seg_actions")
        if n_seg is not None:
            n_seg = int(n_seg)
        return SegmentZernikePassthroughAdapter.from_env(
            env, n_zernike=n_modes,
            n_seg_actions=n_seg,
            env_action_dim=env_action_dim,
            dm_action_scale=float(
                adapter_spec.get("dm_action_scale", 1.0)),
            skip_piston=bool(adapter_spec.get("skip_piston", False)),
            normalize=str(adapter_spec.get("normalize", "wavefront_rms")),
            regularize_rcond=float(
                adapter_spec.get("regularize_rcond", 1e-3)),
            device=device,
            silence=False)

    raise ValueError(f"Unknown adapter type: {a_type!r}")


def read_env_kwarg_overrides(spec_path: str) -> dict:
    """Read the ``env_kwarg_overrides:`` block from a policy spec
    without loading the agent. The rollout driver merges this on top
    of its ROLLOUT_ENV_KWARGS BEFORE building the env, so a composite
    bundle can declare the combined action space (and any other env
    requirements) it needs to run. Returns an empty dict if missing.
    """
    if not spec_path or not os.path.exists(spec_path):
        return {}
    with open(spec_path) as f:
        spec = yaml.safe_load(f) or {}
    return dict(spec.get("env_kwarg_overrides") or {})


def _build_bootstrap_action_mask(config: dict, action_dim: int,
                                 num_apertures_hint: int = 15,
                                 adapter=None):
    """Return a torch float mask [action_dim] if this checkpoint's env
    config has bootstrap_mask_nontarget (or equivalent bootstrap-phase
    markers) set; otherwise None.

    Action layout: when ``adapter`` is a
    ``SegmentZernikePassthroughAdapter``, the native action vector is
    ``[seg_PTT (n_seg_actions) | zernike (n_zernike)]`` and the mask
    is built per-segment for the first block, all-ones for the
    Zernike block (DM head is exploration-free, mirroring training).
    Without that adapter, action_dim is assumed to tile evenly into
    ``dof_per_seg``-DOF segment blocks (the legacy homogeneous-
    composite case).
    """
    env_kwargs = (config or {}).get("env_kwargs", {}) or {}
    if not env_kwargs.get("bootstrap_phase", False):
        return None
    if not env_kwargs.get("bootstrap_mask_nontarget", False):
        return None

    target = int(env_kwargs.get("bootstrap_phased_count", 0))
    command_tip_tilt = bool(env_kwargs.get("command_tip_tilt", False))
    dof_per_seg = 3 if command_tip_tilt else 1
    if dof_per_seg <= 0 or action_dim <= 0:
        return None

    # Determine the segment-PTT block width. When a passthrough adapter
    # is in scope it tells us exactly (n_seg_actions); else infer from
    # action_dim (and bail if it doesn't tile cleanly).
    n_seg_actions = None
    n_tail = 0
    if adapter is not None:
        # Lazy import; output_adapters is a sibling module.
        from train.ppo.output_adapters import (
            SegmentZernikePassthroughAdapter)
        if isinstance(adapter, SegmentZernikePassthroughAdapter):
            n_seg_actions = int(adapter.n_seg_actions)
            n_tail = int(adapter.n_zernike)
    if n_seg_actions is None:
        n_seg = action_dim // dof_per_seg
        if n_seg * dof_per_seg != action_dim:
            # Doesn't tile -- skip masking to be safe.
            return None
        n_seg_actions = action_dim

    n_seg = n_seg_actions // dof_per_seg
    if n_seg * dof_per_seg != n_seg_actions:
        return None

    mask = np.zeros(action_dim, dtype=np.float32)
    if 0 <= target < n_seg:
        mask[target * dof_per_seg] = 1.0
        if command_tip_tilt:
            mask[target * dof_per_seg + 1] = 1.0
            mask[target * dof_per_seg + 2] = 1.0
    # Tail (DM / Zernike head): pass through unmasked. The DM head
    # was trained with bootstrap_mask_nontarget=True restricting only
    # the segment block; the Zernike outputs are free to fire from
    # step 0 of every phase.
    if n_tail > 0:
        mask[n_seg_actions: n_seg_actions + n_tail] = 1.0
    return torch.from_numpy(mask)


def _parse_trigger(until_spec: Optional[dict], max_episode_steps: int):
    """Parse the 'until' block of a phase into a Trigger."""
    if until_spec is None:
        return NeverTrigger()

    if "step" in until_spec:
        return StepTrigger(int(until_spec["step"]))

    if "metric_above" in until_spec:
        mapping = until_spec["metric_above"]
        metric, value = next(iter(mapping.items()))
        return MetricThresholdTrigger(metric, float(value), direction="above")

    if "metric_below" in until_spec:
        mapping = until_spec["metric_below"]
        metric, value = next(iter(mapping.items()))
        return MetricThresholdTrigger(metric, float(value), direction="below")

    if "episode_fraction" in until_spec:
        return EpisodeFractionTrigger(
            float(until_spec["episode_fraction"]),
            max_steps=max_episode_steps,
        )

    raise ValueError(f"Unknown trigger spec: {until_spec}")


def load_policy_spec(
    spec_path: str,
    env,
    device: str = "cpu",
    max_episode_steps: int = 256,
) -> tuple[BaseAgent, float]:
    """Load a policy spec YAML and return (agent, obs_ref_max).

    Parameters
    ----------
    spec_path : str
        Path to the YAML policy specification file.
    env : gymnasium.Env
        The environment (needed to construct agents).
    device : str
        Torch device.
    max_episode_steps : int
        Max steps per episode (used by fraction triggers).

    Returns
    -------
    agent : BaseAgent
        Ready-to-run agent.
    obs_ref_max : float
        Observation reference max for normalization.
    """
    spec_dir = os.path.dirname(os.path.abspath(spec_path))

    with open(spec_path, "r") as f:
        spec = yaml.safe_load(f)

    policy_type = spec.get("type", "single")
    max_steps = spec.get("max_episode_steps", max_episode_steps)

    if policy_type == "single":
        ckpt = _resolve_path(spec["checkpoint"], spec_dir)
        model, obs_ref_max, _cfg = _load_single_model(ckpt, env, device)
        return model, obs_ref_max

    elif policy_type == "composite":
        phase_specs = spec["phases"]
        if len(phase_specs) < 1:
            raise ValueError("Composite policy requires at least 1 phase")

        phases = []
        obs_ref_max = None
        for idx, ps in enumerate(phase_specs):
            ckpt = _resolve_path(ps["checkpoint"], spec_dir)

            # Peek at the checkpoint's policy_head shape to know what
            # action_dim THIS phase needs (which may differ from the
            # env's action_dim when adapters are in play). Falls back
            # to env action_dim when the checkpoint doesn't follow the
            # standard key layout.
            from train.ppo.rollout import _action_dim_from_checkpoint
            _peek = torch.load(ckpt, map_location="cpu", weights_only=False)
            native_dim = _action_dim_from_checkpoint(_peek)
            del _peek

            model, orm, ckpt_cfg = _load_single_model(
                ckpt, env, device,
                action_dim_override=native_dim)
            if obs_ref_max is None:
                obs_ref_max = orm

            trigger = _parse_trigger(ps.get("until"), max_steps)
            name = ps.get("name", os.path.basename(ckpt))

            # Build a hard DOF mask from this checkpoint's training env
            # Per-phase output adapter (slice or zernike_to_dm /
            # segment_zernike_passthrough). None in the legacy
            # homogeneous-composite case; required when phases have
            # heterogeneous heads. Built BEFORE the action mask so
            # the mask builder can ask the adapter where the segment
            # block ends and the DM (Zernike) block begins.
            adapter = _build_output_adapter(ps.get("adapter"), env, device)

            # Build a hard DOF mask from this checkpoint's training env
            # config so the composite rollout enforces the same
            # structural constraint that training did (only the target
            # segment's 3 DOFs can actuate for this phase). Sized to
            # the phase's NATIVE action dim -- the adapter is what
            # places the masked native vector into the wider env
            # action space. When the adapter is a
            # SegmentZernikePassthroughAdapter, the mask builder is
            # told where the segment block ends so the Zernike-DM
            # tail is left unmasked (matches training).
            action_mask = _build_bootstrap_action_mask(
                ckpt_cfg, model.action_dim, adapter=adapter)

            if adapter is not None and adapter.env_action_dim != int(
                    env.action_space.shape[0]):
                raise ValueError(
                    f"Adapter {adapter} env_action_dim doesn't match "
                    f"live env action_dim "
                    f"{env.action_space.shape[0]}. The composite spec "
                    f"likely declares env_kwarg_overrides that weren't "
                    f"applied before env construction.")

            # Pull the original training-time phased_count off the
            # checkpoint so the rollout driver can reconfigure the env
            # at each phase transition.
            ck_env = (ckpt_cfg or {}).get("env_kwargs", {}) or {}
            bp_count = (int(ck_env["bootstrap_phased_count"])
                        if "bootstrap_phased_count" in ck_env
                        else None)

            phases.append(Phase(model, trigger, name=name,
                                action_mask=action_mask,
                                bootstrap_phased_count=bp_count,
                                output_adapter=adapter))

        agent = CompositeAgent(phases, device=device)
        return agent, obs_ref_max

    else:
        raise ValueError(f"Unknown policy type: {policy_type!r}")
