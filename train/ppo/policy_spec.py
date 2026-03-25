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


def _load_single_model(checkpoint_path: str, env, device: str) -> SingleModelAgent:
    """Load a single checkpoint into a SingleModelAgent."""
    from train.ppo.rollout import load_agent
    agent, config, obs_ref_max = load_agent(checkpoint_path, env, device)
    wrapper = PPOActorWrapper(agent)
    return SingleModelAgent(wrapper, device), obs_ref_max


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
        return _load_single_model(ckpt, env, device)

    elif policy_type == "composite":
        phase_specs = spec["phases"]
        if len(phase_specs) < 2:
            raise ValueError("Composite policy requires at least 2 phases")

        phases = []
        obs_ref_max = None
        for idx, ps in enumerate(phase_specs):
            ckpt = _resolve_path(ps["checkpoint"], spec_dir)
            model, orm = _load_single_model(ckpt, env, device)
            if obs_ref_max is None:
                obs_ref_max = orm

            trigger = _parse_trigger(ps.get("until"), max_steps)
            name = ps.get("name", os.path.basename(ckpt))
            phases.append(Phase(model, trigger, name=name))

        agent = CompositeAgent(phases, device=device)
        return agent, obs_ref_max

    else:
        raise ValueError(f"Unknown policy type: {policy_type!r}")
