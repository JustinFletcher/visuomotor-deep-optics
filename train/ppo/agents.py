"""
Agent abstractions for rollout and sweep evaluation.

Provides a unified interface for single-model and composite agents so that
rollout.py and sweep_tiptilt.py can transparently run any policy -- whether
it is a single checkpoint or a multi-phase composite of several models.

Agent protocol
--------------
Every agent exposes:
    agent.get_zero_hidden()  -> hidden state (opaque to caller)
    agent(obs, prior_action, prior_reward, hidden) -> (action, new_hidden)
    agent.notify_step(step_info: dict)  -> None   [optional post-step hook]
    agent.action_dim  -> int

The hidden state is opaque: callers must not inspect its structure.  Composite
agents pack/unpack sub-agent hidden states internally.
"""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from train.ppo.ppo_models import RecurrentActorCritic, PPOActorWrapper


# ---------------------------------------------------------------------------
# Hidden-state container for composite agents
# ---------------------------------------------------------------------------

class CompositeHidden:
    """Bundles per-phase hidden states so callers can treat hidden as opaque."""

    def __init__(self, phase_hiddens: List[Tuple[torch.Tensor, torch.Tensor]],
                 active_phase: int):
        self.phase_hiddens = phase_hiddens
        self.active_phase = active_phase


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """Interface every agent must satisfy."""

    @abstractmethod
    def get_zero_hidden(self) -> Any:
        ...

    @abstractmethod
    def __call__(
        self,
        obs: torch.Tensor,
        prior_action: torch.Tensor,
        prior_reward: torch.Tensor,
        hidden: Any,
    ) -> Tuple[torch.Tensor, Any]:
        ...

    def notify_step(self, step_info: dict) -> None:
        """Post env.step hook.  Default is a no-op."""
        pass

    @property
    @abstractmethod
    def action_dim(self) -> int:
        ...


# ---------------------------------------------------------------------------
# Single-model agent  (wraps the existing PPOActorWrapper)
# ---------------------------------------------------------------------------

class SingleModelAgent(BaseAgent):
    """Drop-in wrapper around a loaded PPOActorWrapper."""

    def __init__(self, wrapper: PPOActorWrapper, device: str = "cpu"):
        self._wrapper = wrapper
        self._device = device

    def get_zero_hidden(self):
        h = self._wrapper.get_zero_hidden()
        return (h[0].to(self._device), h[1].to(self._device))

    def __call__(self, obs, prior_action, prior_reward, hidden):
        with torch.no_grad():
            return self._wrapper(obs, prior_action, prior_reward, hidden)

    @property
    def action_dim(self) -> int:
        return self._wrapper.actor_critic.action_dim


# ---------------------------------------------------------------------------
# Phase transition triggers
# ---------------------------------------------------------------------------

class Trigger(ABC):
    """Decides whether to advance to the next phase."""

    @abstractmethod
    def should_fire(self, step: int, step_info: dict) -> bool:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...


class StepTrigger(Trigger):
    """Fire after a fixed number of steps."""

    def __init__(self, num_steps: int):
        self.num_steps = num_steps
        self._steps_in_phase = 0

    def should_fire(self, step: int, step_info: dict) -> bool:
        self._steps_in_phase += 1
        return self._steps_in_phase >= self.num_steps

    def reset(self) -> None:
        self._steps_in_phase = 0


class MetricThresholdTrigger(Trigger):
    """Fire when a metric crosses a threshold.

    direction='above' fires when metric >= value.
    direction='below' fires when metric <= value.
    """

    def __init__(self, metric: str, value: float, direction: str = "above"):
        self.metric = metric
        self.value = value
        self.direction = direction

    def should_fire(self, step: int, step_info: dict) -> bool:
        v = step_info.get(self.metric)
        if v is None:
            return False
        if self.direction == "above":
            return v >= self.value
        return v <= self.value

    def reset(self) -> None:
        pass


class EpisodeFractionTrigger(Trigger):
    """Fire after a fraction of max_episode_steps."""

    def __init__(self, fraction: float, max_steps: int):
        self.target_step = int(fraction * max_steps)
        self._steps_in_phase = 0

    def should_fire(self, step: int, step_info: dict) -> bool:
        self._steps_in_phase += 1
        return self._steps_in_phase >= self.target_step

    def reset(self) -> None:
        self._steps_in_phase = 0


class NeverTrigger(Trigger):
    """Terminal phase -- never fires."""

    def should_fire(self, step: int, step_info: dict) -> bool:
        return False

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Composite agent
# ---------------------------------------------------------------------------

class Phase:
    """One phase of a composite policy."""

    def __init__(self, agent: SingleModelAgent, trigger: Trigger, name: str = ""):
        self.agent = agent
        self.trigger = trigger
        self.name = name


class CompositeAgent(BaseAgent):
    """Runs a sequence of sub-agents, advancing phases based on triggers.

    Hidden-state handling on phase transition:
      - Each phase maintains its own LSTM hidden state.
      - On a switch, the new phase's hidden is freshly zeroed (the LSTM
        representations are not transferable across independently trained
        models).
      - prior_action and prior_reward carry over naturally since they are
        managed by the caller.
    """

    def __init__(self, phases: List[Phase], device: str = "cpu"):
        if not phases:
            raise ValueError("CompositeAgent requires at least one phase")
        self._phases = phases
        self._device = device
        self._active = 0
        self._global_step = 0

    # -- public interface ---------------------------------------------------

    def get_zero_hidden(self) -> CompositeHidden:
        hiddens = [p.agent.get_zero_hidden() for p in self._phases]
        self._active = 0
        self._global_step = 0
        for p in self._phases:
            p.trigger.reset()
        return CompositeHidden(hiddens, active_phase=0)

    def __call__(self, obs, prior_action, prior_reward, hidden: CompositeHidden):
        phase = self._phases[self._active]
        h = hidden.phase_hiddens[self._active]
        action, new_h = phase.agent(obs, prior_action, prior_reward, h)
        hidden.phase_hiddens[self._active] = new_h
        return action, hidden

    def notify_step(self, step_info: dict) -> None:
        self._global_step += 1
        phase = self._phases[self._active]
        if self._active < len(self._phases) - 1:
            if phase.trigger.should_fire(self._global_step, step_info):
                old_name = phase.name or f"phase-{self._active}"
                self._active += 1
                new_name = self._phases[self._active].name or f"phase-{self._active}"
                print(f"  [composite] step {self._global_step}: "
                      f"{old_name} -> {new_name}")

    @property
    def action_dim(self) -> int:
        return self._phases[0].agent.action_dim
