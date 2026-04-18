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
    """Interface every agent must satisfy.

    Note on action masking:
        ``__call__`` returns the policy's *unmasked* action (after scale
        and clamp). The caller is responsible for applying any env-
        facing mask via ``apply_action_mask`` before stepping the env.
        This mirrors training, where the policy's own ``prior_action``
        input is the pre-mask scaled-and-clamped action; the env-side
        mask is applied only to what the optical system actuates.
    """

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

    def apply_action_mask(self, action: torch.Tensor) -> torch.Tensor:
        """Return the env-facing action (mask-applied). Default is a
        no-op; CompositeAgent overrides with per-phase masking."""
        return action

    def notify_step(self, step_info: dict) -> None:
        """Post env.step hook.  Default is a no-op."""
        pass

    @property
    def just_transitioned(self) -> bool:
        """True on the step immediately after a phase transition fired.

        Caller is expected to zero ``prior_action`` and ``prior_reward``
        when this is True so the incoming phase starts from the same
        (zero, zero) baseline it saw at training step 0. Cleared on the
        next ``__call__``. Default False; only CompositeAgent sets it.
        """
        return False

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
    """One phase of a composite policy.

    Attributes
    ----------
    action_mask : torch.Tensor or None
        Optional per-DOF mask (shape [action_dim]) multiplied into the
        agent's action before it's returned to the caller. Used by the
        bootstrap composite rollout to mirror the hard DOF mask the
        training env applies: only the target segment's 3 DOFs are
        non-zero, everything else is forced to zero. None means no
        masking.
    """

    def __init__(self, agent: SingleModelAgent, trigger: Trigger, name: str = "",
                 action_mask: Optional[torch.Tensor] = None):
        self.agent = agent
        self.trigger = trigger
        self.name = name
        self.action_mask = action_mask


class CompositeAgent(BaseAgent):
    """Runs a sequence of sub-agents, advancing phases based on triggers.

    Hidden-state handling on phase transition:
      - Each phase maintains its own LSTM hidden state.
      - On a switch, the new phase's hidden is freshly zeroed (the LSTM
        representations are not transferable across independently trained
        models).
      - prior_action and prior_reward are ALSO reset to zero on switch,
        because each phase was trained from step 0 with zero prior
        action and zero prior reward. The caller reads ``just_transitioned``
        after ``notify_step`` and zeros its prior_* tensors accordingly.

    Action masking:
      - ``__call__`` returns the *unmasked* scale-and-clamped action
        (matches training, where the policy's own prior_action input is
        the pre-mask value).
      - ``apply_action_mask`` multiplies by the current phase's per-DOF
        mask and returns the env-facing action.
    """

    def __init__(self, phases: List[Phase], device: str = "cpu"):
        if not phases:
            raise ValueError("CompositeAgent requires at least one phase")
        self._phases = phases
        self._device = device
        self._active = 0
        self._global_step = 0
        # Latched True on the step a phase switch happens. Read by the
        # caller immediately after notify_step and cleared on the next
        # __call__ (which is the first __call__ of the new phase).
        self._just_transitioned = False

    # -- public interface ---------------------------------------------------

    def get_zero_hidden(self) -> CompositeHidden:
        hiddens = [p.agent.get_zero_hidden() for p in self._phases]
        self._active = 0
        self._global_step = 0
        self._just_transitioned = False
        for p in self._phases:
            p.trigger.reset()
        return CompositeHidden(hiddens, active_phase=0)

    def __call__(self, obs, prior_action, prior_reward, hidden: CompositeHidden):
        # First call following a transition — clear the flag so the
        # caller's reset (if any) only fires once per transition.
        self._just_transitioned = False
        phase = self._phases[self._active]
        h = hidden.phase_hiddens[self._active]
        action, new_h = phase.agent(obs, prior_action, prior_reward, h)
        hidden.phase_hiddens[self._active] = new_h
        # NOTE: the mask is NOT applied here. Caller must call
        # apply_action_mask(action) before stepping the env.
        return action, hidden

    def apply_action_mask(self, action: torch.Tensor) -> torch.Tensor:
        """Multiply action by the current phase's DOF mask (if any)."""
        phase = self._phases[self._active]
        if phase.action_mask is None:
            return action
        return action * phase.action_mask.to(action.device)

    def notify_step(self, step_info: dict) -> None:
        self._global_step += 1
        phase = self._phases[self._active]
        if self._active < len(self._phases) - 1:
            if phase.trigger.should_fire(self._global_step, step_info):
                old_name = phase.name or f"phase-{self._active}"
                self._active += 1
                new_name = self._phases[self._active].name or f"phase-{self._active}"
                self._just_transitioned = True
                print(f"  [composite] step {self._global_step}: "
                      f"{old_name} -> {new_name}")

    @property
    def just_transitioned(self) -> bool:
        return self._just_transitioned

    @property
    def action_dim(self) -> int:
        return self._phases[0].agent.action_dim
