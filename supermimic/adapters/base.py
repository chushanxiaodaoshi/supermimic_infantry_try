"""Minimal simulator contract consumed by collectors and evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from supermimic.core.spec import RobotSpec, TensorState


@dataclass
class AdapterStep:
    state: TensorState
    actor_observation: torch.Tensor
    command: torch.Tensor
    policy_action: torch.Tensor
    applied_control: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    context: torch.Tensor
    task_id: torch.Tensor
    timestamp: torch.Tensor
    extras: dict[str, Any]

    @property
    def done(self) -> torch.Tensor:
        return self.terminated | self.truncated


class RobotAdapter(ABC):
    """Narrow boundary between simulator-specific code and learning code."""

    spec: RobotSpec

    @abstractmethod
    def reset(self, env_ids: torch.Tensor | None = None) -> AdapterStep:
        """Reset selected environments and return their new state."""

    @abstractmethod
    def step(self, policy_action: torch.Tensor) -> AdapterStep:
        """Advance one control step using normalized policy output."""

    @abstractmethod
    def state(self) -> TensorState:
        """Return the complete privileged simulator state."""

    @abstractmethod
    def observation(self) -> torch.Tensor:
        """Return only quantities available to the deployed actor."""

    @abstractmethod
    def command(self) -> torch.Tensor:
        """Return the current task command."""

    @abstractmethod
    def context(self) -> torch.Tensor:
        """Return simulator/domain parameters in ``spec.context_names`` order."""

    @abstractmethod
    def policy_action_to_control(self, action: torch.Tensor) -> torch.Tensor:
        """Map normalized actions to the actual values applied to physics."""

    @abstractmethod
    def serialize_state(self) -> dict[str, torch.Tensor]:
        """Serialize all state required for deterministic replay."""

    @abstractmethod
    def restore_state(self, payload: dict[str, torch.Tensor]) -> None:
        """Restore a state produced by :meth:`serialize_state`."""
