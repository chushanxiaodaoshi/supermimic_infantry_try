"""Semantically explicit transition record used by all collectors."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.core.spec import RobotSpec


@dataclass
class TransitionRecord:
    state: torch.Tensor
    next_state: torch.Tensor
    actor_observation: torch.Tensor
    observation_history: torch.Tensor
    command: torch.Tensor
    teacher_action: torch.Tensor
    applied_control: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    context: torch.Tensor
    timestamp: torch.Tensor
    dt: torch.Tensor
    task_id: torch.Tensor
    episode_id: torch.Tensor

    def validate(self, spec: RobotSpec) -> None:
        widths = {
            "state": (self.state, spec.state_dim),
            "next_state": (self.next_state, spec.state_dim),
            "actor_observation": (self.actor_observation, spec.actor_obs_dim),
            "command": (self.command, spec.command_dim),
            "teacher_action": (self.teacher_action, spec.policy_action_dim),
            "applied_control": (self.applied_control, spec.control_dim),
            "context": (self.context, spec.context_dim),
        }
        for name, (value, width) in widths.items():
            if value.shape[-1] != width:
                raise ValueError(
                    f"{name} must have width {width}, got {value.shape[-1]}"
                )
        if self.observation_history.shape[-1] != spec.actor_obs_dim:
            raise ValueError("observation_history has the wrong observation width")
        batch_shape = self.state.shape[:-1]
        for name, value in self.__dict__.items():
            if name == "observation_history":
                candidate = value.shape[:-2]
            elif name in widths:
                candidate = value.shape[:-1]
            else:
                candidate = value.shape
            if candidate != batch_shape:
                raise ValueError(
                    f"{name} batch shape {candidate} does not match state {batch_shape}"
                )
