"""Short-horizon differentiable actor rollout utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch
from torch import nn

from .networks import UnifiedActor, WorldModel
from .spec import TensorState

ControlFn = Callable[[torch.Tensor], torch.Tensor]
ObserveFn = Callable[[TensorState, torch.Tensor], torch.Tensor]
RewardFn = Callable[
    [TensorState, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]


@contextmanager
def freeze_parameters(module: nn.Module) -> Iterator[None]:
    """Freeze parameters while retaining derivatives with respect to inputs."""
    previous = [parameter.requires_grad for parameter in module.parameters()]
    try:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(module.parameters(), previous, strict=True):
            parameter.requires_grad_(requires_grad)


def imagined_actor_loss(
    *,
    actor: UnifiedActor,
    world_model: WorldModel,
    initial_state: TensorState,
    observation_history: torch.Tensor,
    context: torch.Tensor,
    control_fn: ControlFn,
    observe_fn: ObserveFn,
    reward_fn: RewardFn,
    horizon: int,
    gamma: float,
    goal: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Roll out Actor -> control transform -> World Model -> reward with BPTT.

    ``reward_fn`` returns ``(reward, soft_done_probability)``.  World-model
    parameters are frozen, but the graph through its inputs remains intact.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    state = initial_state
    history = observation_history
    alive = torch.ones(history.shape[0], device=history.device, dtype=history.dtype)
    objective = torch.zeros_like(alive)
    rewards: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    with freeze_parameters(world_model):
        for step in range(horizon):
            policy_action = actor(history, goal)
            applied_control = control_fn(policy_action)
            state = world_model.step(state, applied_control, context)
            reward, done_probability = reward_fn(state, policy_action, applied_control)
            objective = objective + (gamma**step) * alive * reward
            alive = alive * (1.0 - done_probability).clamp(0.0, 1.0)
            observation = observe_fn(state, policy_action)
            history = torch.cat((history[:, 1:], observation.unsqueeze(1)), dim=1)
            rewards.append(reward)
            actions.append(policy_action)
    diagnostics = {
        "imagined_reward": torch.stack(rewards).mean().detach(),
        "survival": alive.mean().detach(),
        "mean_abs_action": torch.stack(actions).abs().mean().detach(),
    }
    return -objective.mean(), diagnostics
