"""Differentiable policy-action to physical-control conversion."""

from __future__ import annotations

import torch

from .spec import INFANTRY_SPEC

ACTION_SCALE = (0.5, 0.5, 10.0, 0.5, 0.5, 10.0)


def policy_action_to_control(
    action: torch.Tensor,
    *,
    default_motor_targets: torch.Tensor,
    gas_spring_forces: torch.Tensor,
) -> torch.Tensor:
    """Return four position, two velocity, and two tendon-force commands.

    ``action`` remains the normalized six-dimensional policy output.  The
    returned eight values are what the dynamics model must consume.
    """
    if action.shape[-1] != INFANTRY_SPEC.policy_action_dim:
        raise ValueError(
            f"expected action width {INFANTRY_SPEC.policy_action_dim}, got {action.shape[-1]}"
        )
    if default_motor_targets.shape[-1] != INFANTRY_SPEC.policy_action_dim:
        raise ValueError("default_motor_targets must have width 6")
    if gas_spring_forces.shape[-1] != 2:
        raise ValueError("gas_spring_forces must have width 2")
    scale = action.new_tensor(ACTION_SCALE)
    scaled = default_motor_targets + action * scale
    # Wheel action offsets are velocity targets and are always zero in mjlab.
    scaled = torch.stack(
        (
            scaled[..., 0],
            scaled[..., 1],
            action[..., 2] * scale[2],
            scaled[..., 3],
            scaled[..., 4],
            action[..., 5] * scale[5],
        ),
        dim=-1,
    )
    return torch.cat((scaled, gas_spring_forces), dim=-1)
