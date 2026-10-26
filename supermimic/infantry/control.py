"""Differentiable policy-action to physical-control conversion."""

from __future__ import annotations

import torch

from .spec import INFANTRY_SPEC


def policy_action_to_control(
    action: torch.Tensor,
    *,
    default_motor_targets: torch.Tensor,
    gas_spring_forces: torch.Tensor,
    action_scales: torch.Tensor,
    action_targets: tuple[str, ...],
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
    if action_scales.shape[-1] != INFANTRY_SPEC.policy_action_dim:
        raise ValueError("action_scales must have width 6")
    if set(action_targets) != set(INFANTRY_SPEC.policy_action_names):
        raise ValueError("action_targets do not match the infantry policy joints")
    scale = action_scales.to(device=action.device, dtype=action.dtype)
    scaled_by_teacher = action * scale
    order = [action_targets.index(name) for name in INFANTRY_SPEC.policy_action_names]
    ordered = scaled_by_teacher[..., order]
    scaled = default_motor_targets + ordered
    # Wheel action offsets are velocity targets and are always zero in mjlab.
    scaled = torch.stack(
        (
            scaled[..., 0],
            scaled[..., 1],
            ordered[..., 2],
            scaled[..., 3],
            scaled[..., 4],
            ordered[..., 5],
        ),
        dim=-1,
    )
    return torch.cat((scaled, gas_spring_forces), dim=-1)
