"""Differentiable flat-ground MDP matching the mjlab wheel-leg task.

The policy controls four leg position servos and two wheel velocity servos.
The two gas-spring tendon forces remain episode-level robot physics and are
copied from the World Model context; they are never policy outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from supermimic.core.spec import StateCodec, TensorState
from supermimic.infantry.control import policy_action_to_control
from supermimic.infantry.replay_contract import ReplayContract
from supermimic.infantry.spec import (
    INFANTRY_SPEC,
    POLICY_ACTION_NAMES,
)

_CODEC = StateCodec(INFANTRY_SPEC)
_JOINT_INDEX = {name: index for index, name in enumerate(INFANTRY_SPEC.joint_names)}
_POLICY_JOINT_IDS = tuple(_JOINT_INDEX[name] for name in POLICY_ACTION_NAMES)
_LEG_ACTION_IDS = (0, 1, 3, 4)


def _context_indices(suffix: str) -> tuple[int, ...]:
    names = INFANTRY_SPEC.context_names
    return tuple(names.index(f"{name}:{suffix}") for name in POLICY_ACTION_NAMES)


_GAIN_IDS = _context_indices("gain")
_POSITION_BIAS_IDS = _context_indices("position_bias")
_VELOCITY_BIAS_IDS = _context_indices("velocity_bias")
_FORCE_MIN_IDS = _context_indices("force_min")
_FORCE_MAX_IDS = _context_indices("force_max")
_SPRING_IDS = (
    INFANTRY_SPEC.context_names.index("left_gas_spring_force"),
    INFANTRY_SPEC.context_names.index("right_gas_spring_force"),
)


@dataclass(frozen=True)
class FlatRewardWeights:
    """Saved mjlab flat-task reward weights."""

    tracking_lin_vel: float = 1.0
    tracking_lin_vel_enhance: float = 1.0
    tracking_ang_vel: float = 1.0
    nominal_state: float = -1.0
    leg_fore_aft: float = -2.0
    ang_vel_xy: float = -0.2
    orientation: float = -100.0
    dof_vel: float = -1.0e-4
    dof_acc: float = -1.0e-6
    torques: float = -1.0e-4
    action_rate: float = -0.01
    action_smooth: float = -0.01
    dof_pos_limits: float = -1.0


DEFAULT_FLAT_REWARD_WEIGHTS = FlatRewardWeights()


def bounded_policy_action(
    raw_action: torch.Tensor, limit: float | None
) -> torch.Tensor:
    """Smoothly bound a policy action while preserving small-signal behavior."""
    if limit is None:
        return raw_action
    if limit <= 0.0:
        raise ValueError("policy action limit must be positive")
    return torch.tanh(raw_action / limit) * limit


def compose_world_model_control(
    action: torch.Tensor,
    context: torch.Tensor,
    contract: ReplayContract,
) -> torch.Tensor:
    """Map six policy actions to the eight-value recorded WM input.

    The final two values are copied from physics context and detached.  This
    makes the policy/controller boundary explicit while retaining compatibility
    with already-trained eight-input World Model checkpoints.
    """
    spring_ids = torch.as_tensor(_SPRING_IDS, device=context.device)
    springs = context.index_select(-1, spring_ids).detach()
    zeros = torch.zeros_like(action)
    return policy_action_to_control(
        action.clamp(*contract.action_clip),
        default_motor_targets=zeros,
        gas_spring_forces=springs,
        action_scales=action.new_tensor(contract.action_scales),
        action_targets=contract.action_targets,
    )


def actor_observation(
    state: TensorState,
    command: torch.Tensor,
    previous_action: torch.Tensor,
) -> torch.Tensor:
    """Build the noise-free 25-D observation used by the mjlab flat Actor."""
    _, rotation, _, angular_velocity, q, qd = state
    rotation_t = rotation.transpose(-1, -2)
    body_angular_velocity = (
        rotation_t @ angular_velocity.unsqueeze(-1)
    ).squeeze(-1)
    gravity = state[0].new_tensor((0.0, 0.0, -1.0)).expand_as(state[0])
    projected_gravity = (rotation_t @ gravity.unsqueeze(-1)).squeeze(-1)
    policy_ids = torch.as_tensor(_POLICY_JOINT_IDS, device=q.device)
    leg_ids = policy_ids[torch.as_tensor(_LEG_ACTION_IDS, device=q.device)]
    return torch.cat(
        (
            body_angular_velocity.clamp(-20.0, 20.0) * 0.25,
            projected_gravity,
            command * command.new_tensor((3.0, 0.25, 5.0)),
            q.index_select(-1, leg_ids),
            qd.index_select(-1, policy_ids).clamp(-200.0, 200.0) * 0.05,
            previous_action.clamp(-10.0, 10.0),
        ),
        dim=-1,
    )


def estimated_actuator_force(
    state: TensorState,
    motor_control: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    """Reproduce MuJoCo's affine actuator force before transmission.

    ``force = gain * ctrl + bias_pos * q + bias_vel * qd``, followed by
    actuator force clipping.  All coefficients are recorded in the 73-D
    context, so this is preferable to adding a learned force head.
    """
    q, qd = state[4], state[5]
    ids = torch.as_tensor(_POLICY_JOINT_IDS, device=q.device)
    q = q.index_select(-1, ids)
    qd = qd.index_select(-1, ids)
    gain = context[..., list(_GAIN_IDS)]
    position_bias = context[..., list(_POSITION_BIAS_IDS)]
    velocity_bias = context[..., list(_VELOCITY_BIAS_IDS)]
    force = gain * motor_control + position_bias * q + velocity_bias * qd
    minimum = context[..., list(_FORCE_MIN_IDS)]
    maximum = context[..., list(_FORCE_MAX_IDS)]
    return torch.maximum(torch.minimum(force, maximum), minimum)


def _virtual_leg_angles(q: torch.Tensor) -> torch.Tensor:
    """Closed-chain virtual-leg angles used by the saved mjlab rewards."""
    ids = torch.as_tensor(_POLICY_JOINT_IDS, device=q.device)
    joint_pos = q.index_select(-1, ids)
    theta_1 = torch.stack((joint_pos[..., 0], -joint_pos[..., 3]), dim=-1)
    theta_2 = torch.stack(
        (joint_pos[..., 1] + math.pi / 2, -joint_pos[..., 4] + math.pi / 2),
        dim=-1,
    )
    end_x = 0.175 * torch.cos(theta_1) + 0.208 * torch.cos(theta_1 + theta_2)
    end_y = 0.175 * torch.sin(theta_1) + 0.208 * torch.sin(theta_1 + theta_2)
    return torch.atan2(end_y, end_x) - math.pi / 2


def flat_reward_terms(
    previous_state: TensorState,
    state: TensorState,
    command: torch.Tensor,
    action: torch.Tensor,
    previous_action: torch.Tensor,
    previous_previous_action: torch.Tensor,
    motor_control: torch.Tensor,
    context: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return unclipped mjlab flat reward terms before term weights and dt."""
    _, rotation, linear_velocity, angular_velocity, q, qd = state
    body_linear_velocity = (
        rotation.transpose(-1, -2) @ linear_velocity.unsqueeze(-1)
    ).squeeze(-1)
    body_angular_velocity = (
        rotation.transpose(-1, -2) @ angular_velocity.unsqueeze(-1)
    ).squeeze(-1)
    gravity = q.new_tensor((0.0, 0.0, -1.0)).expand(q.shape[0], -1)
    projected_gravity = (
        rotation.transpose(-1, -2) @ gravity.unsqueeze(-1)
    ).squeeze(-1)
    forward_error = (command[..., 0] - body_linear_velocity[..., 0]).square()
    # Replay command layout is (forward velocity, yaw velocity, body height),
    # unlike mjlab's older four-value manager command.
    yaw_error = (command[..., 1] - body_angular_velocity[..., 2]).square()
    leg_angles = _virtual_leg_angles(q)
    policy_ids = torch.as_tensor(_POLICY_JOINT_IDS, device=q.device)
    leg_joint_ids = policy_ids[torch.as_tensor(_LEG_ACTION_IDS, device=q.device)]
    leg_qd = qd.index_select(-1, leg_joint_ids)
    previous_qd = previous_state[5].index_select(-1, policy_ids)
    policy_qd = qd.index_select(-1, policy_ids)
    qdd = (policy_qd - previous_qd) / INFANTRY_SPEC.control_dt
    actuator_force = estimated_actuator_force(state, motor_control, context)

    # Joint limits are static robot properties and are supplied by the trainer.
    # This placeholder is replaced in ``flat_reward`` when limits are provided.
    zeros = q[..., 0] * 0.0
    return {
        "tracking_lin_vel": torch.exp(-forward_error / 0.25),
        "tracking_lin_vel_enhance": torch.exp(-forward_error / 2.5) - 1.0,
        "tracking_ang_vel": torch.exp(-yaw_error / 0.25),
        "nominal_state": (leg_angles[..., 0] - leg_angles[..., 1]).square(),
        "leg_fore_aft": leg_angles.square().mean(dim=-1),
        "ang_vel_xy": body_angular_velocity[..., :2].square().sum(dim=-1),
        "orientation": projected_gravity[..., :2].square().sum(dim=-1),
        "dof_vel": leg_qd.square().sum(dim=-1),
        "dof_acc": qdd.square().sum(dim=-1),
        "torques": actuator_force.square().sum(dim=-1),
        "action_rate": (action - previous_action).square().sum(dim=-1),
        "action_smooth": (
            action[..., _LEG_ACTION_IDS]
            - 2.0 * previous_action[..., _LEG_ACTION_IDS]
            + previous_previous_action[..., _LEG_ACTION_IDS]
        ).square().sum(dim=-1),
        "dof_pos_limits": zeros,
    }


def flat_reward(
    previous_state: TensorState,
    state: TensorState,
    command: torch.Tensor,
    action: torch.Tensor,
    previous_action: torch.Tensor,
    previous_previous_action: torch.Tensor,
    motor_control: torch.Tensor,
    context: torch.Tensor,
    *,
    joint_lower: torch.Tensor,
    joint_upper: torch.Tensor,
    weights: FlatRewardWeights = DEFAULT_FLAT_REWARD_WEIGHTS,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the weighted reward exactly once per 10-ms control step."""
    terms = flat_reward_terms(
        previous_state,
        state,
        command,
        action,
        previous_action,
        previous_previous_action,
        motor_control,
        context,
    )
    q = state[4]
    ids = torch.as_tensor(_POLICY_JOINT_IDS, device=q.device)
    leg_q = q.index_select(-1, ids)[..., _LEG_ACTION_IDS]
    terms["dof_pos_limits"] = (
        torch.relu(leg_q - joint_upper) + torch.relu(joint_lower - leg_q)
    ).sum(dim=-1)
    total = torch.zeros_like(next(iter(terms.values())))
    for name, value in terms.items():
        total = total + getattr(weights, name) * value
    return total * INFANTRY_SPEC.control_dt, terms


def soft_termination(
    state: TensorState,
    *,
    minimum_height: float = 0.08,
    maximum_tilt_radians: float = math.radians(60.0),
    height_temperature: float = 0.01,
    tilt_temperature: float = 0.04,
) -> torch.Tensor:
    """Differentiable state-only proxy for mjlab fall/contact termination."""
    position, rotation = state[0], state[1]
    tilt_cosine = rotation[..., 2, 2]
    limit_cosine = math.cos(maximum_tilt_radians)
    height_failure = torch.sigmoid(
        (minimum_height - position[..., 2]) / height_temperature
    )
    tilt_failure = torch.sigmoid(
        (limit_cosine - tilt_cosine) / tilt_temperature
    )
    return 1.0 - (1.0 - height_failure) * (1.0 - tilt_failure)


def unflatten_state(value: torch.Tensor) -> TensorState:
    return _CODEC.unflatten(value)
