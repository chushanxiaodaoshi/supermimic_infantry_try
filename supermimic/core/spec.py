"""Robot-independent state and control specifications.

The upstream G1 implementation encodes all dimensions in module constants.  This
module keeps the same six-component floating-base state while moving dimensions
and control semantics into a small immutable specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch

from supermimic.rotations import (
    mat_to_6d,
    mat_to_quat,
    quat_to_mat,
    so3_exp,
    so3_log,
    yaw_mat,
)

TensorState: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]

_CONTROL_TYPES = frozenset({"position", "velocity", "effort", "tendon_force"})


@dataclass(frozen=True)
class RobotSpec:
    """Dimensions and named control contract shared by adapters and learners."""

    name: str
    joint_names: tuple[str, ...]
    policy_action_names: tuple[str, ...]
    control_names: tuple[str, ...]
    control_types: tuple[str, ...]
    context_names: tuple[str, ...]
    actor_obs_dim: int
    critic_obs_dim: int
    command_dim: int
    control_dt: float

    def __post_init__(self) -> None:
        named_groups = {
            "joint_names": self.joint_names,
            "policy_action_names": self.policy_action_names,
            "control_names": self.control_names,
            "context_names": self.context_names,
        }
        for label, names in named_groups.items():
            if len(set(names)) != len(names):
                raise ValueError(f"{label} must contain unique names")
        if len(self.control_names) != len(self.control_types):
            raise ValueError("control_names and control_types must have equal length")
        unknown = set(self.control_types) - _CONTROL_TYPES
        if unknown:
            raise ValueError(f"unsupported control types: {sorted(unknown)}")
        for label, value in (
            ("actor_obs_dim", self.actor_obs_dim),
            ("critic_obs_dim", self.critic_obs_dim),
            ("command_dim", self.command_dim),
        ):
            if value < 0:
                raise ValueError(f"{label} cannot be negative")
        if (
            not self.joint_names
            or not self.policy_action_names
            or not self.control_names
        ):
            raise ValueError("joint, policy-action, and control names cannot be empty")
        if self.control_dt <= 0:
            raise ValueError("control_dt must be positive")

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def policy_action_dim(self) -> int:
        return len(self.policy_action_names)

    @property
    def control_dim(self) -> int:
        return len(self.control_names)

    @property
    def context_dim(self) -> int:
        return len(self.context_names)

    @property
    def state_dim(self) -> int:
        # p(3), quaternion wxyz(4), v(3), w(3), q(n), qd(n)
        return 13 + 2 * self.num_joints

    @property
    def invariant_dim(self) -> int:
        # height, tilt 6D, body-frame linear/angular velocity, q, qd
        return 13 + 2 * self.num_joints

    @property
    def delta_dim(self) -> int:
        # dp, dtheta, dv, dw, dq, dqd
        return 12 + 2 * self.num_joints


class StateCodec:
    """Floating-base state transforms used by all robot-specific world models."""

    def __init__(self, spec: RobotSpec):
        self.spec = spec

    def flatten(self, state: TensorState) -> torch.Tensor:
        p, rotation, linear_velocity, angular_velocity, q, qd = state
        self.validate(state)
        return torch.cat(
            (p, mat_to_quat(rotation), linear_velocity, angular_velocity, q, qd),
            dim=-1,
        )

    def unflatten(self, value: torch.Tensor) -> TensorState:
        if value.shape[-1] != self.spec.state_dim:
            raise ValueError(
                f"expected flattened state width {self.spec.state_dim}, got {value.shape[-1]}"
            )
        n = self.spec.num_joints
        return (
            value[..., 0:3],
            quat_to_mat(value[..., 3:7]),
            value[..., 7:10],
            value[..., 10:13],
            value[..., 13 : 13 + n],
            value[..., 13 + n : 13 + 2 * n],
        )

    def validate(self, state: TensorState) -> None:
        if len(state) != 6:
            raise ValueError("state must contain (p, R, v, w, q, qd)")
        p, rotation, linear_velocity, angular_velocity, q, qd = state
        if (
            p.shape[-1] != 3
            or linear_velocity.shape[-1] != 3
            or angular_velocity.shape[-1] != 3
        ):
            raise ValueError("root position and velocities must have width 3")
        if rotation.shape[-2:] != (3, 3):
            raise ValueError("root rotation must end in shape (3, 3)")
        if q.shape[-1] != self.spec.num_joints or qd.shape[-1] != self.spec.num_joints:
            raise ValueError(f"joint state must have width {self.spec.num_joints}")

    def invariant_features(self, state: TensorState) -> torch.Tensor:
        self.validate(state)
        p, rotation, linear_velocity, angular_velocity, q, qd = state
        heading = yaw_mat(rotation)
        rotation_t = rotation.transpose(-1, -2)
        return torch.cat(
            (
                p[..., 2:3],
                mat_to_6d(heading.transpose(-1, -2) @ rotation),
                (rotation_t @ linear_velocity.unsqueeze(-1)).squeeze(-1),
                (rotation_t @ angular_velocity.unsqueeze(-1)).squeeze(-1),
                q,
                qd,
            ),
            dim=-1,
        )

    def targets(self, state: TensorState, next_state: TensorState) -> torch.Tensor:
        """Frame-local transition targets, matching the upstream G1 formulation."""
        self.validate(state)
        self.validate(next_state)
        p, rotation, linear_velocity, angular_velocity, q, qd = state
        p1, rotation1, linear_velocity1, angular_velocity1, q1, qd1 = next_state
        heading = yaw_mat(rotation)
        rotation_t = rotation.transpose(-1, -2)
        return torch.cat(
            (
                (heading.transpose(-1, -2) @ (p1 - p).unsqueeze(-1)).squeeze(-1),
                so3_log(rotation_t @ rotation1),
                (
                    rotation_t @ (linear_velocity1 - linear_velocity).unsqueeze(-1)
                ).squeeze(-1),
                (
                    rotation_t @ (angular_velocity1 - angular_velocity).unsqueeze(-1)
                ).squeeze(-1),
                q1 - q,
                qd1 - qd,
            ),
            dim=-1,
        )

    def integrate(self, state: TensorState, delta: torch.Tensor) -> TensorState:
        self.validate(state)
        if delta.shape[-1] != self.spec.delta_dim:
            raise ValueError(
                f"expected delta width {self.spec.delta_dim}, got {delta.shape[-1]}"
            )
        p, rotation, linear_velocity, angular_velocity, q, qd = state
        n = self.spec.num_joints
        dp, dtheta, dv, dw, dq, dqd = delta.split((3, 3, 3, 3, n, n), dim=-1)
        heading = yaw_mat(rotation)
        return (
            p + (heading @ dp.unsqueeze(-1)).squeeze(-1),
            rotation @ so3_exp(dtheta),
            linear_velocity + (rotation @ dv.unsqueeze(-1)).squeeze(-1),
            angular_velocity + (rotation @ dw.unsqueeze(-1)).squeeze(-1),
            q + dq,
            qd + dqd,
        )

    def state_error(self, prediction: TensorState, target: TensorState) -> torch.Tensor:
        """Return local prediction errors in the same layout as transition targets."""
        self.validate(prediction)
        self.validate(target)
        p, rotation, linear_velocity, angular_velocity, q, qd = target
        ph, rotation_h, linear_velocity_h, angular_velocity_h, qh, qdh = prediction
        heading = yaw_mat(rotation)
        rotation_t = rotation.transpose(-1, -2)
        return torch.cat(
            (
                (heading.transpose(-1, -2) @ (ph - p).unsqueeze(-1)).squeeze(-1),
                so3_log(rotation_t @ rotation_h),
                (
                    rotation_t @ (linear_velocity_h - linear_velocity).unsqueeze(-1)
                ).squeeze(-1),
                (
                    rotation_t @ (angular_velocity_h - angular_velocity).unsqueeze(-1)
                ).squeeze(-1),
                qh - q,
                qdh - qd,
            ),
            dim=-1,
        )
