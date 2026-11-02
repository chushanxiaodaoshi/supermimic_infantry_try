from __future__ import annotations

import torch

from supermimic.core.spec import StateCodec
from supermimic.infantry.flat_mdp import (
    actor_observation,
    compose_world_model_control,
    estimated_actuator_force,
    flat_reward,
    soft_termination,
)
from supermimic.infantry.replay_contract import ReplayContract
from supermimic.infantry.spec import INFANTRY_SPEC


def _contract() -> ReplayContract:
    return ReplayContract(
        source="test",
        source_sha256="0" * 64,
        leg_joints=(
            "Left_front_joint",
            "Left_rear_joint",
            "Right_front_joint",
            "Right_rear_joint",
        ),
        policy_joints=INFANTRY_SPEC.policy_action_names,
        action_targets=INFANTRY_SPEC.policy_action_names,
        action_scales=(0.5, 0.5, 10.0, 0.5, 0.5, 10.0),
        torque_limits=(28.0, 28.0, 4.0, 28.0, 28.0, 4.0),
        action_clip=(-100.0, 100.0),
        gas_spring_targets=(300.0, 300.0),
    )


def _context(batch: int = 2) -> torch.Tensor:
    value = torch.zeros(batch, INFANTRY_SPEC.context_dim)
    names = INFANTRY_SPEC.context_names
    for index, name in enumerate(INFANTRY_SPEC.policy_action_names):
        value[:, names.index(f"{name}:gain")] = 2.0
        value[:, names.index(f"{name}:position_bias")] = -2.0
        value[:, names.index(f"{name}:velocity_bias")] = -0.1
        value[:, names.index(f"{name}:force_min")] = -5.0
        value[:, names.index(f"{name}:force_max")] = 5.0
    value[:, names.index("left_gas_spring_force")] = 290.0
    value[:, names.index("right_gas_spring_force")] = 310.0
    return value


def _state(batch: int = 2):
    flat = torch.zeros(batch, INFANTRY_SPEC.state_dim)
    flat[:, 2] = 0.3
    flat[:, 3] = 1.0
    return StateCodec(INFANTRY_SPEC).unflatten(flat)


def test_tendons_are_context_physics_not_policy_actions() -> None:
    action = torch.ones(2, 6, requires_grad=True)
    context = _context()
    context.requires_grad_(True)
    control = compose_world_model_control(action, context, _contract())
    assert control.shape == (2, 8)
    torch.testing.assert_close(control[:, -2:], torch.tensor([[290.0, 310.0]] * 2))
    control.sum().backward()
    assert action.grad is not None and torch.isfinite(action.grad).all()
    # The explicit detach prevents a policy objective from treating tendon force
    # as a controllable dimension.
    assert context.grad is None


def test_observation_and_affine_actuator_force_match_contract() -> None:
    state = _state()
    observation = actor_observation(
        state, torch.zeros(2, 3), torch.zeros(2, 6)
    )
    assert observation.shape == (2, 25)
    control = torch.ones(2, 6)
    force = estimated_actuator_force(state, control, _context())
    torch.testing.assert_close(force, torch.full_like(force, 2.0))


def test_flat_reward_backpropagates_through_predicted_state_and_action() -> None:
    codec = StateCodec(INFANTRY_SPEC)
    previous = torch.zeros(2, INFANTRY_SPEC.state_dim)
    previous[:, 2] = 0.3
    previous[:, 3] = 1.0
    current = previous.clone().requires_grad_()
    current_state = codec.unflatten(current)
    action = torch.zeros(2, 6, requires_grad=True)
    reward, terms = flat_reward(
        codec.unflatten(previous),
        current_state,
        torch.tensor([[1.0, 0.2, 0.13]] * 2),
        action,
        torch.zeros_like(action),
        torch.zeros_like(action),
        torch.zeros(2, 6),
        _context(),
        joint_lower=torch.full((4,), -100.0),
        joint_upper=torch.full((4,), 100.0),
    )
    assert set(terms) == {
        "tracking_lin_vel",
        "tracking_lin_vel_enhance",
        "tracking_ang_vel",
        "nominal_state",
        "leg_fore_aft",
        "ang_vel_xy",
        "orientation",
        "dof_vel",
        "dof_acc",
        "torques",
        "action_rate",
        "action_smooth",
        "dof_pos_limits",
    }
    (-reward.mean()).backward()
    assert current.grad is not None and torch.isfinite(current.grad).all()
    assert action.grad is not None and torch.isfinite(action.grad).all()


def test_soft_termination_separates_upright_and_fallen_states() -> None:
    upright = _state(1)
    fallen_flat = StateCodec(INFANTRY_SPEC).flatten(upright)
    fallen_flat[:, 2] = 0.02
    fallen = StateCodec(INFANTRY_SPEC).unflatten(fallen_flat)
    assert soft_termination(upright).item() < 0.01
    assert soft_termination(fallen).item() > 0.99
