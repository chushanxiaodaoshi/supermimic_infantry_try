from __future__ import annotations

import torch

from supermimic.infantry.control import policy_action_to_control
from supermimic.infantry.spec import INFANTRY_SPEC


def test_verified_infantry_dimensions() -> None:
    assert INFANTRY_SPEC.num_joints == 14
    assert INFANTRY_SPEC.policy_action_dim == 6
    assert INFANTRY_SPEC.control_dim == 8
    assert INFANTRY_SPEC.state_dim == 41
    assert INFANTRY_SPEC.actor_obs_dim == 25
    assert INFANTRY_SPEC.control_dt == 0.01


def test_mixed_action_semantics_are_explicit() -> None:
    action = torch.ones(2, 6, requires_grad=True)
    offsets = torch.tensor([[0.1, 0.2, 99.0, 0.3, 0.4, -99.0]]).expand(2, -1)
    springs = torch.tensor([[290.0, 310.0]]).expand(2, -1)
    control = policy_action_to_control(
        action,
        default_motor_targets=offsets,
        gas_spring_forces=springs,
    )
    expected = torch.tensor([0.6, 0.7, 10.0, 0.8, 0.9, 10.0, 290.0, 310.0])
    torch.testing.assert_close(control[0], expected)
    control.sum().backward()
    assert action.grad is not None
    torch.testing.assert_close(
        action.grad[0], torch.tensor([0.5, 0.5, 10.0, 0.5, 0.5, 10.0])
    )
