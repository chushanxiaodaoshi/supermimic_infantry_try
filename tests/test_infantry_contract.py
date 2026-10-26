from __future__ import annotations

import torch

from supermimic.infantry.control import policy_action_to_control
from supermimic.infantry.replay_contract import ReplayContract, extract_replay_constants
from supermimic.infantry.spec import INFANTRY_SPEC


def test_verified_infantry_dimensions() -> None:
    assert INFANTRY_SPEC.num_joints == 14
    assert INFANTRY_SPEC.policy_action_dim == 6
    assert INFANTRY_SPEC.control_dim == 8
    assert INFANTRY_SPEC.state_dim == 41
    assert INFANTRY_SPEC.actor_obs_dim == 25
    assert INFANTRY_SPEC.control_dt == 0.01


def test_mixed_action_semantics_are_explicit() -> None:
    action = torch.arange(1.0, 7.0).repeat(2, 1).requires_grad_()
    offsets = torch.tensor([[0.1, 0.2, 99.0, 0.3, 0.4, -99.0]]).expand(2, -1)
    springs = torch.tensor([[290.0, 310.0]]).expand(2, -1)
    control = policy_action_to_control(
        action,
        default_motor_targets=offsets,
        gas_spring_forces=springs,
        action_scales=torch.tensor([0.5, 0.5, 10.0, 0.5, 0.5, 10.0]),
        action_targets=(
            "Left_rear_joint",
            "Left_front_joint",
            "Left_Wheel_joint",
            "Right_rear_joint",
            "Right_front_joint",
            "Right_Wheel_joint",
        ),
    )
    expected = torch.tensor([1.1, 0.7, 30.0, 2.8, 2.4, 60.0, 290.0, 310.0])
    torch.testing.assert_close(control[0], expected)
    control.sum().backward()
    assert action.grad is not None
    torch.testing.assert_close(action.grad[0], torch.tensor([0.5, 0.5, 10.0, 0.5, 0.5, 10.0]))


def test_replay_contract_is_extracted_without_importing_replay(tmp_path) -> None:
    source = tmp_path / "run_policy.py"
    source.write_text(
        """
import definitely_missing_runtime
import numpy as np
LEG_JOINTS = ("lf", "lr", "rf", "rr")
POLICY_JOINTS = ("lf", "lr", "lw", "rf", "rr", "rw")
ACTION_TARGETS = ("lr", "lf", "lw", "rr", "rf", "rw")
ACTION_SCALES = np.array((0.5, 0.5, 10.0, 0.5, 0.5, 10.0))
POLICY_TORQUE_LIMITS = np.array((28.0, 28.0, 4.0, 28.0, 28.0, 4.0))
def run(action):
    action = np.clip(action, -100.0, 100.0)
    self.data.ctrl[self.gas_ids] = 300.0
""",
        encoding="utf-8",
    )
    constants = extract_replay_constants(source)
    assert constants["ACTION_TARGETS"][:2] == ("lr", "lf")
    contract = ReplayContract.from_replay(source)
    assert contract.observation_dim == 150
    assert contract.action_scales[2] == 10.0
    assert contract.action_clip == (-100.0, 100.0)
    assert contract.gas_spring_targets == (300.0, 300.0)
