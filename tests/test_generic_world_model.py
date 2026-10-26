from __future__ import annotations

from typing import cast

import pytest
import torch

from supermimic.core.bptt import imagined_actor_loss
from supermimic.core.networks import UnifiedActor, WorldModel
from supermimic.core.spec import StateCodec, TensorState
from supermimic.infantry.control import policy_action_to_control
from supermimic.infantry.spec import INFANTRY_SPEC


def identity_state(batch: int) -> TensorState:
    n = INFANTRY_SPEC.num_joints
    p = torch.zeros(batch, 3)
    rotation = torch.eye(3).expand(batch, -1, -1).clone()
    v = torch.zeros(batch, 3)
    w = torch.zeros(batch, 3)
    q = torch.zeros(batch, n)
    qd = torch.zeros(batch, n)
    return p, rotation, v, w, q, qd


def test_state_codec_round_trip() -> None:
    codec = StateCodec(INFANTRY_SPEC)
    state = identity_state(3)
    flattened = codec.flatten(state)
    restored = codec.unflatten(flattened)
    assert flattened.shape == (3, INFANTRY_SPEC.state_dim)
    for before, after in zip(state, restored, strict=True):
        torch.testing.assert_close(before, after)


def test_world_model_action_gradient_matches_finite_difference() -> None:
    torch.manual_seed(7)
    spec = INFANTRY_SPEC
    model = WorldModel(spec, hidden=(32, 32), zero_init_output=False).double()
    state = cast(TensorState, tuple(value.double() for value in identity_state(1)))
    context = torch.zeros(1, spec.context_dim, dtype=torch.double)
    control = torch.randn(1, spec.control_dim, dtype=torch.double, requires_grad=True)

    def scalar(value: torch.Tensor) -> torch.Tensor:
        prediction = model.step(state, value, context)
        return model.codec.flatten(prediction).square().sum()

    automatic = torch.autograd.grad(scalar(control), control)[0]
    epsilon = 1.0e-6
    finite = torch.zeros_like(control)
    for index in range(spec.control_dim):
        plus = control.detach().clone()
        minus = control.detach().clone()
        plus[0, index] += epsilon
        minus[0, index] -= epsilon
        finite[0, index] = (scalar(plus) - scalar(minus)) / (2 * epsilon)
    assert torch.isfinite(automatic).all()
    assert automatic.abs().max() > 0
    torch.testing.assert_close(automatic, finite, rtol=2.0e-4, atol=2.0e-6)


@pytest.mark.parametrize("detach_dynamics", [False, True])
def test_bptt_updates_actor_without_world_model_parameter_gradients(
    detach_dynamics: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(11)
    spec = INFANTRY_SPEC
    actor = UnifiedActor(
        spec.actor_obs_dim, spec.policy_action_dim, history=2, hidden=(32,)
    )
    model = WorldModel(spec, hidden=(32,), zero_init_output=False)
    if detach_dynamics:
        original_step = model.step

        def detached_step(
            state: TensorState, control: torch.Tensor, context: torch.Tensor
        ) -> TensorState:
            return cast(
                TensorState,
                tuple(x.detach() for x in original_step(state, control, context)),
            )

        monkeypatch.setattr(model, "step", detached_step)
    state = identity_state(4)
    history = torch.zeros(4, 2, spec.actor_obs_dim)
    context = torch.zeros(4, spec.context_dim)
    offsets = torch.zeros(4, spec.policy_action_dim)
    springs = torch.full((4, 2), 300.0)

    loss, diagnostics = imagined_actor_loss(
        actor=actor,
        world_model=model,
        initial_state=state,
        observation_history=history,
        context=context,
        control_fn=lambda action: policy_action_to_control(
            action,
            default_motor_targets=offsets,
            gas_spring_forces=springs,
            action_scales=torch.tensor(
                [0.5, 0.5, 10.0, 0.5, 0.5, 10.0], device=action.device
            ),
        ),
        observe_fn=lambda next_state, action: torch.cat(
            (
                next_state[3],
                -next_state[1][..., 2, :],
                torch.zeros(4, 3),
                next_state[4][..., [6, 0, 10, 3]],
                next_state[5][..., [6, 0, 2, 10, 3, 5]],
                action,
            ),
            dim=-1,
        ),
        reward_fn=lambda next_state, action, control: (
            -next_state[0].square().sum(dim=-1),
            torch.zeros(action.shape[0]),
        ),
        horizon=3,
        gamma=0.99,
    )
    if detach_dynamics:
        assert not loss.requires_grad
        assert all(parameter.grad is None for parameter in actor.parameters())
        return
    loss.backward()
    actor_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in actor.parameters()
        if parameter.grad is not None
    )
    assert actor_grad > 0
    assert all(
        torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None
    )
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(parameter.requires_grad for parameter in model.parameters())
    before = [p.detach().clone() for p in actor.parameters()]
    torch.optim.SGD(actor.parameters(), lr=0.01).step()
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, actor.parameters(), strict=True)
    )
    assert torch.isfinite(loss)
    assert set(diagnostics) == {"imagined_reward", "survival", "mean_abs_action"}
