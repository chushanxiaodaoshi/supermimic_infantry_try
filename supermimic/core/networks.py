"""Configurable networks for non-G1 robots while preserving SuperMimic's design."""

from __future__ import annotations

import torch
from torch import nn

from .spec import RobotSpec, StateCodec, TensorState


def mlp(
    inp: int,
    hidden: tuple[int, ...] | list[int],
    out: int,
    *,
    activation: type[nn.Module] = nn.ELU,
    layer_norm: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = inp
    for next_width in hidden:
        layers.append(nn.Linear(width, next_width))
        if layer_norm:
            layers.append(nn.LayerNorm(next_width))
        layers.append(activation())
        width = next_width
    layers.append(nn.Linear(width, out))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Streaming normalizer with statistics excluded from autograd."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count = max_count
        self.eps = eps

    @torch.no_grad()
    def update(self, value: torch.Tensor) -> None:
        value = value.reshape(-1, value.shape[-1]).float()
        if value.numel() == 0 or self.count >= self.max_count:
            return
        n = value.shape[0]
        mean = value.mean(dim=0)
        variance = value.var(dim=0, unbiased=False)
        total = self.count + n
        delta = mean - self.mean
        self.mean += delta * n / total
        self.var = (
            self.var * self.count
            + variance * n
            + delta.square() * self.count * n / total
        ) / total
        self.count = total

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.mean) / self.std()

    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var + self.eps**2)


class WorldModel(nn.Module):
    """Deterministic primary dynamics with optional epistemic shadow models."""

    def __init__(
        self,
        spec: RobotSpec,
        hidden: tuple[int, ...] = (512, 512),
        *,
        zero_init_output: bool = True,
        ensemble_size: int = 1,
        termination_head: bool = False,
        termination_hidden: tuple[int, ...] = (128, 64),
    ):
        super().__init__()
        if ensemble_size <= 0:
            raise ValueError("ensemble_size must be positive")
        self.spec = spec
        self.codec = StateCodec(spec)
        self.ensemble_size = ensemble_size
        self.in_dim = spec.invariant_dim + spec.control_dim + spec.context_dim

        def dynamics_net() -> nn.Sequential:
            result = mlp(
                self.in_dim,
                hidden,
                spec.delta_dim,
                activation=nn.SiLU,
                layer_norm=True,
            )
            if zero_init_output:
                output_layer = result[-1]
                assert isinstance(output_layer, nn.Linear)
                nn.init.zeros_(output_layer.weight)
                nn.init.zeros_(output_layer.bias)
            return result

        # Keep ``net`` as the primary-model key for old checkpoint compatibility.
        self.net = dynamics_net()
        self.shadow_nets = nn.ModuleList(
            dynamics_net() for _ in range(ensemble_size - 1)
        )
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(spec.delta_dim, eps=1e-4)
        self.termination_net: nn.Sequential | None = None
        self.termination_norm: RunningNorm | None = None
        if termination_head:
            self.termination_norm = RunningNorm(spec.invariant_dim)
            self.termination_net = mlp(
                spec.invariant_dim,
                termination_hidden,
                1,
                activation=nn.SiLU,
                layer_norm=True,
            )
            output_layer = self.termination_net[-1]
            assert isinstance(output_layer, nn.Linear)
            nn.init.zeros_(output_layer.weight)
            nn.init.constant_(output_layer.bias, -4.0)

    def inputs(
        self,
        state: TensorState,
        applied_control: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if applied_control.shape[-1] != self.spec.control_dim:
            raise ValueError(
                f"expected applied-control width {self.spec.control_dim}, "
                f"got {applied_control.shape[-1]}"
            )
        if context.shape[-1] != self.spec.context_dim:
            raise ValueError(
                f"expected context width {self.spec.context_dim}, got {context.shape[-1]}"
            )
        return torch.cat(
            (self.codec.invariant_features(state), applied_control, context),
            dim=-1,
        )

    def transition_targets(
        self, state: TensorState, next_state: TensorState
    ) -> torch.Tensor:
        return self.codec.targets(state, next_state)

    def step(
        self,
        state: TensorState,
        applied_control: torch.Tensor,
        context: torch.Tensor,
        *,
        member: int = 0,
    ) -> TensorState:
        model_input = self.in_norm(self.inputs(state, applied_control, context))
        normalized_delta = self._member(member)(model_input)
        delta = normalized_delta * self.out_norm.std() + self.out_norm.mean
        return self.codec.integrate(state, delta)

    def _member(self, index: int) -> nn.Module:
        if index == 0:
            return self.net
        if not 0 < index < self.ensemble_size:
            raise IndexError(f"world-model member {index} is out of range")
        return self.shadow_nets[index - 1]

    def ensemble_disagreement(
        self,
        state: TensorState,
        applied_control: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Return dimensionless RMS disagreement of predicted transition deltas."""
        if self.ensemble_size == 1:
            return applied_control.new_zeros(applied_control.shape[:-1])
        model_input = self.in_norm(self.inputs(state, applied_control, context))
        prediction = torch.stack(
            [self._member(index)(model_input) for index in range(self.ensemble_size)]
        )
        return prediction.var(dim=0, unbiased=False).mean(dim=-1).sqrt()

    @property
    def has_termination_head(self) -> bool:
        return self.termination_net is not None

    def termination_logit(self, state: TensorState) -> torch.Tensor:
        if self.termination_net is None or self.termination_norm is None:
            raise RuntimeError("World Model has no termination head")
        features = self.codec.invariant_features(state)
        return self.termination_net(self.termination_norm(features)).squeeze(-1)

    def termination_probability(self, state: TensorState) -> torch.Tensor:
        return self.termination_logit(state).sigmoid()

    @torch.no_grad()
    def initialize_shadows_from_primary(self, noise_std: float = 1.0e-3) -> None:
        """Seed newly added shadow members from a pretrained primary model."""
        if noise_std < 0.0:
            raise ValueError("shadow initialization noise must be non-negative")
        primary = self.net.state_dict()
        for shadow in self.shadow_nets:
            shadow.load_state_dict(primary)
            if noise_std:
                for parameter in shadow.parameters():
                    parameter.add_(noise_std * torch.randn_like(parameter))

    def state_error(self, prediction: TensorState, target: TensorState) -> torch.Tensor:
        return self.codec.state_error(prediction, target)


class UnifiedActor(nn.Module):
    """One actor for teacher-guided and teacher-free objectives.

    ``goal_dim=0`` yields the teacher-free command-conditioned actor.  A
    positive goal width adds the reference/teacher encoder used for behavioral
    cloning or state-trajectory tracking, without changing the decoder path.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        history: int = 5,
        goal_dim: int = 0,
        latent_dim: int = 32,
        hidden: tuple[int, ...] = (256, 128),
        goal_hidden: tuple[int, ...] = (256, 128),
    ):
        super().__init__()
        if obs_dim <= 0 or action_dim <= 0 or history <= 0 or goal_dim < 0:
            raise ValueError("actor dimensions must be positive (goal_dim may be zero)")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.history = history
        self.goal_dim = goal_dim
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_dim) if goal_dim else None
        if goal_dim:
            self.goal_encoder: nn.Module | None = mlp(
                goal_dim, goal_hidden, latent_dim, activation=nn.ELU
            )
            decoder_input = history * obs_dim + latent_dim
        else:
            self.goal_encoder = None
            decoder_input = history * obs_dim
        self.decoder = mlp(decoder_input, hidden, action_dim, activation=nn.ELU)
        with torch.no_grad():
            output_layer = self.decoder[-1]
            assert isinstance(output_layer, nn.Linear)
            output_layer.weight.mul_(0.01)
            output_layer.bias.zero_()

    def forward(
        self,
        observation_history: torch.Tensor,
        goal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected = (self.history, self.obs_dim)
        if observation_history.shape[-2:] != expected:
            raise ValueError(
                f"expected observation history ending in {expected}, "
                f"got {tuple(observation_history.shape[-2:])}"
            )
        features = [self.obs_norm(observation_history).flatten(start_dim=-2)]
        if self.goal_encoder is not None:
            if goal is None or goal.shape[-1] != self.goal_dim:
                raise ValueError(f"goal must have width {self.goal_dim}")
            assert self.goal_norm is not None
            features.insert(0, torch.tanh(self.goal_encoder(self.goal_norm(goal))))
        elif goal is not None:
            raise ValueError("teacher-free actor does not accept a goal tensor")
        return self.decoder(torch.cat(features, dim=-1))


class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden: tuple[int, ...] = (256, 128)):
        super().__init__()
        self.norm = RunningNorm(input_dim)
        self.net = mlp(input_dim, hidden, 1, activation=nn.ELU)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(value)).squeeze(-1)
