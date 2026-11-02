"""Train a teacher-free flat policy through a frozen Infantry World Model."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import yaml
from torch import nn

from supermimic.core.networks import Critic, UnifiedActor, WorldModel
from supermimic.core.spec import TensorState
from supermimic.data.dataset import ChunkDataset, DatasetSource, MixtureDataset
from supermimic.infantry.flat_mdp import (
    FlatRewardWeights,
    actor_observation,
    bounded_policy_action,
    compose_world_model_control,
    flat_reward,
    soft_termination,
)
from supermimic.infantry.replay_contract import ReplayContract, sha256_file
from supermimic.infantry.spec import INFANTRY_SPEC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--world-model", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--iterations", type=int)
    return parser.parse_args()


def _resolve(value: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (REPO / path).resolve() if not path.is_absolute() else path.resolve()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    return value


def load_config(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    cfg = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "config")
    for section in (
        "world_model",
        "controller",
        "actor",
        "critic",
        "training",
        "reward",
    ):
        _mapping(cfg.get(section), section)
    datasets = cfg.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("datasets must be a non-empty list")
    if args.world_model is not None:
        cfg["world_model"]["checkpoint"] = str(args.world_model)
    if args.output is not None:
        cfg["output"] = str(args.output)
    if args.device is not None:
        cfg["device"] = args.device
    if args.iterations is not None:
        cfg["training"]["iterations"] = args.iterations
    for name in ("output", "device"):
        if not cfg.get(name):
            raise ValueError(f"{name} is required")
    if not cfg["world_model"].get("checkpoint"):
        raise ValueError("world_model.checkpoint is required")
    cfg.setdefault("seed", 42)
    return cfg


def build_dataset(cfg: dict[str, Any]) -> MixtureDataset:
    sources = [
        DatasetSource(
            name=str(item["name"]),
            dataset=ChunkDataset(_resolve(item["path"]), seed=int(cfg["seed"]) + index),
            weight=float(item.get("weight", 1.0)),
        )
        for index, item in enumerate(cfg["datasets"])
    ]
    for source in sources:
        missing = {
            "state",
            "observation_history",
            "command",
            "context",
        } - source.dataset.data.keys()
        if missing:
            raise KeyError(f"{source.name} is missing policy fields: {sorted(missing)}")
    return MixtureDataset(sources, seed=int(cfg["seed"]) + 1000)


def load_world_model(path: Path, device: torch.device) -> tuple[WorldModel, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("robot_spec") not in (None, asdict(INFANTRY_SPEC)):
        raise ValueError("World Model checkpoint uses an incompatible robot spec")
    checkpoint_cfg = _mapping(payload.get("config"), "World Model checkpoint config")
    model_cfg = _mapping(checkpoint_cfg.get("model"), "World Model model config")
    hidden = tuple(int(value) for value in model_cfg.get("hidden", (512, 512)))
    model = WorldModel(INFANTRY_SPEC, hidden=hidden).to(device)
    model.load_state_dict(payload["world_model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def critic_features(
    model: WorldModel,
    state: TensorState,
    command: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    return torch.cat((model.codec.invariant_features(state), command, context), dim=-1)


@contextmanager
def freeze_parameters(module: nn.Module) -> Iterator[None]:
    previous = [parameter.requires_grad for parameter in module.parameters()]
    try:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in zip(module.parameters(), previous, strict=True):
            parameter.requires_grad_(flag)


def fit_actor_normalizer(
    actor: UnifiedActor,
    dataset: MixtureDataset,
    *,
    samples: int,
    batch_size: int,
    device: torch.device,
) -> None:
    remaining = samples
    while remaining:
        count = min(remaining, batch_size)
        batch = dataset.sample_windows(
            count,
            1,
            split=0,
            device=device,
            fields=("observation_history",),
        )
        actor.obs_norm.update(batch["observation_history"][:, 0])
        remaining -= count


def _rng_payload(dataset: MixtureDataset) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "dataset": dataset.rng_state(),
    }


def _restore_rng(payload: dict[str, Any], dataset: MixtureDataset) -> None:
    random.setstate(payload["python"])
    torch.set_rng_state(payload["torch"].cpu())
    if payload.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in payload["cuda"]])
    dataset.set_rng_state(payload["dataset"])


def resume_signature(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fields that define one policy optimization run.

    Iteration and logging/checkpoint intervals are intentionally excluded so a
    run can be extended without changing its learning problem.
    """
    training = cfg["training"]
    return {
        "world_model_sha256": cfg["world_model"]["sha256"],
        "datasets": cfg["datasets"],
        "controller_contract": cfg["controller"]["contract"],
        "actor": cfg["actor"],
        "critic": cfg["critic"],
        "reward": cfg["reward"],
        "training": {
            key: training.get(key)
            for key in (
                "batch_size",
                "learning_rate",
                "gamma",
                "gae_lambda",
                "max_grad_norm",
                "support_penalty_weight",
            )
        },
        "seed": cfg["seed"],
    }


def critic_bootstrap_weight(iteration: int, warmup: int, ramp: int) -> float:
    """Delay and smoothly introduce the critic term into the Actor objective."""
    if iteration < 0 or warmup < 0 or ramp < 0:
        raise ValueError("critic bootstrap schedule values must be non-negative")
    completed = iteration + 1
    if completed <= warmup:
        return 0.0
    if ramp == 0:
        return 1.0
    return min((completed - warmup) / ramp, 1.0)


def save_checkpoint(
    path: Path,
    *,
    actor: UnifiedActor,
    critic: Critic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    iteration: int,
    elapsed_seconds: float,
    cfg: dict[str, Any],
    dataset: MixtureDataset,
    world_model_path: Path,
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    payload = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "iteration": iteration,
        "elapsed_seconds": elapsed_seconds,
        "config": cfg,
        "robot_spec": asdict(INFANTRY_SPEC),
        "world_model": {
            "path": str(world_model_path),
            "sha256": sha256_file(world_model_path),
        },
        "resume_signature": resume_signature(cfg),
        "rng": _rng_payload(dataset),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    cfg = load_config(config_path, args)
    seed = int(cfg["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output = _resolve(cfg["output"])
    resume = args.resume.expanduser().resolve() if args.resume else None
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError(f"output must be empty for a new run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    world_model_path = _resolve(cfg["world_model"]["checkpoint"])
    if not world_model_path.is_file():
        raise FileNotFoundError(f"World Model checkpoint: {world_model_path}")
    replay_source = _resolve(cfg["controller"]["replay_source"])
    contract = ReplayContract.from_replay(replay_source)
    dataset = build_dataset(cfg)
    model, world_model_payload = load_world_model(world_model_path, device)

    actor_cfg = cfg["actor"]
    critic_cfg = cfg["critic"]
    training = cfg["training"]
    actor = UnifiedActor(
        INFANTRY_SPEC.actor_obs_dim,
        INFANTRY_SPEC.policy_action_dim,
        history=int(actor_cfg.get("history", 5)),
        hidden=tuple(int(value) for value in actor_cfg.get("hidden", (512, 256, 128))),
    ).to(device)
    critic_dim = INFANTRY_SPEC.invariant_dim + INFANTRY_SPEC.command_dim + INFANTRY_SPEC.context_dim
    critic = Critic(
        critic_dim,
        hidden=tuple(int(value) for value in critic_cfg.get("hidden", (256, 128))),
    ).to(device)
    learning_rate = float(training.get("learning_rate", 2.0e-4))
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate, betas=(0.9, 0.95))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate)
    start_iteration = 0
    elapsed_offset = 0.0

    resolved_cfg = dict(cfg)
    resolved_cfg["output"] = str(output)
    resolved_cfg["world_model"] = {
        **cfg["world_model"],
        "checkpoint": str(world_model_path),
        "sha256": sha256_file(world_model_path),
        "source_iteration": int(world_model_payload.get("iteration", 0)),
    }
    resolved_cfg["controller"] = {
        **cfg["controller"],
        "replay_source": str(replay_source),
        "contract": contract.to_dict(),
    }
    resolved_cfg["datasets"] = [
        {**item, "path": str(_resolve(item["path"]))} for item in cfg["datasets"]
    ]
    resolved_path = output / "resolved_config.yaml"

    if resume is None:
        fit_actor_normalizer(
            actor,
            dataset,
            samples=int(actor_cfg.get("normalizer_samples", 100_000)),
            batch_size=int(training.get("batch_size", 4096)),
            device=device,
        )
        resolved_path.write_text(
            yaml.safe_dump(resolved_cfg, sort_keys=False), encoding="utf-8"
        )
    else:
        payload = torch.load(resume, map_location=device, weights_only=False)
        expected_hash = payload["world_model"]["sha256"]
        if sha256_file(world_model_path) != expected_hash:
            raise ValueError("resume World Model differs from the original run")
        if payload.get("resume_signature") != resume_signature(resolved_cfg):
            raise ValueError(
                "resume checkpoint is incompatible with the current policy config"
            )
        actor.load_state_dict(payload["actor"], strict=True)
        critic.load_state_dict(payload["critic"], strict=True)
        actor_optimizer.load_state_dict(payload["actor_optimizer"])
        critic_optimizer.load_state_dict(payload["critic_optimizer"])
        _restore_rng(payload["rng"], dataset)
        start_iteration = int(payload["iteration"])
        elapsed_offset = float(payload.get("elapsed_seconds", 0.0))

    weights = FlatRewardWeights(**cfg["reward"].get("weights", {}))
    joint_lower = torch.tensor(
        cfg["reward"].get("joint_lower", [-100.0] * 4), device=device
    )
    joint_upper = torch.tensor(
        cfg["reward"].get("joint_upper", [100.0] * 4), device=device
    )
    horizon = int(actor_cfg.get("bptt_horizon", 16))
    action_limit_value = actor_cfg.get("output_limit")
    action_limit = (
        None if action_limit_value is None else float(action_limit_value)
    )
    if action_limit is not None and action_limit <= 0.0:
        raise ValueError("actor.output_limit must be positive")
    bootstrap_warmup = int(critic_cfg.get("bootstrap_warmup_iterations", 0))
    bootstrap_ramp = int(critic_cfg.get("bootstrap_ramp_iterations", 0))
    if bootstrap_warmup < 0 or bootstrap_ramp < 0:
        raise ValueError("critic bootstrap schedule must be non-negative")
    batch_size = int(training.get("batch_size", 4096))
    gamma = float(training.get("gamma", 0.99))
    gae_lambda = float(training.get("gae_lambda", 0.95))
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    iterations = int(training["iterations"])
    critic_enabled = bool(critic_cfg.get("enabled", True))
    if not critic_enabled and critic_bootstrap_weight(
        iterations - 1, bootstrap_warmup, bootstrap_ramp
    ) > 0.0:
        raise ValueError("disabled Critic cannot contribute an Actor bootstrap")
    checkpoint_every = int(training.get("checkpoint_every", 1000))
    log_every = int(training.get("log_every", 50))
    log_path = output / "train_log.jsonl"
    start_time = time.time() - elapsed_offset

    for iteration in range(start_iteration, iterations):
        bootstrap_weight = critic_bootstrap_weight(
            iteration, bootstrap_warmup, bootstrap_ramp
        )
        batch = dataset.sample_windows(
            batch_size,
            1,
            split=0,
            device=device,
            fields=("state", "observation_history", "command", "context"),
        )
        state = model.codec.unflatten(batch["state"][:, 0])
        history = batch["observation_history"][:, 0]
        command = batch["command"][:, 0]
        context = batch["context"][:, 0]
        previous_action = history[:, -1, -INFANTRY_SPEC.policy_action_dim :]
        previous_previous_action = history[:, -2, -INFANTRY_SPEC.policy_action_dim :]
        alive = torch.ones(batch_size, device=device)
        discount = 1.0
        objective = torch.zeros(batch_size, device=device)
        rewards: list[torch.Tensor] = []
        dones: list[torch.Tensor] = []
        critic_inputs: list[torch.Tensor] = []
        support_values: list[torch.Tensor] = []
        action_values: list[torch.Tensor] = []

        with freeze_parameters(critic):
            for _ in range(horizon):
                if critic_enabled:
                    critic_inputs.append(
                        critic_features(model, state, command, context)
                    )
                raw_action = actor(history)
                action = bounded_policy_action(raw_action, action_limit).clamp(
                    *contract.action_clip
                )
                control = compose_world_model_control(action, context, contract)
                next_state = model.step(state, control, context)
                reward, _ = flat_reward(
                    state,
                    next_state,
                    command,
                    action,
                    previous_action,
                    previous_previous_action,
                    control[:, :6],
                    context,
                    joint_lower=joint_lower,
                    joint_upper=joint_upper,
                    weights=weights,
                )
                done = soft_termination(next_state)
                normalized_input = model.in_norm(model.inputs(state, control, context))
                support = normalized_input.square().mean(dim=-1)
                support_weight = float(training.get("support_penalty_weight", 0.0))
                reward = reward - support_weight * support
                objective = objective + discount * alive * reward
                alive = alive * (1.0 - done).clamp(0.0, 1.0)
                observation = actor_observation(next_state, command, action)
                history = torch.cat((history[:, 1:], observation.unsqueeze(1)), dim=1)
                previous_previous_action = previous_action
                previous_action = action
                state = next_state
                discount *= gamma
                rewards.append(reward)
                dones.append(done)
                support_values.append(support)
                action_values.append(action)
            final_critic_input = None
            if critic_enabled or bootstrap_weight > 0.0:
                final_critic_input = critic_features(
                    model, state, command, context
                )
            if bootstrap_weight > 0.0:
                assert final_critic_input is not None
                objective = objective + (
                    bootstrap_weight
                    * discount
                    * alive
                    * critic(final_critic_input)
                )
            actor_loss = -objective.mean()

        if not torch.isfinite(actor_loss):
            raise FloatingPointError(f"non-finite actor loss at iteration {iteration + 1}")
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad = nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
        if not torch.isfinite(actor_grad):
            raise FloatingPointError(f"non-finite actor gradient at iteration {iteration + 1}")
        actor_optimizer.step()

        critic_loss: torch.Tensor | None = None
        critic_grad: torch.Tensor | None = None
        if critic_enabled:
            assert final_critic_input is not None
            with torch.no_grad():
                detached_inputs = [value.detach() for value in critic_inputs]
                critic.norm.update(
                    torch.cat(detached_inputs + [final_critic_input.detach()])
                )
                values = [critic(value) for value in detached_inputs]
                values.append(critic(final_critic_input.detach()))
                target = values[-1]
                targets: list[torch.Tensor] = []
                for step in range(horizon - 1, -1, -1):
                    target = rewards[step].detach() + gamma * (
                        1.0 - dones[step].detach()
                    ) * (
                        (1.0 - gae_lambda) * values[step + 1]
                        + gae_lambda * target
                    )
                    targets.append(target)
                targets.reverse()
            critic_prediction = critic(torch.cat(detached_inputs))
            critic_target = torch.cat(targets)
            critic_loss = (critic_prediction - critic_target).square().mean()
            if not torch.isfinite(critic_loss):
                raise FloatingPointError(
                    f"non-finite critic loss at iteration {iteration + 1}"
                )
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_grad = nn.utils.clip_grad_norm_(
                critic.parameters(), max_grad_norm
            )
            if not torch.isfinite(critic_grad):
                raise FloatingPointError(
                    f"non-finite critic gradient at iteration {iteration + 1}"
                )
            critic_optimizer.step()

        if (iteration + 1) % log_every == 0 or iteration == start_iteration:
            elapsed = time.time() - start_time
            row = {
                "iteration": iteration + 1,
                "elapsed_seconds": elapsed,
                "actor_loss": float(actor_loss.detach()),
                "critic_loss": (
                    None if critic_loss is None else float(critic_loss.detach())
                ),
                "imagined_reward": float(torch.stack(rewards).mean().detach()),
                "survival": float(alive.mean().detach()),
                "mean_abs_action": float(torch.stack(action_values).abs().mean().detach()),
                "wm_input_rms": float(torch.stack(support_values).mean().sqrt().detach()),
                "actor_grad_norm": float(actor_grad.detach()),
                "critic_grad_norm": (
                    None if critic_grad is None else float(critic_grad.detach())
                ),
                "critic_bootstrap_weight": bootstrap_weight,
                "iterations_per_second": (iteration + 1 - start_iteration) / max(elapsed - elapsed_offset, 1.0e-6),
                "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

        if (iteration + 1) % checkpoint_every == 0 or iteration + 1 == iterations:
            save_checkpoint(
                output / f"checkpoint_{iteration + 1:08d}.pt",
                actor=actor,
                critic=critic,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                iteration=iteration + 1,
                elapsed_seconds=time.time() - start_time,
                cfg=resolved_cfg,
                dataset=dataset,
                world_model_path=world_model_path,
            )


if __name__ == "__main__":
    main()
