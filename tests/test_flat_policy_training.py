from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from supermimic.core.networks import WorldModel
from supermimic.data.chunks import ChunkWriter
from supermimic.infantry.spec import INFANTRY_SPEC
from tools import evaluate_flat_policy_isaaclab, train_flat_policy


def _write_replay(path: Path) -> None:
    path.write_text(
        """
import numpy as np
LEG_JOINTS = ("Left_front_joint", "Left_rear_joint", "Right_front_joint", "Right_rear_joint")
POLICY_JOINTS = ("Left_front_joint", "Left_rear_joint", "Left_Wheel_joint", "Right_front_joint", "Right_rear_joint", "Right_Wheel_joint")
ACTION_TARGETS = ("Left_front_joint", "Left_rear_joint", "Left_Wheel_joint", "Right_front_joint", "Right_rear_joint", "Right_Wheel_joint")
ACTION_SCALES = np.array((0.5, 0.5, 10.0, 0.5, 0.5, 10.0))
POLICY_TORQUE_LIMITS = np.array((28.0, 28.0, 4.0, 28.0, 28.0, 4.0))
ACTION_CLIP = (-100.0, 100.0)
GAS_SPRING_TARGET = 300.0
""",
        encoding="utf-8",
    )


def _write_dataset(path: Path) -> None:
    writer = ChunkWriter(path, INFANTRY_SPEC, chunk_steps=20, split_seed=0)
    for step in range(20):
        state = torch.zeros(1, INFANTRY_SPEC.state_dim)
        state[:, 2] = 0.3
        state[:, 3] = 1.0
        writer.append(
            {
                "state": state,
                "observation_history": torch.zeros(
                    1, 5, INFANTRY_SPEC.actor_obs_dim
                ),
                "command": torch.tensor([[1.0, 0.0, 0.0]]),
                "context": torch.zeros(1, INFANTRY_SPEC.context_dim),
                "applied_control": torch.zeros(1, INFANTRY_SPEC.control_dim),
                "timestamp": torch.tensor([step * INFANTRY_SPEC.control_dt]),
                "run_id": torch.tensor([0]),
                "episode_id": torch.tensor([0]),
            }
        )
    writer.close()


def _write_world_model(path: Path) -> None:
    model = WorldModel(INFANTRY_SPEC, hidden=(16,))
    torch.save(
        {
            "world_model": model.state_dict(),
            "robot_spec": asdict(INFANTRY_SPEC),
            "iteration": 3,
            "config": {"model": {"hidden": [16]}},
        },
        path,
    )


def test_flat_policy_training_and_resume(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "dataset"
    replay = tmp_path / "run_policy.py"
    world_model = tmp_path / "world_model.pt"
    output = tmp_path / "policy"
    _write_dataset(dataset)
    _write_replay(replay)
    _write_world_model(world_model)
    config = {
        "seed": 7,
        "device": "cpu",
        "output": str(output),
        "world_model": {"checkpoint": str(world_model)},
        "controller": {"replay_source": str(replay)},
        "datasets": [{"name": "flat", "path": str(dataset), "weight": 1.0}],
        "actor": {
            "hidden": [16],
            "history": 5,
            "bptt_horizon": 2,
            "output_limit": 10.0,
            "normalizer_samples": 8,
        },
        "critic": {
            "hidden": [16],
            "bootstrap_warmup_iterations": 1,
            "bootstrap_ramp_iterations": 1,
        },
        "training": {
            "iterations": 1,
            "batch_size": 4,
            "learning_rate": 0.0002,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "max_grad_norm": 1.0,
            "checkpoint_every": 1,
            "log_every": 1,
            "support_penalty_weight": 0.0,
        },
        "reward": {
            "joint_lower": [-100.0] * 4,
            "joint_upper": [100.0] * 4,
            "weights": {},
        },
    }
    config_path = tmp_path / "policy.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["train_flat_policy.py", "--config", str(config_path)]
    )
    train_flat_policy.main()
    first = output / "checkpoint_00000001.pt"
    payload = torch.load(first, map_location="cpu", weights_only=False)
    assert payload["iteration"] == 1
    assert payload["actor_optimizer"]["state"]
    assert payload["critic_optimizer"]["state"]
    assert payload["world_model"]["sha256"]
    assert payload["resume_signature"] == train_flat_policy.resume_signature(
        payload["config"]
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_flat_policy.py",
            "--config",
            str(config_path),
            "--resume",
            str(first),
            "--iterations",
            "2",
        ],
    )
    train_flat_policy.main()
    resumed = torch.load(
        output / "checkpoint_00000002.pt", map_location="cpu", weights_only=False
    )
    assert resumed["iteration"] == 2
    assert resumed["actor_optimizer"]["state"]

    actor, actor_payload = evaluate_flat_policy_isaaclab.load_actor(
        output / "checkpoint_00000002.pt", torch.device("cpu")
    )
    assert actor_payload["iteration"] == 2
    with torch.inference_mode():
        action = actor(torch.zeros(3, 5, INFANTRY_SPEC.actor_obs_dim))
    assert action.shape == (3, INFANTRY_SPEC.policy_action_dim)

    next_world_model = tmp_path / "next_world_model.pt"
    _write_world_model(next_world_model)
    warm_output = tmp_path / "policy_warm"
    warm_config = yaml.safe_load(yaml.safe_dump(config))
    warm_config["output"] = str(warm_output)
    warm_config["world_model"]["checkpoint"] = str(next_world_model)
    warm_config["training"]["iterations"] = 1
    warm_config["training"]["learning_rate"] = 0.0
    warm_path = tmp_path / "policy_warm.yaml"
    warm_path.write_text(yaml.safe_dump(warm_config), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_flat_policy.py",
            "--config",
            str(warm_path),
            "--init-from",
            str(output / "checkpoint_00000002.pt"),
        ],
    )
    train_flat_policy.main()
    warm = torch.load(
        warm_output / "checkpoint_00000001.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert warm["config"]["initialization"]["source_iteration"] == 2
    assert warm["config"]["initialization"]["optimizer"] == "reset"
    assert warm["world_model"]["sha256"] != resumed["world_model"]["sha256"]
    for name, value in resumed["actor"].items():
        torch.testing.assert_close(warm["actor"][name], value)


def test_critic_bootstrap_schedule_delays_actor_bootstrap() -> None:
    assert train_flat_policy.critic_bootstrap_weight(0, 2, 2) == 0.0
    assert train_flat_policy.critic_bootstrap_weight(1, 2, 2) == 0.0
    assert train_flat_policy.critic_bootstrap_weight(2, 2, 2) == 0.5
    assert train_flat_policy.critic_bootstrap_weight(3, 2, 2) == 1.0
    assert train_flat_policy.critic_bootstrap_weight(100, 0, 0) == 1.0


def test_epistemic_trust_and_learned_termination_bound_imagination() -> None:
    torch.manual_seed(19)
    model = WorldModel(
        INFANTRY_SPEC,
        hidden=(16,),
        zero_init_output=False,
        ensemble_size=3,
        termination_head=True,
        termination_hidden=(8,),
    )
    model.initialize_shadows_from_primary(noise_std=0.02)
    flat_state = torch.zeros(4, INFANTRY_SPEC.state_dim)
    flat_state[:, 2] = 0.3
    flat_state[:, 3] = 1.0
    state = model.codec.unflatten(flat_state)
    control = torch.randn(4, INFANTRY_SPEC.control_dim, requires_grad=True)
    context = torch.zeros(4, INFANTRY_SPEC.context_dim)
    trust, disagreement = train_flat_policy.epistemic_trust(
        model,
        state,
        control,
        context,
        threshold=0.15,
        temperature=0.03,
    )
    assert trust.shape == disagreement.shape == (4,)
    assert ((trust > 0.0) & (trust < 1.0)).all()
    assert (disagreement > 0.0).all()
    disagreement.mean().backward()
    assert control.grad is not None
    assert torch.isfinite(control.grad).all()

    physical = train_flat_policy.soft_termination(state)
    conservative = train_flat_policy.conservative_termination(model, state)
    assert (conservative >= physical).all()
    assert (conservative >= model.termination_probability(state)).all()


def test_evaluation_asset_config_resolves_relative_paths(tmp_path) -> None:
    config_dir = tmp_path / "configs"
    asset_dir = tmp_path / "assets"
    config_dir.mkdir()
    asset_dir.mkdir()
    for name in ("run_policy.py", "scene.xml", "robot.usda"):
        (asset_dir / name).write_text("placeholder", encoding="utf-8")
    path = config_dir / "eval.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "asset": {
                    "replay_source": "../assets/run_policy.py",
                    "mjcf": "../assets/scene.xml",
                    "usd": "../assets/robot.usda",
                },
                "commands": {
                    "lin_vel_x": [-1.0, 1.0],
                    "ang_vel_z": [0.0, 0.0],
                    "height": [0.1, 0.2],
                },
            }
        ),
        encoding="utf-8",
    )
    config = evaluate_flat_policy_isaaclab.load_asset_config(path)
    assert config["asset"]["usd"] == (asset_dir / "robot.usda").resolve()
