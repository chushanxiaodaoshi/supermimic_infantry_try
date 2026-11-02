from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest
import yaml

from tools import collect_isaaclab_student, run_flat_online_cycle


def _base_actor_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "seed": 1,
                "device": "cpu",
                "output": "unused",
                "world_model": {"checkpoint": "unused.pt"},
                "controller": {"replay_source": "run_policy.py"},
                "datasets": [{"name": "old", "path": "old", "weight": 1.0}],
                "actor": {
                    "hidden": [16],
                    "history": 5,
                    "bptt_horizon": 32,
                    "output_limit": 10.0,
                },
                "critic": {"enabled": False},
                "training": {
                    "iterations": 999,
                    "batch_size": 8,
                    "checkpoint_every": 10,
                    "log_every": 5,
                },
                "reward": {"weights": {}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _online_config(tmp_path: Path) -> tuple[Path, Path, Path]:
    actor, world_model = tmp_path / "actor.pt", tmp_path / "world_model.pt"
    actor.touch()
    world_model.touch()
    base_actor = tmp_path / "base_actor.yaml"
    _base_actor_config(base_actor)
    config = {
        "seed": 4,
        "device": "cpu",
        "output": str(tmp_path / "online"),
        "cycles": 2,
        "asset_config": "asset.yaml",
        "initial_actor": str(actor),
        "initial_world_model": str(world_model),
        "teacher_datasets": [
            {"name": "teacher", "path": "teacher_data", "weight": 3.0}
        ],
        "student_collection": {
            "steps": 40,
            "max_episode_steps": 10,
            "chunk_steps": 8,
            "world_model_weight": 1.0,
            "actor_start_weight": 2.0,
        },
        "world_model": {
            "hidden": [16],
            "training": {
                "iterations": 3,
                "batch_size": 4,
                "window": 2,
                "checkpoint_every": 3,
            },
            "validation": {"horizons": [1, 2]},
        },
        "actor": {
            "base_config": str(base_actor),
            "iterations": 5,
            "batch_size": 4,
            "checkpoint_every": 5,
            "log_every": 1,
        },
        "evaluation": {"steps": 20, "vx": 1.0, "yaw": 0.0, "height": 0.2},
    }
    path = tmp_path / "online.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path, actor, world_model


def test_student_collector_rejects_invalid_ranges() -> None:
    args = argparse.Namespace(
        steps=0,
        max_episode_steps=10,
        chunk_steps=8,
        seed=0,
        split_seed=0,
        action_noise_std=0.0,
    )
    with pytest.raises(ValueError, match="steps"):
        collect_isaaclab_student.validate_args(args)
    args.steps = 1
    args.action_noise_std = -0.1
    with pytest.raises(ValueError, match="action-noise"):
        collect_isaaclab_student.validate_args(args)


def test_online_dry_run_builds_accumulating_mixtures(tmp_path, monkeypatch) -> None:
    config_path, actor, _ = _online_config(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_flat_online_cycle.py",
            "--config",
            str(config_path),
            "--actor",
            str(actor),
            "--dry-run",
        ],
    )
    run_flat_online_cycle.main()

    root = tmp_path / "online"
    manifest = json.loads((root / "online_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "planned"
    assert len(manifest["cycles"]) == 2
    for index, row in enumerate(manifest["cycles"], start=1):
        assert len(row["commands"]) == 4
        assert row["commands"][3][row["commands"][3].index("--num-envs") + 1] == "1"
        wm = yaml.safe_load(
            (root / f"cycle_{index:04d}" / "world_model.yaml").read_text()
        )
        policy = yaml.safe_load(
            (root / f"cycle_{index:04d}" / "actor.yaml").read_text()
        )
        assert [item["name"] for item in wm["datasets"]] == [
            "teacher",
            *[f"student_cycle_{cycle:04d}" for cycle in range(1, index + 1)],
        ]
        assert len(policy["datasets"]) == index + 1
        assert policy["actor"]["bptt_horizon"] == 32
        assert policy["critic"]["enabled"] is False
        assert policy["world_model"]["checkpoint"].endswith(
            f"cycle_{index:04d}/world_model/checkpoint_00000003.pt"
        )
        assert row["actor_checkpoint"].endswith("checkpoint_00000005.pt")


def test_online_actor_contract_rejects_critic(tmp_path) -> None:
    config_path, _, _ = _online_config(tmp_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = Path(raw["actor"]["base_config"])
    actor = yaml.safe_load(base.read_text(encoding="utf-8"))
    actor["critic"]["enabled"] = True
    base.write_text(yaml.safe_dump(actor), encoding="utf-8")
    args = argparse.Namespace(actor=None, world_model=None, output=None, cycles=None)
    config = run_flat_online_cycle.load_config(config_path, args)
    layout = run_flat_online_cycle.cycle_layout(tmp_path / "result", 1)
    with pytest.raises(ValueError, match="reward-only"):
        run_flat_online_cycle.actor_config(
            config, layout, tmp_path / "wm.pt", [tmp_path / "student"]
        )


def _manifest(path: Path, *, source: str, gain: float) -> None:
    path.mkdir()
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "replay_contract": {
                        "source": source,
                        "source_sha256": source * 64,
                        "policy_joints": ["joint"],
                    },
                    "asset": {
                        "path": source,
                        "sha256": source * 64,
                        "timestep": 0.0025,
                        "controller": {"joint": {"gain": gain}},
                    },
                }
            }
        ),
        encoding="utf-8",
    )


def test_dataset_contract_ignores_paths_but_rejects_dynamics(tmp_path) -> None:
    teacher, student, incompatible = (
        tmp_path / "teacher",
        tmp_path / "student",
        tmp_path / "incompatible",
    )
    _manifest(teacher, source="a", gain=15.0)
    _manifest(student, source="b", gain=15.0)
    _manifest(incompatible, source="c", gain=14.0)
    result = run_flat_online_cycle.validate_dataset_contracts([teacher, student])
    assert result["status"] == "compatible"
    with pytest.raises(ValueError, match="dynamics/control"):
        run_flat_online_cycle.validate_dataset_contracts([teacher, incompatible])
