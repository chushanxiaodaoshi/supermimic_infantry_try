from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest
import torch
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
        "cycles": 4,
        "asset_config": "asset.yaml",
        "initial_actor": str(actor),
        "initial_world_model": str(world_model),
        "teacher_datasets": [
            {"name": "teacher", "path": "teacher_data", "weight": 1.0}
        ],
        "replay": {
            "teacher_anchor_scale": 0.25,
            "recent_actor_cycles": 2,
            "actor_weight": 1.0,
            "actor_recency_decay": 0.5,
        },
        "student_collection": {
            "steps": 40,
            "max_episode_steps": 10,
            "chunk_steps": 8,
            "action_noise_std": 0.1,
            "action_noise_hold_steps": 10,
            "command_resample_steps": 50,
        },
        "world_model": {
            "hidden": [16],
            "ensemble_size": 3,
            "termination_head": True,
            "termination_hidden": [8],
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
        action_noise_hold_steps=1,
        command_resample_steps=0,
    )
    with pytest.raises(ValueError, match="steps"):
        collect_isaaclab_student.validate_args(args)
    args.steps = 1
    args.action_noise_std = -0.1
    with pytest.raises(ValueError, match="action-noise"):
        collect_isaaclab_student.validate_args(args)


def test_online_dry_run_builds_bounded_active_replay(tmp_path, monkeypatch) -> None:
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
    assert len(manifest["cycles"]) == 4
    for index, row in enumerate(manifest["cycles"], start=1):
        assert len(row["commands"]) == 4
        assert row["commands"][3][row["commands"][3].index("--num-envs") + 1] == "1"
        wm = yaml.safe_load(
            (root / f"cycle_{index:04d}" / "world_model.yaml").read_text()
        )
        policy = yaml.safe_load(
            (root / f"cycle_{index:04d}" / "actor.yaml").read_text()
        )
        first = max(1, index - 1)
        expected = [
            "teacher",
            *[f"actor_cycle_{cycle:04d}" for cycle in range(first, index + 1)],
        ]
        assert [item["name"] for item in wm["datasets"]] == expected
        assert [item["name"] for item in policy["datasets"]] == expected
        assert wm["datasets"][0]["weight"] == pytest.approx(0.25)
        assert wm["datasets"][-1]["weight"] == pytest.approx(1.0)
        assert wm["model"]["ensemble_size"] == 3
        assert wm["model"]["termination_head"] is True
        assert row["active_replay_sources"] == wm["datasets"]
        assert policy["actor"]["bptt_horizon"] == 32
        assert policy["critic"]["enabled"] is False
        checkpoint = Path(policy["world_model"]["checkpoint"])
        assert checkpoint.parts[-3:] == (
            f"cycle_{index:04d}",
            "world_model",
            "checkpoint_00000003.pt",
        )
        assert row["actor_checkpoint"].endswith("checkpoint_00000005.pt")
        collection_command = row["commands"][0]
        assert collection_command[
            collection_command.index("--action-noise-hold-steps") + 1
        ] == "10"


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


def _student_dataset(path: Path, actor: Path) -> None:
    path.mkdir()
    torch.save(
        {
            "student_action": torch.zeros(2, 6),
            "executed_action": torch.zeros(2, 6),
            "applied_control": torch.zeros(2, 8),
            "state": torch.zeros(2, 41),
            "next_state": torch.ones(2, 41),
            "terminated": torch.tensor([False, True]),
            "episode_id": torch.tensor([0, 0]),
            "run_id": torch.tensor([7, 7]),
        },
        path / "chunk_000000.pt",
    )
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "transitions": 2,
                "metadata": {
                    "student": {
                        "checkpoint": str(actor),
                        "sha256": run_flat_online_cycle.sha256_file(actor),
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_student_audit_binds_current_actor_and_terminal_next_state(tmp_path) -> None:
    actor = tmp_path / "actor.pt"
    actor.write_bytes(b"current actor")
    dataset = tmp_path / "student"
    _student_dataset(dataset, actor)
    audit = run_flat_online_cycle.audit_student_dataset(dataset, actor)
    assert audit["transitions"] == 2
    assert audit["episodes"] == 1
    assert audit["recorded_dimensions"]["applied_control"] == 8
    wrong_actor = tmp_path / "wrong.pt"
    wrong_actor.write_bytes(b"teacher or stale actor")
    with pytest.raises(ValueError, match="current Actor"):
        run_flat_online_cycle.audit_student_dataset(dataset, wrong_actor)


def test_cycle_summary_reports_comparable_survival_gap(tmp_path) -> None:
    layout = run_flat_online_cycle.cycle_layout(tmp_path, 1)
    layout["world_model"].mkdir(parents=True)
    layout["actor"].mkdir()
    validation = {
        "sources": {
            "teacher": {"metrics": {"32": {"all_state_rmse": 1.0}}},
            "student_cycle_0001": {
                "split_summary": {"validation_episodes": 2},
                "metrics": {"32": {"status": "unavailable", "windows": 0}},
            },
        }
    }
    (layout["world_model"] / "validation.jsonl").write_text(
        json.dumps(validation) + "\n", encoding="utf-8"
    )
    (layout["actor"] / "train_log.jsonl").write_text(
        json.dumps({"survival": 0.7, "wm_input_rms": 1.2}) + "\n",
        encoding="utf-8",
    )
    layout["evaluation"].write_text(
        json.dumps(
            {
                "model_survival_horizon_mean": 0.6,
                "survival_horizon_fraction": 0.25,
                "mean_episode_length": 23.0,
                "fall_fraction": 1.0,
                "forward_velocity_mae": 0.5,
                "yaw_velocity_mae": 0.1,
            }
        ),
        encoding="utf-8",
    )
    summary = run_flat_online_cycle.summarize_cycle(
        layout, ["teacher"], "student_cycle_0001"
    )
    assert summary["model_survival_32"] == pytest.approx(0.6)
    assert summary["training_mixture_model_survival_32"] == pytest.approx(0.7)
    assert summary["real_survival_32"] == pytest.approx(0.25)
    assert summary["survival_gap_absolute"] == pytest.approx(0.35)
    assert summary["student_32_step_prediction"]["status"] == "unavailable"
