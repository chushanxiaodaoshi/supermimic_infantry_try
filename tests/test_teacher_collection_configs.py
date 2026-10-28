from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from tools.collect_isaaclab_teacher import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "infantry" / "collection"


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


def test_teacher_roles_follow_replay_weight_selection() -> None:
    expected = {
        "isaaclab_flat.yaml": ("flat", "rough.onnx"),
        "isaaclab_jump.yaml": ("jump", "jump.onnx"),
        "isaaclab_rotation.yaml": (
            "rotation",
            "spin_move_testA++rewardfine1.onnx",
        ),
    }
    for filename, (skill, basename) in expected.items():
        teacher = _load(filename)["teacher"]
        assert teacher["skill"] == skill
        assert Path(teacher["onnx"]).name == basename


def test_rotation_collection_matches_in_place_spin_replay_command() -> None:
    commands = _load("isaaclab_rotation.yaml")["commands"]
    assert commands == {
        "lin_vel_x": [0.0, 0.0],
        "ang_vel_z": [8.0, 8.0],
        "height": [0.15, 0.15],
    }


def test_collection_steps_can_be_overridden_from_cli(tmp_path) -> None:
    config_path = tmp_path / "collection.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "asset": {
                    "replay_source": "replay.py",
                    "mjcf": "robot.xml",
                    "usd": "robot.usda",
                },
                "teacher": {"onnx": "teacher.onnx"},
                "collection": {"output": "output", "steps": 20_000},
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        config=config_path,
        replay=None,
        mjcf=None,
        usd=None,
        onnx=None,
        output=None,
        steps=200,
        validate_only=False,
    )
    assert load_config(args)["collection"]["steps"] == 200
