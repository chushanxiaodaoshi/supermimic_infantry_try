from __future__ import annotations

from pathlib import Path

import yaml

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
