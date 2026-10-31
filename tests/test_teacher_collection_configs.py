from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from tools.collect_isaaclab_teacher import load_config
from tools.replay_isaaclab_teacher import KeypadControls

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


def test_teacher_configs_resolve_vendored_reference_without_mjlab_root(
    monkeypatch,
) -> None:
    monkeypatch.delenv("MJLAB_ROOT", raising=False)
    for filename in (
        "isaaclab_flat.yaml",
        "isaaclab_jump.yaml",
        "isaaclab_rotation.yaml",
    ):
        args = SimpleNamespace(
            config=CONFIG_DIR / filename,
            replay=None,
            mjcf=None,
            usd=None,
            onnx=None,
            output=None,
            steps=None,
            seed=None,
            validate_only=True,
        )
        config = load_config(args)
        reference_paths = (
            config["asset"]["replay_source"],
            config["asset"]["mjcf"],
            config["teacher"]["onnx"],
        )
        assert all(path.is_file() for path in reference_paths)
        assert all(path.is_relative_to(ROOT / "mujoco390") for path in reference_paths)


def test_rotation_collection_covers_translation_while_spinning() -> None:
    commands = _load("isaaclab_rotation.yaml")["commands"]
    assert commands == {
        "lin_vel_x": [-1.0, 1.0],
        "ang_vel_z": [6.0, 10.0],
        "height": [0.15, 0.15],
    }


def test_fixed_command_range_is_the_replay_neutral_value() -> None:
    assert KeypadControls.neutral_value((8.0, 8.0)) == 8.0
    assert KeypadControls.neutral_value((6.0, 10.0)) == 8.0
    assert KeypadControls.neutral_value((-1.0, 1.0)) == 0.0
    assert KeypadControls.directional_value(-1.0, (8.0, 8.0), 0.5) == 8.0
    assert KeypadControls.directional_value(-1.0, (6.0, 10.0), 0.5) == 6.0
    assert KeypadControls.directional_value(1.0, (6.0, 10.0), 0.5) == 10.0
    assert KeypadControls.command_default("vx", 1.0, (-1.0, 1.0)) == 1.0


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
        seed=123,
        validate_only=False,
    )
    collection = load_config(args)["collection"]
    assert collection["steps"] == 200
    assert collection["seed"] == 123
