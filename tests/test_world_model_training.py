from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

from supermimic.data.chunks import ChunkWriter, episode_split
from supermimic.infantry.spec import INFANTRY_SPEC
from tools import train_world_model


def _episode_for_split(split: int, seed: int, run_id: int) -> int:
    for episode in range(1000):
        value = episode_split(
            torch.tensor([episode]), seed, torch.tensor([run_id])
        )
        if int(value[0]) == split:
            return episode
    raise AssertionError("unable to find deterministic episode split")


def _dataset(path: Path, offset: float) -> None:
    writer = ChunkWriter(path, INFANTRY_SPEC, chunk_steps=8, split_seed=7)
    for split in (0, 1):
        episode = _episode_for_split(split, 7, 11)
        for step in range(20):
            state = torch.zeros(1, INFANTRY_SPEC.state_dim)
            state[:, 3] = 1.0
            state[:, 0] = offset + step * 0.01
            next_state = state.clone()
            next_state[:, 0] += 0.01
            writer.append(
                {
                    "state": state,
                    "next_state": next_state,
                    "applied_control": torch.zeros(1, INFANTRY_SPEC.control_dim),
                    "context": torch.zeros(1, INFANTRY_SPEC.context_dim),
                    "timestamp": torch.tensor([step * INFANTRY_SPEC.control_dt]),
                    "run_id": torch.tensor([11]),
                    "episode_id": torch.tensor([episode]),
                }
            )
    writer.close()


def test_config_training_and_resume_restore_full_state(tmp_path, monkeypatch) -> None:
    first, second = tmp_path / "flat", tmp_path / "rotation"
    _dataset(first, 0.0)
    _dataset(second, 1.0)
    output = tmp_path / "run"
    config = {
        "seed": 5,
        "device": "cpu",
        "output": str(output),
        "datasets": [
            {"name": "flat", "path": str(first), "weight": 1.0},
            {"name": "rotation", "path": str(second), "weight": 2.0},
        ],
        "model": {"hidden": [16]},
        "training": {
            "iterations": 2,
            "batch_size": 4,
            "window": 2,
            "learning_rate": 0.0002,
            "normalizer_samples": 16,
            "max_grad_norm": 1.0,
            "checkpoint_every": 1,
        },
        "validation": {
            "interval": 1,
            "split": 1,
            "horizons": [1, 2],
            "windows_per_source": 2,
            "constraint_samples": 2,
            "mjcf_path": None,
        },
    }
    config_path = tmp_path / "wm.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train_world_model.py", "--config", str(config_path)])
    train_world_model.main()

    checkpoint_2 = output / "checkpoint_00000002.pt"
    first_payload = torch.load(checkpoint_2, map_location="cpu", weights_only=False)
    assert first_payload["optimizer"]["state"]
    norm_count = first_payload["world_model"]["in_norm.count"].clone()
    resolved_before = (output / "resolved_config.yaml").read_bytes()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_world_model.py",
            "--config",
            str(config_path),
            "--resume",
            str(checkpoint_2),
            "--iterations",
            "3",
        ],
    )
    train_world_model.main()
    resumed = torch.load(
        output / "checkpoint_00000003.pt", map_location="cpu", weights_only=False
    )
    torch.testing.assert_close(resumed["world_model"]["in_norm.count"], norm_count)
    assert resumed["iteration"] == 3
    assert resumed["optimizer"]["state"]
    assert (output / "resolved_config.yaml").read_bytes() == resolved_before
