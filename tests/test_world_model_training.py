from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml

from supermimic.core.networks import WorldModel
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


def test_normalizer_rebase_preserves_physical_predictions() -> None:
    torch.manual_seed(9)
    model = WorldModel(INFANTRY_SPEC, hidden=(16,), zero_init_output=False)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.1, 0.1)
        model.in_norm.mean.copy_(torch.randn(model.in_dim))
        model.in_norm.var.copy_(torch.rand(model.in_dim) + 0.2)
        model.in_norm.count.fill_(100)
        model.out_norm.mean.copy_(torch.randn(INFANTRY_SPEC.delta_dim))
        model.out_norm.var.copy_(torch.rand(INFANTRY_SPEC.delta_dim) + 0.2)
        model.out_norm.count.fill_(100)

    values = torch.randn(7, model.in_dim)

    def physical_prediction() -> torch.Tensor:
        normalized = model.net(model.in_norm(values))
        return normalized * model.out_norm.std() + model.out_norm.mean

    before = physical_prediction().detach().clone()
    old_input_mean = model.in_norm.mean.clone()
    old_input_std = model.in_norm.std().clone()
    old_output_mean = model.out_norm.mean.clone()
    old_output_std = model.out_norm.std().clone()

    with torch.no_grad():
        model.in_norm.mean.copy_(torch.randn(model.in_dim) * 2.0)
        model.in_norm.var.copy_(torch.rand(model.in_dim) * 2.0 + 0.1)
        model.in_norm.count.fill_(200)
        model.out_norm.mean.copy_(torch.randn(INFANTRY_SPEC.delta_dim) * 2.0)
        model.out_norm.var.copy_(torch.rand(INFANTRY_SPEC.delta_dim) * 2.0 + 0.1)
        model.out_norm.count.fill_(200)

    train_world_model.rebase_network_for_normalizers(
        model,
        old_input_mean=old_input_mean,
        old_input_std=old_input_std,
        old_output_mean=old_output_mean,
        old_output_std=old_output_std,
    )
    torch.testing.assert_close(physical_prediction(), before, rtol=2e-5, atol=2e-5)


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
            "horizons": [1, 2, 32],
            "windows_per_source": 2,
            "constraint_samples": 2,
            "mjcf_path": None,
        },
    }
    config_path = tmp_path / "wm.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train_world_model.py", "--config", str(config_path)])
    train_world_model.main()

    validation = json.loads(
        (output / "validation.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )
    assert validation["sources"]["flat"]["metrics"]["32"]["status"] == "unavailable"
    assert validation["sources"]["rotation"]["metrics"]["32"]["windows"] == 0
    assert validation["overall"]["32"]["status"] == "unavailable"

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

    warm_output = tmp_path / "warm_run"
    warm_config = yaml.safe_load(yaml.safe_dump(config))
    warm_config["output"] = str(warm_output)
    warm_config["datasets"] = [
        {"name": "flat", "path": str(first), "weight": 1.0},
        {"name": "jump", "path": str(second), "weight": 1.0},
    ]
    warm_config["training"]["iterations"] = 1
    warm_config["training"]["normalizer_samples"] = 8
    warm_path = tmp_path / "warm.yaml"
    warm_path.write_text(yaml.safe_dump(warm_config), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_world_model.py",
            "--config",
            str(warm_path),
            "--init-from",
            str(checkpoint_2),
        ],
    )
    train_world_model.main()
    warm = torch.load(
        warm_output / "checkpoint_00000001.pt", map_location="cpu", weights_only=False
    )
    assert warm["iteration"] == 1
    assert warm["optimizer"]["state"]
    assert warm["config"]["initialization"]["source_iteration"] == 2
    assert warm["config"]["initialization"]["optimizer"] == "reset"
    assert warm["world_model"]["in_norm.count"] == 8
