from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from supermimic.data.chunks import ChunkWriter
from supermimic.data.dataset import ChunkDataset
from supermimic.infantry.spec import INFANTRY_SPEC


def test_episode_windows_do_not_cross_boundaries(tmp_path: Path) -> None:
    writer = ChunkWriter(tmp_path, INFANTRY_SPEC, chunk_steps=1)
    for step in range(3):
        batch = 2
        episode_id = torch.tensor([10, 20])
        writer.append(
            {
                "state": torch.full((batch, INFANTRY_SPEC.state_dim), float(step)),
                "next_state": torch.full(
                    (batch, INFANTRY_SPEC.state_dim), float(step + 1)
                ),
                "applied_control": torch.zeros(batch, INFANTRY_SPEC.control_dim),
                "context": torch.zeros(batch, INFANTRY_SPEC.context_dim),
                "episode_id": episode_id,
                "timestamp": torch.full((batch,), step * INFANTRY_SPEC.control_dt),
            }
        )
    writer.close()
    dataset = ChunkDataset(tmp_path)
    split = int(dataset.data["split"][0])
    windows = dataset.sample_windows(8, 3, split=split)
    state_steps = windows["state"][..., 0]
    assert torch.all(state_steps[:, 1:] - state_steps[:, :-1] == 1)


def write_dataset(directory: Path) -> None:
    writer = ChunkWriter(directory, INFANTRY_SPEC, chunk_steps=4)
    for step in range(12):
        writer.append(
            {
                "state": torch.full((1, INFANTRY_SPEC.state_dim), float(step)),
                "next_state": torch.full((1, INFANTRY_SPEC.state_dim), float(step + 1)),
                "applied_control": torch.zeros(1, INFANTRY_SPEC.control_dim),
                "context": torch.zeros(1, INFANTRY_SPEC.context_dim),
                "episode_id": torch.tensor([0]),
                "timestamp": torch.tensor([step * INFANTRY_SPEC.control_dt]),
            }
        )
    writer.close()


def test_writer_rejects_existing_data_without_modifying_it(tmp_path: Path) -> None:
    write_dataset(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(FileExistsError, match="must be empty"):
        ChunkWriter(tmp_path, INFANTRY_SPEC)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_dataset_rejects_stale_chunk(tmp_path: Path) -> None:
    write_dataset(tmp_path)
    torch.save({}, tmp_path / "chunk_000003.pt")
    with pytest.raises(ValueError, match="do not match manifest"):
        ChunkDataset(tmp_path)


def test_writer_snapshots_cpu_tensors(tmp_path: Path) -> None:
    writer = ChunkWriter(tmp_path, INFANTRY_SPEC, chunk_steps=4)
    timestamp = torch.tensor([0.0])
    writer.append({"episode_id": torch.tensor([0]), "timestamp": timestamp})
    timestamp.add_(9)
    writer.close()
    saved = torch.load(tmp_path / "chunk_000000.pt", weights_only=True)
    torch.testing.assert_close(saved["timestamp"], torch.tensor([0.0]))


def test_window_sampling_has_an_independent_seed(tmp_path: Path) -> None:
    write_dataset(tmp_path)
    first = ChunkDataset(tmp_path, seed=7)
    second = ChunkDataset(tmp_path, seed=7)
    different = ChunkDataset(tmp_path, seed=8)
    for _ in range(2):
        expected = first.sample_windows(32, 3)["state"]
        random.seed(999)
        for _ in range(10):
            random.random()
        torch.testing.assert_close(second.sample_windows(32, 3)["state"], expected)
        assert not torch.equal(different.sample_windows(32, 3)["state"], expected)
