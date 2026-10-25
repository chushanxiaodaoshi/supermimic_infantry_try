from __future__ import annotations

from pathlib import Path

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
