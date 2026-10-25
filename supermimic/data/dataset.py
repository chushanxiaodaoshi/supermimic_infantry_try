"""Episode-aware sampling from chunked teacher datasets."""

from __future__ import annotations

import random
from pathlib import Path

import torch


class ChunkDataset:
    """Load chunk tensors and sample windows without crossing episodes."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser().resolve()
        files = sorted(self.directory.glob("chunk_*.pt"))
        if not files:
            raise FileNotFoundError(f"no chunk_*.pt files in {self.directory}")
        chunks = [
            torch.load(path, map_location="cpu", weights_only=True) for path in files
        ]
        keys = set(chunks[0])
        if any(set(chunk) != keys for chunk in chunks[1:]):
            raise ValueError("dataset chunks have inconsistent fields")
        self.data = {key: torch.cat([chunk[key] for chunk in chunks]) for key in keys}
        self.episodes: dict[int, torch.Tensor] = {}
        episode_ids = self.data["episode_id"].long()
        for episode_id in episode_ids.unique().tolist():
            indices = (episode_ids == episode_id).nonzero(as_tuple=False).flatten()
            order = torch.argsort(self.data["timestamp"][indices])
            self.episodes[int(episode_id)] = indices[order]

    def sample_windows(
        self,
        batch_size: int,
        window: int,
        *,
        split: int = 0,
        device: str | torch.device = "cpu",
    ) -> dict[str, torch.Tensor]:
        candidates = [
            indices
            for indices in self.episodes.values()
            if len(indices) >= window and int(self.data["split"][indices[0]]) == split
        ]
        if not candidates:
            raise ValueError(
                f"no split={split} episodes contain a {window}-step window"
            )
        rows = []
        for _ in range(batch_size):
            episode = random.choice(candidates)
            start = random.randrange(len(episode) - window + 1)
            rows.append(episode[start : start + window])
        index = torch.stack(rows)
        fields = ("state", "next_state", "applied_control", "context")
        return {name: self.data[name][index].to(device) for name in fields}

    def num_episodes(self, split: int | None = None) -> int:
        if split is None:
            return len(self.episodes)
        return sum(
            int(self.data["split"][indices[0]]) == split
            for indices in self.episodes.values()
        )
