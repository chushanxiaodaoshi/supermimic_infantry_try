"""Episode-aware sampling from chunked teacher datasets."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch


class ChunkDataset:
    """Load chunk tensors and sample windows without crossing episodes."""

    def __init__(self, directory: str | Path, *, seed: int = 42):
        self.directory = Path(directory).expanduser().resolve()
        self.rng = random.Random(seed)
        self.manifest = json.loads(
            (self.directory / "manifest.json").read_text(encoding="utf-8")
        )
        manifest = self.manifest
        files = sorted(self.directory.glob("chunk_*.pt"))
        expected = [
            self.directory / f"chunk_{index:06d}.pt"
            for index in range(manifest["chunks"])
        ]
        if files != expected:
            raise ValueError(
                "dataset chunk files do not match manifest; missing or stale chunks"
            )
        if not files:
            raise FileNotFoundError(f"no chunk_*.pt files in {self.directory}")
        chunks = [
            torch.load(path, map_location="cpu", weights_only=True) for path in files
        ]
        keys = set(chunks[0])
        if any(set(chunk) != keys for chunk in chunks[1:]):
            raise ValueError("dataset chunks have inconsistent fields")
        self.data = {key: torch.cat([chunk[key] for chunk in chunks]) for key in keys}
        if any(len(value) != manifest["transitions"] for value in self.data.values()):
            raise ValueError("dataset transition counts do not match manifest")
        self.episodes: dict[tuple[int, int], torch.Tensor] = {}
        episode_ids = self.data["episode_id"].long()
        run_ids = self.data.get("run_id", torch.zeros_like(episode_ids)).long()
        group_ids = torch.stack((run_ids, episode_ids), dim=-1)
        for run_id, episode_id in group_ids.unique(dim=0).tolist():
            indices = (
                (run_ids == run_id) & (episode_ids == episode_id)
            ).nonzero(as_tuple=False).flatten()
            order = torch.argsort(self.data["timestamp"][indices])
            self.episodes[(int(run_id), int(episode_id))] = indices[order]

    def _candidates(self, split: int | None, window: int) -> list[torch.Tensor]:
        return [
            indices
            for indices in self.episodes.values()
            if len(indices) >= window
            and (
                split is None
                or int(self.data["split"][indices[0]]) == split
            )
        ]

    def has_windows(self, split: int | None, window: int) -> bool:
        return bool(self._candidates(split, window))

    def sample_windows(
        self,
        batch_size: int,
        window: int,
        *,
        split: int | None = 0,
        device: str | torch.device = "cpu",
        fields: Iterable[str] | None = None,
        rng: random.Random | None = None,
    ) -> dict[str, torch.Tensor]:
        candidates = self._candidates(split, window)
        if not candidates:
            raise ValueError(
                f"no split={split} episodes contain a {window}-step window"
            )
        sampler = rng or self.rng
        rows = []
        for _ in range(batch_size):
            episode = sampler.choice(candidates)
            start = sampler.randrange(len(episode) - window + 1)
            rows.append(episode[start : start + window])
        index = torch.stack(rows)
        selected = tuple(
            fields or ("state", "next_state", "applied_control", "context")
        )
        missing = set(selected) - self.data.keys()
        if missing:
            raise KeyError(f"dataset is missing fields: {sorted(missing)}")
        return {name: self.data[name][index].to(device) for name in selected}

    def rng_state(self) -> object:
        return self.rng.getstate()

    def set_rng_state(self, state: object) -> None:
        self.rng.setstate(state)  # type: ignore[arg-type]

    def num_episodes(self, split: int | None = None) -> int:
        if split is None:
            return len(self.episodes)
        return sum(
            int(self.data["split"][indices[0]]) == split
            for indices in self.episodes.values()
        )

    def split_summary(self) -> dict[str, int]:
        labels = {0: "train", 1: "validation", 2: "test"}
        result: dict[str, int] = {}
        for split, label in labels.items():
            episodes = [
                indices
                for indices in self.episodes.values()
                if int(self.data["split"][indices[0]]) == split
            ]
            result[f"{label}_episodes"] = len(episodes)
            result[f"{label}_transitions"] = sum(len(indices) for indices in episodes)
        return result


@dataclass(frozen=True)
class DatasetSource:
    """One independently collected run in a weighted training mixture."""

    name: str
    dataset: ChunkDataset
    weight: float = 1.0


class MixtureDataset:
    """Weighted sampling without merging run-local episode identifiers."""

    def __init__(self, sources: Iterable[DatasetSource], *, seed: int = 42):
        self.sources = tuple(sources)
        if not self.sources:
            raise ValueError("at least one dataset source is required")
        if len({source.name for source in self.sources}) != len(self.sources):
            raise ValueError("dataset source names must be unique")
        if any(source.weight <= 0 for source in self.sources):
            raise ValueError("dataset source weights must be positive")
        dimensions = {
            (
                source.dataset.manifest["state_dim"],
                len(source.dataset.manifest["applied_control_names"]),
                len(source.dataset.manifest["context_names"]),
            )
            for source in self.sources
        }
        if len(dimensions) != 1:
            raise ValueError("dataset sources use incompatible robot contracts")
        self.rng = random.Random(seed)

    def sample_windows(
        self,
        batch_size: int,
        window: int,
        *,
        split: int = 0,
        device: str | torch.device = "cpu",
        fields: Iterable[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        choices = self.rng.choices(
            range(len(self.sources)),
            weights=[source.weight for source in self.sources],
            k=batch_size,
        )
        parts: list[tuple[int, dict[str, torch.Tensor]]] = []
        for source_index in sorted(set(choices)):
            count = choices.count(source_index)
            source = self.sources[source_index]
            values = source.dataset.sample_windows(
                count,
                window,
                split=split,
                device=device,
                fields=fields,
            )
            parts.append((source_index, values))
        keys = parts[0][1].keys()
        result = {
            key: torch.cat([values[key] for _, values in parts], dim=0)
            for key in keys
        }
        result["source_index"] = torch.cat(
            [
                torch.full(
                    (next(iter(values.values())).shape[0],),
                    source_index,
                    dtype=torch.long,
                    device=device,
                )
                for source_index, values in parts
            ]
        )
        return result

    def rng_state(self) -> dict[str, object]:
        return {
            "mixture": self.rng.getstate(),
            "sources": [source.dataset.rng_state() for source in self.sources],
        }

    def set_rng_state(self, state: dict[str, object]) -> None:
        self.rng.setstate(state["mixture"])  # type: ignore[arg-type]
        source_states = state["sources"]
        if not isinstance(source_states, list) or len(source_states) != len(
            self.sources
        ):
            raise ValueError("checkpoint dataset RNG state does not match sources")
        for source, source_state in zip(self.sources, source_states, strict=True):
            source.dataset.set_rng_state(source_state)
