"""Chunked transition writer with episode-level data splitting."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from supermimic.core.spec import RobotSpec


def episode_split(episode_id: torch.Tensor, seed: int) -> torch.Tensor:
    """Map whole episodes to train=0, validation=1, test=2."""
    bucket = ((episode_id * 1_103_515_245 + seed) & 0x7FFF_FFFF) % 100
    return torch.where(bucket < 80, 0, torch.where(bucket < 90, 1, 2)).to(torch.uint8)


class ChunkWriter:
    def __init__(
        self,
        destination: str | Path,
        spec: RobotSpec,
        *,
        chunk_steps: int = 256,
        split_seed: int = 42,
        metadata: dict | None = None,
    ):
        self.destination = Path(destination).expanduser().resolve()
        self.destination.mkdir(parents=True, exist_ok=True)
        self.spec = spec
        self.chunk_steps = chunk_steps
        self.split_seed = split_seed
        self.metadata = metadata or {}
        self.pending: dict[str, list[torch.Tensor]] = {}
        self.pending_steps = 0
        self.chunk_index = 0
        self.transitions = 0

    def append(self, values: dict[str, torch.Tensor]) -> None:
        if not values:
            raise ValueError("cannot append an empty transition")
        batch = next(iter(values.values())).shape[0]
        if any(value.shape[0] != batch for value in values.values()):
            raise ValueError("all transition tensors must share the leading batch size")
        payload = dict(values)
        payload["split"] = episode_split(payload["episode_id"], self.split_seed)
        for name, value in payload.items():
            self.pending.setdefault(name, []).append(value.detach().cpu())
        self.pending_steps += 1
        self.transitions += batch
        if self.pending_steps >= self.chunk_steps:
            self.flush()

    def flush(self) -> None:
        if not self.pending_steps:
            return
        chunk = {name: torch.cat(parts, dim=0) for name, parts in self.pending.items()}
        path = self.destination / f"chunk_{self.chunk_index:06d}.pt"
        torch.save(chunk, path)
        self.pending.clear()
        self.pending_steps = 0
        self.chunk_index += 1
        self._write_manifest()

    def close(self) -> None:
        self.flush()
        self._write_manifest()

    def _write_manifest(self) -> None:
        manifest = {
            "format_version": 1,
            "robot": self.spec.name,
            "state_dim": self.spec.state_dim,
            "actor_obs_dim": self.spec.actor_obs_dim,
            "policy_action_names": self.spec.policy_action_names,
            "applied_control_names": self.spec.control_names,
            "context_names": self.spec.context_names,
            "split_labels": {"0": "train", "1": "validation", "2": "test"},
            "split_unit": "episode_id",
            "split_seed": self.split_seed,
            "chunks": self.chunk_index,
            "transitions": self.transitions,
            "metadata": self.metadata,
        }
        (self.destination / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
