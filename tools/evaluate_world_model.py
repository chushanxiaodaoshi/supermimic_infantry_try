"""Evaluate a World Model on completely independent collected trajectories."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from supermimic.core.networks import WorldModel
from supermimic.data.dataset import ChunkDataset
from supermimic.infantry.spec import INFANTRY_SPEC
from tools.train_world_model import (
    COMPONENTS,
    EqualityConstraintEvaluator,
    PHASE_LABELS,
    rollout_error,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Independent dataset to evaluate; may be repeated.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    parser.add_argument("--windows", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--mjcf", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def dataset_arg(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise ValueError(f"dataset must be NAME=PATH, got {value!r}")
    return name, Path(path).expanduser().resolve()


def metric_row(errors: torch.Tensor) -> dict[str, float]:
    row = {
        name: float(errors[:, indices].square().mean().sqrt())
        for name, indices in COMPONENTS.items()
    }
    row["all_state_rmse"] = float(errors.square().mean().sqrt())
    row["max_absolute_error"] = float(errors.abs().max())
    return row


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.windows <= 0 or any(horizon <= 0 for horizon in args.horizons):
        raise ValueError("windows and horizons must be positive")
    device = torch.device(args.device)
    checkpoint = args.checkpoint.expanduser().resolve()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("robot_spec") != asdict(INFANTRY_SPEC):
        raise ValueError("checkpoint robot spec is incompatible")
    hidden = tuple(int(value) for value in payload["config"]["model"]["hidden"])
    model = WorldModel(INFANTRY_SPEC, hidden=hidden).to(device)
    model.load_state_dict(payload["world_model"], strict=True)
    model.eval()

    constraint = EqualityConstraintEvaluator(
        args.mjcf.expanduser().resolve() if args.mjcf else None
    )
    report: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": int(payload["iteration"]),
        "horizons": args.horizons,
        "windows_per_horizon": args.windows,
        "sources": {},
    }
    for source_index, value in enumerate(args.dataset):
        name, path = dataset_arg(value)
        dataset = ChunkDataset(path, seed=args.seed + source_index)
        source: dict[str, Any] = {
            "path": str(path),
            "transitions": int(dataset.manifest["transitions"]),
            "episodes": dataset.num_episodes(),
            "metrics": {},
        }
        if "command" in dataset.data:
            command = dataset.data["command"].float()
            source["command_min"] = command.min(dim=0).values.tolist()
            source["command_max"] = command.max(dim=0).values.tolist()

        for horizon in args.horizons:
            fields = ["state", "next_state", "applied_control", "context"]
            has_phase = "motion_phase" in dataset.data
            if has_phase:
                fields.append("motion_phase")
            batch = dataset.sample_windows(
                args.windows,
                horizon,
                split=None,
                device=device,
                fields=fields,
                rng=random.Random(args.seed + 10_000 * source_index + horizon),
            )
            errors, prediction = rollout_error(model, batch)
            final_error = errors[:, -1]
            if not torch.isfinite(final_error).all() or not torch.isfinite(prediction).all():
                raise FloatingPointError(f"non-finite rollout for {name} horizon={horizon}")
            row: dict[str, Any] = metric_row(final_error)
            row["finite_fraction"] = float(torch.isfinite(prediction).float().mean())
            if has_phase:
                phase = batch["motion_phase"][:, -1]
                row["motion_phases"] = {
                    PHASE_LABELS.get(int(label), f"unknown_{int(label)}"): {
                        **metric_row(final_error[phase == label]),
                        "windows": int((phase == label).sum()),
                    }
                    for label in phase.unique().tolist()
                }
            row["closed_chain_prediction"] = constraint.evaluate(prediction, 128)
            row["closed_chain_target"] = constraint.evaluate(
                batch["next_state"][:, -1], 128
            )
            source["metrics"][str(horizon)] = row
        report["sources"][name] = source

    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
