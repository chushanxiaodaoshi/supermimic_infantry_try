"""Train the shared differentiable world model on collected transitions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
from torch import nn

from supermimic.core.networks import WorldModel
from supermimic.data.dataset import ChunkDataset
from supermimic.infantry.spec import INFANTRY_SPEC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    return parser.parse_args()


def error_floor(num_joints: int, device: torch.device) -> torch.Tensor:
    return torch.cat(
        (
            torch.full((3,), 3.0e-3),
            torch.full((3,), 5.0e-3),
            torch.full((3,), 2.0e-2),
            torch.full((3,), 5.0e-2),
            torch.full((num_joints,), 3.0e-3),
            torch.full((num_joints,), 5.0e-2),
        )
    ).to(device)


def save_checkpoint(
    path: Path,
    model: WorldModel,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    args: argparse.Namespace,
) -> None:
    payload = {
        "world_model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "robot_spec": asdict(INFANTRY_SPEC),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset = ChunkDataset(args.dataset, seed=args.seed)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "args.json").write_text(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    model = WorldModel(INFANTRY_SPEC, hidden=tuple(args.hidden)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    floor = error_floor(INFANTRY_SPEC.num_joints, device)

    # Bootstrap normalization from real transitions before optimizing.
    normalizer_batch = dataset.sample_windows(
        max(args.batch_size, 1024), 1, split=0, device=device
    )
    state = model.codec.unflatten(normalizer_batch["state"][:, 0])
    next_state = model.codec.unflatten(normalizer_batch["next_state"][:, 0])
    model.in_norm.update(
        model.inputs(
            state,
            normalizer_batch["applied_control"][:, 0],
            normalizer_batch["context"][:, 0],
        )
    )
    model.out_norm.update(model.transition_targets(state, next_state))
    log_path = output / "train_log.jsonl"
    with log_path.open("a", encoding="utf-8") as log:
        for iteration in range(1, args.iterations + 1):
            batch = dataset.sample_windows(
                args.batch_size, args.window, split=0, device=device
            )
            prediction = model.codec.unflatten(batch["state"][:, 0])
            step_errors = []
            normalizer = torch.maximum(model.out_norm.std(), floor)
            for step in range(args.window):
                prediction = model.step(
                    prediction,
                    batch["applied_control"][:, step],
                    batch["context"][:, step],
                )
                target = model.codec.unflatten(batch["next_state"][:, step])
                error = model.state_error(prediction, target) / normalizer
                step_errors.append(error.square().mean(dim=-1))
            errors = torch.stack(step_errors, dim=1)
            loss = errors.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            row = {
                "iteration": iteration,
                "loss": float(loss.detach()),
                "one_step_error": float(errors[:, 0].mean().detach()),
                "last_step_error": float(errors[:, -1].mean().detach()),
                "gradient_norm": float(grad_norm.detach()),
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            if iteration == 1 or iteration % 100 == 0:
                print(json.dumps(row), flush=True)
            if iteration % args.checkpoint_every == 0 or iteration == args.iterations:
                save_checkpoint(output / "last.pt", model, optimizer, iteration, args)


if __name__ == "__main__":
    main()
