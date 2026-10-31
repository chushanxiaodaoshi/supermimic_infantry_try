"""Live Matplotlib monitor for World Model JSONL logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()

    run = args.run.expanduser().resolve()
    train_path = run / "train_log.jsonl"
    validation_path = run / "validation.jsonl"

    plt.ion()
    figure, (loss_axis, validation_axis) = plt.subplots(2, 1, figsize=(11, 8))
    figure.canvas.manager.set_window_title(f"World Model: {run.name}")

    while plt.fignum_exists(figure.number):
        training = load_jsonl(train_path)
        validation = load_jsonl(validation_path)

        loss_axis.clear()
        validation_axis.clear()

        if training:
            iterations = [row["iteration"] for row in training]
            loss_axis.plot(iterations, [row["loss"] for row in training], label="rollout loss")
            loss_axis.plot(
                iterations,
                [row["one_step_normalized_mse"] for row in training],
                label="1-step normalized MSE",
                alpha=0.8,
            )
            loss_axis.plot(
                iterations,
                [row["last_step_normalized_mse"] for row in training],
                label="last-step normalized MSE",
                alpha=0.8,
            )
            loss_axis.set_yscale("log")
            loss_axis.set_title(f"Training — iteration {iterations[-1]}")
            loss_axis.set_ylabel("error (log scale)")
            loss_axis.legend()
            loss_axis.grid(True, alpha=0.25)

        if validation:
            val_iterations = [row["iteration"] for row in validation]
            horizons = validation[-1]["horizons"]
            for horizon in horizons:
                key = str(horizon)
                validation_axis.plot(
                    val_iterations,
                    [row["overall"][key]["all_state_rmse"] for row in validation],
                    marker="o",
                    label=f"{horizon}-step RMSE",
                )
            validation_axis.set_yscale("log")
            validation_axis.set_title("Validation")
            validation_axis.set_xlabel("iteration")
            validation_axis.set_ylabel("state RMSE (log scale)")
            validation_axis.legend()
            validation_axis.grid(True, alpha=0.25)

        figure.tight_layout()
        plt.pause(max(args.interval, 0.1))


if __name__ == "__main__":
    main()
