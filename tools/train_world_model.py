"""Pretrain one shared World Model from weighted PPO Teacher datasets."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import yaml
from torch import nn

from supermimic.core.networks import WorldModel
from supermimic.data.dataset import ChunkDataset, DatasetSource, MixtureDataset
from supermimic.infantry.spec import INFANTRY_SPEC

COMPONENTS = {
    "root_position": slice(0, 3),
    "root_orientation": slice(3, 6),
    "root_linear_velocity": slice(6, 9),
    "root_angular_velocity": slice(9, 12),
    "joint_position": slice(12, 12 + INFANTRY_SPEC.num_joints),
    "joint_velocity": slice(12 + INFANTRY_SPEC.num_joints, None),
}
PHASE_LABELS = {
    0: "contact",
    1: "takeoff_contact",
    2: "airborne_ascent",
    3: "airborne_descent",
    4: "landing_contact",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int)
    return parser.parse_args()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    return value


def load_config(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = _mapping(loaded, "world-model config")
    for section in ("model", "training", "validation"):
        _mapping(cfg.get(section), section)
    datasets = cfg.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("datasets must be a non-empty list")
    for index, source in enumerate(datasets):
        _mapping(source, f"datasets[{index}]")
        if not source.get("name") or not source.get("path"):
            raise ValueError(f"datasets[{index}] requires name and path")
    if args.device is not None:
        cfg["device"] = args.device
    if args.output is not None:
        cfg["output"] = str(args.output)
    if args.iterations is not None:
        cfg["training"]["iterations"] = args.iterations
    if not cfg.get("output"):
        raise ValueError("output is required")
    cfg.setdefault("seed", 42)
    cfg.setdefault("device", "cuda:0")
    return cfg


def _resolve(value: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (REPO / path).resolve() if not path.is_absolute() else path.resolve()


def build_datasets(cfg: dict[str, Any]) -> tuple[MixtureDataset, list[DatasetSource]]:
    seed = int(cfg["seed"])
    sources = [
        DatasetSource(
            name=str(item["name"]),
            dataset=ChunkDataset(_resolve(item["path"]), seed=seed + index + 1),
            weight=float(item.get("weight", 1.0)),
        )
        for index, item in enumerate(cfg["datasets"])
    ]
    return MixtureDataset(sources, seed=seed + 97), sources


def error_floor(device: torch.device) -> torch.Tensor:
    return torch.cat(
        (
            torch.full((3,), 3.0e-3),
            torch.full((3,), 5.0e-3),
            torch.full((3,), 2.0e-2),
            torch.full((3,), 5.0e-2),
            torch.full((INFANTRY_SPEC.num_joints,), 3.0e-3),
            torch.full((INFANTRY_SPEC.num_joints,), 5.0e-2),
        )
    ).to(device)


def fit_normalizers(
    model: WorldModel,
    dataset: MixtureDataset,
    *,
    samples: int,
    batch_size: int,
    device: torch.device,
) -> None:
    """Fit statistics exclusively from split=0 (training) transitions."""
    remaining = samples
    while remaining > 0:
        count = min(batch_size, remaining)
        batch = dataset.sample_windows(count, 1, split=0, device=device)
        state = model.codec.unflatten(batch["state"][:, 0])
        next_state = model.codec.unflatten(batch["next_state"][:, 0])
        model.in_norm.update(
            model.inputs(
                state,
                batch["applied_control"][:, 0],
                batch["context"][:, 0],
            )
        )
        model.out_norm.update(model.transition_targets(state, next_state))
        remaining -= count


def rollout_error(
    model: WorldModel, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = model.codec.unflatten(batch["state"][:, 0])
    errors = []
    for step in range(batch["state"].shape[1]):
        prediction = model.step(
            prediction,
            batch["applied_control"][:, step],
            batch["context"][:, step],
        )
        target = model.codec.unflatten(batch["next_state"][:, step])
        errors.append(model.state_error(prediction, target))
    return torch.stack(errors, dim=1), model.codec.flatten(prediction)


class EqualityConstraintEvaluator:
    """Optional MuJoCo equality residual check for predicted qpos states."""

    def __init__(self, mjcf_path: Path | None):
        self.path = mjcf_path
        self.reason: str | None = None
        self.mujoco: Any = None
        self.model: Any = None
        if mjcf_path is None:
            self.reason = "validation.mjcf_path is not configured"
            return
        try:
            import mujoco

            self.mujoco = mujoco
            self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            self.reason = f"unable to load MuJoCo model: {error}"

    def evaluate(self, states: torch.Tensor, limit: int) -> dict[str, Any]:
        if self.model is None:
            return {"status": "unavailable", "reason": self.reason}
        if self.model.nq != INFANTRY_SPEC.state_dim - 20:
            return {
                "status": "unavailable",
                "reason": f"MJCF nq={self.model.nq} does not match collected qpos width 21",
            }
        values: list[float] = []
        data = self.mujoco.MjData(self.model)
        equality_type = int(self.mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        for state in states[:limit].detach().cpu().double().numpy():
            data.qpos[:7] = state[0:7]
            data.qpos[7:] = state[13 : 13 + INFANTRY_SPEC.num_joints]
            self.mujoco.mj_forward(self.model, data)
            mask = data.efc_type[: data.nefc] == equality_type
            if mask.any():
                values.extend(abs(data.efc_pos[: data.nefc][mask]).tolist())
        if not values:
            return {"status": "unavailable", "reason": "no active equality rows"}
        tensor = torch.tensor(values)
        return {
            "status": "ok",
            "rms": float(tensor.square().mean().sqrt()),
            "max": float(tensor.max()),
            "samples": min(len(states), limit),
        }


def _metric_row(errors: torch.Tensor) -> dict[str, float]:
    result = {
        name: float(errors[:, indices].square().mean().sqrt())
        for name, indices in COMPONENTS.items()
    }
    result["all_state_rmse"] = float(errors.square().mean().sqrt())
    result["max_absolute_error"] = float(errors.abs().max())
    return result


@torch.no_grad()
def evaluate(
    model: WorldModel,
    sources: list[DatasetSource],
    validation_cfg: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    horizons = [int(value) for value in validation_cfg.get("horizons", [1, 4, 8, 16])]
    batch_size = int(validation_cfg.get("windows_per_source", 256))
    split = int(validation_cfg.get("split", 1))
    requested_fields = (
        "state",
        "next_state",
        "applied_control",
        "context",
        "motion_phase",
    )
    mjcf_value = validation_cfg.get("mjcf_path")
    constraint = EqualityConstraintEvaluator(_resolve(mjcf_value) if mjcf_value else None)
    constraint_limit = int(validation_cfg.get("constraint_samples", 64))
    report: dict[str, Any] = {"horizons": horizons, "sources": {}, "overall": {}}
    overall_errors: dict[int, list[torch.Tensor]] = {horizon: [] for horizon in horizons}

    for source_index, source in enumerate(sources):
        source_report: dict[str, Any] = {
            "manifest": source.dataset.manifest.get("metadata", {}),
            "split_summary": source.dataset.split_summary(),
            "metrics": {},
        }
        has_phase = "motion_phase" in source.dataset.data
        fields = requested_fields if has_phase else requested_fields[:-1]
        for horizon in horizons:
            batch = source.dataset.sample_windows(
                batch_size,
                horizon,
                split=split,
                device=device,
                fields=fields,
                rng=random.Random(seed + 10_000 * source_index + horizon),
            )
            errors, predicted_state = rollout_error(model, batch)
            final_error = errors[:, -1]
            if not torch.isfinite(final_error).all() or not torch.isfinite(predicted_state).all():
                raise FloatingPointError(
                    f"non-finite validation rollout for {source.name} at horizon {horizon}"
                )
            row: dict[str, Any] = _metric_row(final_error)
            row["windows"] = batch_size
            row["finite_fraction"] = float(torch.isfinite(predicted_state).float().mean())
            if has_phase:
                phase = batch["motion_phase"][:, -1]
                row["motion_phases"] = {
                    PHASE_LABELS.get(int(value), f"unknown_{int(value)}"): {
                        **_metric_row(final_error[phase == value]),
                        "windows": int((phase == value).sum()),
                    }
                    for value in phase.unique().tolist()
                }
            row["closed_chain_prediction"] = constraint.evaluate(
                predicted_state, constraint_limit
            )
            row["closed_chain_target"] = constraint.evaluate(
                batch["next_state"][:, -1], constraint_limit
            )
            source_report["metrics"][str(horizon)] = row
            overall_errors[horizon].append(final_error.detach().cpu())
        report["sources"][source.name] = source_report

    for horizon, parts in overall_errors.items():
        report["overall"][str(horizon)] = {
            **_metric_row(torch.cat(parts)),
            "windows": sum(len(part) for part in parts),
        }
    if "1" in report["overall"] and len(horizons) > 1:
        baseline = max(report["overall"]["1"]["all_state_rmse"], 1.0e-12)
        report["rollout_growth_vs_step_1"] = {
            str(horizon): report["overall"][str(horizon)]["all_state_rmse"] / baseline
            for horizon in horizons
        }
    model.train()
    return report


def _resume_signature(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "datasets": cfg["datasets"],
        "model": cfg["model"],
        "batch_size": cfg["training"]["batch_size"],
        "window": cfg["training"]["window"],
        "seed": cfg["seed"],
    }


def save_checkpoint(
    path: Path,
    model: WorldModel,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    cfg: dict[str, Any],
    dataset: MixtureDataset,
    elapsed_seconds: float,
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    payload = {
        "world_model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "robot_spec": asdict(INFANTRY_SPEC),
        "config": cfg,
        "resume_signature": _resume_signature(cfg),
        "dataset_rng_state": dataset.rng_state(),
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "elapsed_seconds": elapsed_seconds,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def restore_checkpoint(
    path: Path,
    model: WorldModel,
    optimizer: torch.optim.Optimizer,
    dataset: MixtureDataset,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[int, float]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload["resume_signature"] != _resume_signature(cfg):
        raise ValueError("resume checkpoint is incompatible with the current config")
    model.load_state_dict(payload["world_model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    dataset.set_rng_state(payload["dataset_rng_state"])
    random.setstate(payload["python_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    cuda_state = payload.get("cuda_rng_state_all")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
    return int(payload["iteration"]), float(payload.get("elapsed_seconds", 0.0))


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    cfg = load_config(config_path, args)
    seed = int(cfg["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    output = _resolve(cfg["output"])
    resume = args.resume.expanduser().resolve() if args.resume else None
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError(f"output must be empty for a new run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    resolved_config = dict(cfg)
    resolved_config["output"] = str(output)
    resolved_config["datasets"] = [
        {**item, "path": str(_resolve(item["path"]))} for item in cfg["datasets"]
    ]
    resolved_path = output / "resolved_config.yaml"
    if resume is None:
        resolved_path.write_text(
            yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8"
        )
    elif not resolved_path.is_file():
        raise FileNotFoundError(
            f"resume output is missing its original resolved config: {resolved_path}"
        )

    dataset, sources = build_datasets(resolved_config)
    training = resolved_config["training"]
    validation = resolved_config["validation"]
    window = int(training["window"])
    horizons = [int(value) for value in validation.get("horizons", [1, 4, 8, 16])]
    for source in sources:
        for split, label, required_window in (
            (0, "training", window),
            (int(validation.get("split", 1)), "validation", max(horizons)),
        ):
            if not source.dataset.has_windows(split, required_window):
                raise ValueError(
                    f"source {source.name!r} has no {label} episode with "
                    f"{required_window} consecutive transitions"
                )

    hidden = tuple(int(value) for value in resolved_config["model"].get("hidden", [256, 256]))
    model = WorldModel(INFANTRY_SPEC, hidden=hidden).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training.get("learning_rate", 2.0e-4))
    )
    start_iteration = 0
    elapsed_before = 0.0
    if resume is not None:
        start_iteration, elapsed_before = restore_checkpoint(
            resume, model, optimizer, dataset, resolved_config, device
        )
    else:
        fit_normalizers(
            model,
            dataset,
            samples=int(training.get("normalizer_samples", 100_000)),
            batch_size=int(training["batch_size"]),
            device=device,
        )

    if model.in_norm.count <= 0 or model.out_norm.count <= 0:
        raise RuntimeError("normalization statistics are empty")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    floor = error_floor(device)
    batch_size = int(training["batch_size"])
    iterations = int(training["iterations"])
    checkpoint_every = int(training.get("checkpoint_every", 1000))
    validation_every = int(validation.get("interval", checkpoint_every))
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    log_path = output / "train_log.jsonl"
    validation_path = output / "validation.jsonl"
    started = time.perf_counter()
    transitions_seen = 0

    with log_path.open("a", encoding="utf-8") as log:
        for iteration in range(start_iteration + 1, iterations + 1):
            batch = dataset.sample_windows(
                batch_size, window, split=0, device=device
            )
            errors, _ = rollout_error(model, batch)
            normalizer = torch.maximum(model.out_norm.std(), floor)
            scaled = errors / normalizer
            loss = scaled.square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at iteration {iteration}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"non-finite gradient norm at iteration {iteration}"
                )
            optimizer.step()
            transitions_seen += batch_size * window
            runtime = time.perf_counter() - started
            row = {
                "iteration": iteration,
                "loss": float(loss.detach()),
                "one_step_normalized_mse": float(scaled[:, 0].square().mean().detach()),
                "last_step_normalized_mse": float(scaled[:, -1].square().mean().detach()),
                "gradient_norm": float(grad_norm.detach()),
                "transitions_per_second": transitions_seen / max(runtime, 1.0e-9),
                "elapsed_seconds": elapsed_before + runtime,
                "gpu_peak_bytes": (
                    torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else 0
                ),
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            if iteration == start_iteration + 1 or iteration % 100 == 0:
                print(json.dumps(row), flush=True)

            if iteration % validation_every == 0 or iteration == iterations:
                validation_report = evaluate(
                    model,
                    sources,
                    validation,
                    seed=seed,
                    device=device,
                )
                validation_row = {
                    "iteration": iteration,
                    "elapsed_seconds": elapsed_before + runtime,
                    **validation_report,
                }
                with validation_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(validation_row, ensure_ascii=False) + "\n")

            if iteration % checkpoint_every == 0 or iteration == iterations:
                save_checkpoint(
                    output / f"checkpoint_{iteration:08d}.pt",
                    model,
                    optimizer,
                    iteration,
                    resolved_config,
                    dataset,
                    elapsed_before + runtime,
                )


if __name__ == "__main__":
    main()
