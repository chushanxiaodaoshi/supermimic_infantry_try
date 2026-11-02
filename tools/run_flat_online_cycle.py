"""Run small synchronous Student/World-Model/Actor/Newton flat cycles."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import yaml

from supermimic.infantry.spec import INFANTRY_SPEC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--actor", type=Path)
    parser.add_argument("--world-model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cycles", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    return value


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (REPO / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    cfg = _mapping(
        yaml.safe_load(path.expanduser().resolve().read_text(encoding="utf-8")),
        "online cycle config",
    )
    for section in ("student_collection", "world_model", "actor", "evaluation"):
        _mapping(cfg.get(section), section)
    _mapping(cfg["world_model"].get("training"), "world_model.training")
    _mapping(cfg["world_model"].get("validation"), "world_model.validation")
    replay = cfg.setdefault("replay", {})
    _mapping(replay, "replay")
    teachers = cfg.get("teacher_datasets")
    if not isinstance(teachers, list) or not teachers:
        raise ValueError("teacher_datasets must be a non-empty list")
    if args.actor is not None:
        cfg["initial_actor"] = str(args.actor)
    if args.world_model is not None:
        cfg["initial_world_model"] = str(args.world_model)
    if args.output is not None:
        cfg["output"] = str(args.output)
    if args.cycles is not None:
        cfg["cycles"] = args.cycles
    for name in ("initial_actor", "initial_world_model", "output", "asset_config"):
        if not cfg.get(name):
            raise ValueError(f"{name} is required")
    if int(cfg.get("cycles", 1)) <= 0:
        raise ValueError("cycles must be positive")
    if int(replay.get("recent_actor_cycles", 3)) <= 0:
        raise ValueError("replay.recent_actor_cycles must be positive")
    for name, default in (
        ("teacher_anchor_scale", 0.25),
        ("actor_weight", 1.0),
        ("actor_recency_decay", 0.7),
    ):
        value = float(replay.get(name, default))
        if value <= 0.0 or (name == "actor_recency_decay" and value > 1.0):
            raise ValueError(f"replay.{name} is outside its valid range")
    return cfg


def checkpoint_path(directory: Path, iterations: int) -> Path:
    return directory / f"checkpoint_{iterations:08d}.pt"


def cycle_layout(root: Path, index: int) -> dict[str, Path]:
    cycle = root / f"cycle_{index:04d}"
    return {
        "root": cycle,
        "student": cycle / "student_dataset",
        "world_model": cycle / "world_model",
        "world_model_config": cycle / "world_model.yaml",
        "actor": cycle / "actor",
        "actor_config": cycle / "actor.yaml",
        "evaluation": cycle / "evaluation.json",
    }


def active_replay_sources(
    cfg: dict[str, Any], student_datasets: list[Path]
) -> list[dict[str, Any]]:
    """Build a bounded, recency-weighted replay mixture for one Dyna cycle."""
    replay = cfg.get("replay", {})
    teacher_scale = float(replay.get("teacher_anchor_scale", 0.25))
    sources = [
        {
            **copy.deepcopy(item),
            "weight": float(item.get("weight", 1.0)) * teacher_scale,
            "role": "teacher_anchor",
        }
        for item in cfg["teacher_datasets"]
    ]
    capacity = int(replay.get("recent_actor_cycles", 3))
    recent = student_datasets[-capacity:]
    first_index = len(student_datasets) - len(recent) + 1
    actor_weight = float(replay.get("actor_weight", 1.0))
    decay = float(replay.get("actor_recency_decay", 0.7))
    newest_index = len(student_datasets)
    sources.extend(
        {
            "name": f"actor_cycle_{index:04d}",
            "path": str(path),
            "weight": actor_weight * decay ** (newest_index - index),
            "role": "current_actor_replay",
        }
        for index, path in enumerate(recent, start=first_index)
    )
    return sources


def world_model_config(
    cfg: dict[str, Any],
    layout: dict[str, Path],
    student_datasets: list[Path],
    cycle_index: int,
) -> dict[str, Any]:
    datasets = active_replay_sources(cfg, student_datasets)
    wm = cfg["world_model"]
    return {
        "seed": int(cfg.get("seed", 42)) + cycle_index,
        "device": cfg.get("device", "cuda:0"),
        "output": str(layout["world_model"]),
        "datasets": datasets,
        "model": {"hidden": list(wm.get("hidden", [512, 512]))},
        "training": copy.deepcopy(wm["training"]),
        "validation": copy.deepcopy(wm["validation"]),
    }


def actor_config(
    cfg: dict[str, Any],
    layout: dict[str, Path],
    world_model_checkpoint: Path,
    student_datasets: list[Path],
) -> dict[str, Any]:
    actor_cfg = cfg["actor"]
    base_path = resolve(actor_cfg["base_config"])
    result = _mapping(
        yaml.safe_load(base_path.read_text(encoding="utf-8")), "Actor base config"
    )
    result = copy.deepcopy(result)
    result["output"] = str(layout["actor"])
    result["world_model"]["checkpoint"] = str(world_model_checkpoint)
    result["datasets"] = active_replay_sources(cfg, student_datasets)
    training = result["training"]
    for name in ("iterations", "batch_size", "checkpoint_every", "log_every"):
        if name in actor_cfg:
            training[name] = int(actor_cfg[name])
    if int(result["actor"].get("bptt_horizon", 0)) != 32:
        raise ValueError("online flat Actor must use a 32-step BPTT horizon")
    if bool(result.get("critic", {}).get("enabled", True)):
        raise ValueError("online flat Actor must remain reward-only (critic.enabled=false)")
    return result


def command_plan(
    cfg: dict[str, Any],
    layout: dict[str, Path],
    actor_checkpoint: Path,
    world_model_checkpoint: Path,
    next_actor_checkpoint: Path,
    cycle_index: int,
) -> list[list[str]]:
    python = sys.executable
    collection = cfg["student_collection"]
    evaluation = cfg["evaluation"]
    seed = int(cfg.get("seed", 42)) + cycle_index
    return [
        [
            python,
            str(REPO / "tools/collect_isaaclab_student.py"),
            "--config",
            str(resolve(cfg["asset_config"])),
            "--checkpoint",
            str(actor_checkpoint),
            "--output",
            str(layout["student"]),
            "--steps",
            str(int(collection.get("steps", 400))),
            "--seed",
            str(seed),
            "--max-episode-steps",
            str(int(collection.get("max_episode_steps", 40))),
            "--chunk-steps",
            str(int(collection.get("chunk_steps", 64))),
            "--split-seed",
            str(int(collection.get("split_seed", 2026))),
            "--action-noise-std",
            str(float(collection.get("action_noise_std", 0.0))),
            "--action-noise-hold-steps",
            str(int(collection.get("action_noise_hold_steps", 1))),
            "--command-resample-steps",
            str(int(collection.get("command_resample_steps", 0))),
        ],
        [
            python,
            str(REPO / "tools/train_world_model.py"),
            "--config",
            str(layout["world_model_config"]),
            "--init-from",
            str(world_model_checkpoint),
        ],
        [
            python,
            str(REPO / "tools/train_flat_policy.py"),
            "--config",
            str(layout["actor_config"]),
            "--init-from",
            str(actor_checkpoint),
        ],
        [
            python,
            str(REPO / "tools/evaluate_flat_policy_isaaclab.py"),
            "--config",
            str(resolve(cfg["asset_config"])),
            "--checkpoint",
            str(next_actor_checkpoint),
            "--num-envs",
            "1",
            "--steps",
            str(int(evaluation.get("steps", 200))),
            "--vx",
            str(float(evaluation.get("vx", 1.0))),
            "--yaw",
            str(float(evaluation.get("yaw", 0.0))),
            "--height",
            str(float(evaluation.get("height", 0.2))),
            "--report",
            str(layout["evaluation"]),
        ],
    ]


def write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def write_manifest(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def dataset_semantic_contract(path: Path) -> dict[str, Any]:
    """Return dynamics semantics while ignoring relocatable file identities."""
    manifest_path = path / "manifest.json"
    manifest = _mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        f"dataset manifest {manifest_path}",
    )
    metadata = _mapping(manifest.get("metadata"), f"metadata in {manifest_path}")
    replay = copy.deepcopy(
        _mapping(metadata.get("replay_contract"), f"replay contract in {manifest_path}")
    )
    asset = copy.deepcopy(_mapping(metadata.get("asset"), f"asset in {manifest_path}"))
    replay.pop("source", None)
    replay.pop("source_sha256", None)
    asset.pop("path", None)
    asset.pop("sha256", None)
    return {"replay_contract": replay, "asset": asset}


def validate_dataset_contracts(paths: list[Path]) -> dict[str, Any]:
    reference = dataset_semantic_contract(paths[0])
    for path in paths[1:]:
        actual = dataset_semantic_contract(path)
        if actual != reference:
            raise ValueError(
                f"dataset dynamics/control contract differs from {paths[0]}: {path}"
            )
    return {
        "status": "compatible",
        "datasets": [str(path) for path in paths],
        "comparison": "semantic fields; source/path identities excluded",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_student_dataset(path: Path, expected_actor: Path) -> dict[str, Any]:
    manifest = _mapping(
        json.loads((path / "manifest.json").read_text(encoding="utf-8")),
        "Student dataset manifest",
    )
    metadata = _mapping(manifest.get("metadata"), "Student dataset metadata")
    student = _mapping(metadata.get("student"), "Student provenance")
    expected_sha = sha256_file(expected_actor)
    if student.get("sha256") != expected_sha:
        raise ValueError("Student dataset was not collected by the expected current Actor")
    chunks = sorted(path.glob("chunk_*.pt"))
    if not chunks:
        raise FileNotFoundError(f"Student dataset has no chunks: {path}")
    required = {
        "student_action": INFANTRY_SPEC.policy_action_dim,
        "executed_action": INFANTRY_SPEC.policy_action_dim,
        "applied_control": INFANTRY_SPEC.control_dim,
        "state": INFANTRY_SPEC.state_dim,
        "next_state": INFANTRY_SPEC.state_dim,
    }
    transitions = 0
    episodes: set[tuple[int, int]] = set()
    terminals = 0
    for chunk_path in chunks:
        chunk = torch.load(chunk_path, map_location="cpu", weights_only=True)
        missing = set(required) - chunk.keys()
        if missing:
            raise KeyError(f"Student chunk is missing fields: {sorted(missing)}")
        for name, width in required.items():
            if chunk[name].ndim != 2 or chunk[name].shape[1] != width:
                raise ValueError(
                    f"Student field {name} must have shape [N,{width}], "
                    f"got {tuple(chunk[name].shape)}"
                )
        if not torch.isfinite(chunk["next_state"]).all():
            raise FloatingPointError("Student next_state contains non-finite values")
        terminal = chunk["terminated"].bool()
        if terminal.any() and not torch.isfinite(chunk["next_state"][terminal]).all():
            raise FloatingPointError("terminal next_state contains non-finite values")
        transitions += len(chunk["state"])
        terminals += int(terminal.sum())
        run_ids = chunk.get("run_id", torch.zeros_like(chunk["episode_id"]))
        episodes.update(
            (int(run_id), int(episode_id))
            for run_id, episode_id in zip(run_ids, chunk["episode_id"], strict=True)
        )
    if transitions != int(manifest["transitions"]):
        raise ValueError("Student audited transition count differs from manifest")
    return {
        "status": "valid",
        "actor_checkpoint": str(expected_actor),
        "actor_sha256": expected_sha,
        "transitions": transitions,
        "episodes": len(episodes),
        "terminal_transitions": terminals,
        "recorded_dimensions": required,
        "terminal_next_state": "recorded_before_reset_and_finite",
    }


def _last_jsonl(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not lines:
        raise ValueError(f"empty JSONL report: {path}")
    return _mapping(json.loads(lines[-1]), f"last row in {path}")


def summarize_cycle(
    layout: dict[str, Path], teacher_names: list[str], student_name: str
) -> dict[str, Any]:
    validation = _last_jsonl(layout["world_model"] / "validation.jsonl")
    actor = _last_jsonl(layout["actor"] / "train_log.jsonl")
    evaluation = _mapping(
        json.loads(layout["evaluation"].read_text(encoding="utf-8")),
        "real evaluation report",
    )
    sources = _mapping(validation.get("sources"), "World Model validation sources")
    student = _mapping(sources.get(student_name), f"validation source {student_name}")
    training_model_survival = float(actor["survival"])
    model_survival_value = evaluation.get("model_survival_horizon_mean")
    model_survival = (
        None if model_survival_value is None else float(model_survival_value)
    )
    real_survival_value = evaluation.get("survival_horizon_fraction")
    real_survival = None if real_survival_value is None else float(real_survival_value)
    return {
        "student_data": student["split_summary"],
        "teacher_validation": {
            name: _mapping(sources.get(name), f"validation source {name}")["metrics"]
            for name in teacher_names
        },
        "student_validation": student["metrics"],
        "student_32_step_prediction": student["metrics"].get("32"),
        "model_survival_32": model_survival,
        "training_mixture_model_survival_32": training_model_survival,
        "real_survival_32": real_survival,
        "survival_gap_absolute": (
            None
            if model_survival is None or real_survival is None
            else abs(model_survival - real_survival)
        ),
        "real_mean_episode_length": evaluation.get("mean_episode_length"),
        "real_fall_fraction": evaluation.get("fall_fraction"),
        "forward_velocity_mae": evaluation.get("forward_velocity_mae"),
        "yaw_velocity_mae": evaluation.get("yaw_velocity_mae"),
        "wm_input_rms": actor.get("wm_input_rms"),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args)
    root = resolve(cfg["output"])
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"online output must be empty: {root}")
    initial_actor = resolve(cfg["initial_actor"])
    initial_world_model = resolve(cfg["initial_world_model"])
    for label, path in (
        ("initial Actor", initial_actor),
        ("initial World Model", initial_world_model),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "status": "planned" if args.dry_run else "running",
        "config": str(args.config.expanduser().resolve()),
        "initial_actor": str(initial_actor),
        "initial_world_model": str(initial_world_model),
        "cycles": [],
    }
    manifest_path = root / "online_manifest.json"
    actor_checkpoint = initial_actor
    world_model_checkpoint = initial_world_model
    student_datasets: list[Path] = []

    try:
        for cycle_index in range(1, int(cfg.get("cycles", 1)) + 1):
            layout = cycle_layout(root, cycle_index)
            layout["root"].mkdir(parents=True, exist_ok=False)
            student_datasets.append(layout["student"])
            wm_cfg = world_model_config(
                cfg, layout, student_datasets, cycle_index
            )
            wm_iterations = int(wm_cfg["training"]["iterations"])
            next_world_model = checkpoint_path(
                layout["world_model"], wm_iterations
            )
            pi_cfg = actor_config(
                cfg, layout, next_world_model, student_datasets
            )
            active_sources = active_replay_sources(cfg, student_datasets)
            actor_iterations = int(pi_cfg["training"]["iterations"])
            next_actor = checkpoint_path(layout["actor"], actor_iterations)
            write_yaml(layout["world_model_config"], wm_cfg)
            write_yaml(layout["actor_config"], pi_cfg)
            commands = command_plan(
                cfg,
                layout,
                actor_checkpoint,
                world_model_checkpoint,
                next_actor,
                cycle_index,
            )
            row = {
                "cycle": cycle_index,
                "status": "planned",
                "student_dataset": str(layout["student"]),
                "world_model_checkpoint": str(next_world_model),
                "actor_checkpoint": str(next_actor),
                "evaluation": str(layout["evaluation"]),
                "active_replay_sources": active_sources,
                "commands": commands,
            }
            manifest["cycles"].append(row)
            write_manifest(manifest_path, manifest)
            if args.dry_run:
                actor_checkpoint = next_actor
                world_model_checkpoint = next_world_model
                continue
            for stage, command in zip(
                ("collect_student", "update_world_model", "update_actor", "evaluate"),
                commands,
                strict=True,
            ):
                row["status"] = stage
                write_manifest(manifest_path, manifest)
                subprocess.run(command, cwd=REPO, check=True)
                if stage == "collect_student":
                    row["student_audit"] = audit_student_dataset(
                        layout["student"], actor_checkpoint
                    )
                    row["dataset_contract"] = validate_dataset_contracts(
                        [resolve(item["path"]) for item in active_sources]
                    )
                    write_manifest(manifest_path, manifest)
                elif stage == "evaluate":
                    row["evaluation_metrics"] = json.loads(
                        layout["evaluation"].read_text(encoding="utf-8")
                    )
                    row["summary"] = summarize_cycle(
                        layout,
                        [item["name"] for item in cfg["teacher_datasets"]],
                        f"actor_cycle_{cycle_index:04d}",
                    )
            row["status"] = "complete"
            actor_checkpoint = next_actor
            world_model_checkpoint = next_world_model
            write_manifest(manifest_path, manifest)
        manifest["status"] = "planned" if args.dry_run else "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        write_manifest(manifest_path, manifest)
        raise
    write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
