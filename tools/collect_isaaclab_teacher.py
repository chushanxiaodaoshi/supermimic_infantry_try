"""Collect ONNX Teacher transitions with Isaac Lab 3 and MuJoCo-Warp."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import yaml

from supermimic.data.chunks import ChunkWriter
from supermimic.infantry.replay_contract import (
    ReplayContract,
    extract_mjcf_physics,
    validate_onnx_contract,
)
from supermimic.infantry.spec import INFANTRY_SPEC


def _parser(with_launcher: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--mjcf", type=Path)
    parser.add_argument("--usd", type=Path)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--validate-only", action="store_true")
    if with_launcher:
        from isaaclab.app import add_launcher_args

        add_launcher_args(parser)
    return parser


def _resolve(value: str | Path | None, base: Path) -> Path | None:
    if value is None:
        return None
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("collection config must be a mapping")
    base = config_path.parent
    asset = config.setdefault("asset", {})
    teacher = config.setdefault("teacher", {})
    collection = config.setdefault("collection", {})
    overrides = (
        (asset, "replay_source", args.replay),
        (asset, "mjcf", args.mjcf),
        (asset, "usd", args.usd),
        (teacher, "onnx", args.onnx),
        (collection, "output", args.output),
    )
    for section, key, value in overrides:
        if value is not None:
            section[key] = value
    if args.steps is not None:
        if args.steps <= 0:
            raise ValueError("--steps must be positive")
        collection["steps"] = args.steps
    for section, keys in (
        (asset, ("replay_source", "mjcf", "usd")),
        (teacher, ("onnx",)),
        (collection, ("output",)),
    ):
        for key in keys:
            section[key] = _resolve(section.get(key), base)
    required = {
        "asset.replay_source": asset["replay_source"],
        "asset.mjcf": asset["mjcf"],
        "teacher.onnx": teacher["onnx"],
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"missing required config paths: {missing}")
    if not args.validate_only and (
        asset["usd"] is None or collection["output"] is None
    ):
        raise ValueError("asset.usd and collection.output are required for collection")
    config["config_path"] = config_path
    return config


def validate_config(config: dict[str, Any]) -> tuple[ReplayContract, dict, dict]:
    asset, teacher = config["asset"], config["teacher"]
    for label, path in (
        ("replay source", asset["replay_source"]),
        ("MJCF", asset["mjcf"]),
        ("Teacher", teacher["onnx"]),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")
    contract = ReplayContract.from_replay(asset["replay_source"])
    teacher_info = validate_onnx_contract(teacher["onnx"], contract)
    physics = extract_mjcf_physics(asset["mjcf"], contract)
    if abs(physics["timestep"] * 4 - INFANTRY_SPEC.control_dt) > 1.0e-9:
        raise ValueError(
            "the pinned Isaac Lab adapter uses four physics substeps; MJCF timestep "
            f"{physics['timestep']} does not yield control dt {INFANTRY_SPEC.control_dt}"
        )
    skill = teacher.get("skill")
    if skill not in {"flat", "rotation", "jump"}:
        raise ValueError("teacher.skill must be flat, rotation, or jump")
    return contract, teacher_info, physics


class OnnxTeacher:
    def __init__(self, path: Path, provider: str):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "install onnxruntime-gpu in the collection image"
            ) from exc
        available = set(ort.get_available_providers())
        if provider not in available:
            raise RuntimeError(
                f"ONNX provider {provider!r} unavailable; found {sorted(available)}"
            )
        self.session = ort.InferenceSession(str(path), providers=[provider])
        self.input_name = self.session.get_inputs()[0].name
        self.fixed_batch = self.session.get_inputs()[0].shape[0] == 1

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        value = observation.detach().cpu().numpy()
        if self.fixed_batch and value.shape[0] != 1:
            outputs = [
                self.session.run(None, {self.input_name: row[None]})[0][0]
                for row in value
            ]
            import numpy as np

            result = np.stack(outputs)
        else:
            result = self.session.run(None, {self.input_name: value})[0]
        return torch.as_tensor(
            result, device=observation.device, dtype=observation.dtype
        )


def _git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def collect(
    args: argparse.Namespace, config: dict[str, Any], contract, teacher_info, physics
) -> None:
    from isaaclab.app import launch_simulation

    from supermimic.infantry.isaaclab_runtime import InfantryCollectionSim, make_sim_cfg

    asset, teacher_cfg, collection = (
        config["asset"],
        config["teacher"],
        config["collection"],
    )
    output = collection["output"]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"dataset destination must be empty: {output}")
    num_envs = int(collection.get("num_envs", 1))
    steps = int(collection.get("steps", 1000))
    seed = int(collection.get("seed", 42))
    device = str(collection.get("device", "cuda:0"))
    torch.manual_seed(seed)
    sim_cfg = make_sim_cfg(device, INFANTRY_SPEC.control_dt)
    with launch_simulation(sim_cfg, args):
        env = InfantryCollectionSim(
            sim_cfg,
            num_envs=num_envs,
            usd_path=asset["usd"],
            contract=contract,
            context=torch.tensor(physics["context"], dtype=torch.float32),
            controller=physics["controller"],
            command_ranges={
                key: tuple(value) for key, value in config["commands"].items()
            },
            max_episode_steps=int(collection.get("max_episode_steps", 1000)),
            seed=seed,
        )
        policy = OnnxTeacher(
            teacher_cfg["onnx"],
            str(collection.get("onnx_provider", "CPUExecutionProvider")),
        )
        run_uuid = uuid.uuid4()
        run_id = run_uuid.int & 0x7FFF_FFFF_FFFF_FFFF
        metadata = {
            "backend": "isaaclab3-newton-mujoco-warp",
            "isaaclab_commit": "46d6cc3d7b115906fad12efd372fbf69f4e3d3ed",
            "supermimic_commit": _git_commit(REPO),
            "collection_run_uuid": str(run_uuid),
            "teacher": teacher_info
            | {
                "name": teacher_cfg["name"],
                "skill": teacher_cfg["skill"],
                "teacher_id": int(teacher_cfg["id"]),
                "task_id": int(teacher_cfg["task_id"]),
            },
            "replay_contract": contract.to_dict(),
            "asset": physics,
            "randomization": "disabled; fixed context stored on every transition",
            "config": json.loads(json.dumps(config, default=str)),
            "reward_usage": "statistics_only_not_world_model_loss",
        }
        writer = ChunkWriter(
            output,
            INFANTRY_SPEC,
            chunk_steps=int(collection.get("chunk_steps", 128)),
            split_seed=int(collection.get("split_seed", 2026)),
            metadata=metadata,
        )
        env_ids = torch.arange(num_envs, device=device)
        noise_std = float(collection.get("action_noise_std", 0.0))
        noise_rng = torch.Generator(device=device).manual_seed(seed + 1_000_003)
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                for _ in range(steps):
                    current, history, teacher_input = env.observation()
                    teacher_action = policy(teacher_input)
                    executed = teacher_action
                    if noise_std:
                        executed = executed + noise_std * torch.randn(
                            executed.shape,
                            device=device,
                            generator=noise_rng,
                        )
                    transition = env.step(executed, current, history)
                    done = transition["terminated"] | transition["truncated"]
                    episode_id = env_ids + num_envs * env.episodes
                    vertical_velocity = transition["next_state"][:, 9]
                    phase = torch.where(
                        vertical_velocity > 0.15,
                        1,
                        torch.where(vertical_velocity < -0.15, 2, 0),
                    )
                    writer.append(
                        transition
                        | {
                            "teacher_action": teacher_action,
                            "reward": torch.zeros(num_envs, device=device),
                            "dt": torch.full(
                                (num_envs,), INFANTRY_SPEC.control_dt, device=device
                            ),
                            "teacher_id": torch.full_like(
                                episode_id, int(teacher_cfg["id"])
                            ),
                            "task_id": torch.full_like(
                                episode_id, int(teacher_cfg["task_id"])
                            ),
                            "run_id": torch.full_like(episode_id, run_id),
                            "episode_id": episode_id,
                            "motion_phase": phase,
                        }
                    )
                    done_ids = done.nonzero(as_tuple=False).flatten()
                    if len(done_ids):
                        env.episodes[done_ids] += 1
                        env.reset(done_ids)
        finally:
            duration = time.perf_counter() - started
            metadata["collection_runtime"] = {
                "seconds": duration,
                "transitions": writer.transitions,
                "transitions_per_second": writer.transitions / max(duration, 1.0e-9),
                "gpu_peak_bytes": (
                    torch.cuda.max_memory_allocated(device)
                    if torch.cuda.is_available()
                    else 0
                ),
            }
            writer.close()


def main() -> None:
    preliminary, _ = _parser().parse_known_args()
    config = load_config(preliminary)
    contract, teacher_info, physics = validate_config(config)
    summary = {
        "status": "contract_valid",
        "skill": config["teacher"]["skill"],
        "teacher": teacher_info,
        "asset": physics,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if preliminary.validate_only:
        return
    args = _parser(with_launcher=True).parse_args()
    config = load_config(args)
    collect(args, config, contract, teacher_info, physics)


if __name__ == "__main__":
    main()
