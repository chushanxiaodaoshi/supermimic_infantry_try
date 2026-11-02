"""Collect real Isaac Lab/Newton transitions from a trained flat Student."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch

from supermimic.data.chunks import ChunkWriter
from supermimic.infantry.flat_mdp import (
    FlatRewardWeights,
    bounded_policy_action,
    flat_reward,
)
from supermimic.infantry.replay_contract import (
    ReplayContract,
    extract_mjcf_physics,
    sha256_file,
)
from supermimic.infantry.spec import INFANTRY_SPEC
from tools.evaluate_flat_policy_isaaclab import load_actor, load_asset_config


def make_parser(with_launcher: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episode-steps", type=int, default=40)
    parser.add_argument("--chunk-steps", type=int, default=128)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--action-noise-std", type=float, default=0.0)
    if with_launcher:
        from isaaclab.app import add_launcher_args

        add_launcher_args(parser)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("steps", "max_episode_steps", "chunk_steps"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.seed < 0 or args.split_seed < 0:
        raise ValueError("seeds must be non-negative")
    if args.action_noise_std < 0.0:
        raise ValueError("--action-noise-std must be non-negative")


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def git_dirty() -> bool:
    return bool(
        subprocess.check_output(
            ["git", "-C", str(REPO), "status", "--porcelain"], text=True
        ).strip()
    )


def collect(args: argparse.Namespace, config: dict[str, Any]) -> None:
    from isaaclab.app import launch_simulation

    from supermimic.infantry.isaaclab_runtime import (
        InfantryCollectionSim,
        make_sim_cfg,
    )

    validate_args(args)
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"dataset destination must be empty: {output}")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Student checkpoint: {checkpoint}")

    asset = config["asset"]
    contract = ReplayContract.from_replay(asset["replay_source"])
    physics = extract_mjcf_physics(asset["mjcf"], contract)
    collection_cfg = config.get("collection", {})
    device = torch.device(collection_cfg.get("device", "cuda:0"))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    actor, actor_payload = load_actor(checkpoint, device)
    actor_cfg = actor_payload["config"]["actor"]
    action_limit_value = actor_cfg.get("output_limit")
    action_limit = (
        None if action_limit_value is None else float(action_limit_value)
    )
    policy_reward = actor_payload["config"].get("reward", {})
    reward_weights = FlatRewardWeights(**policy_reward.get("weights", {}))
    joint_lower = torch.tensor(
        policy_reward.get("joint_lower", [-100.0] * 4), device=device
    )
    joint_upper = torch.tensor(
        policy_reward.get("joint_upper", [100.0] * 4), device=device
    )
    command_ranges = {
        key: tuple(float(value) for value in bounds)
        for key, bounds in config["commands"].items()
    }
    sim_cfg = make_sim_cfg(str(device), INFANTRY_SPEC.control_dt)

    with launch_simulation(sim_cfg, args):
        env = InfantryCollectionSim(
            sim_cfg,
            num_envs=1,
            usd_path=asset["usd"],
            contract=contract,
            context=torch.tensor(physics["context"], dtype=torch.float32),
            controller=physics["controller"],
            command_ranges=command_ranges,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed,
        )
        run_uuid = uuid.uuid4()
        run_id = run_uuid.int & 0x7FFF_FFFF_FFFF_FFFF
        metadata = {
            "backend": "isaaclab3-newton-mujoco-warp",
            "source": "student_policy_real_simulation",
            "supermimic_commit": git_commit(),
            "supermimic_dirty": git_dirty(),
            "collection_run_uuid": str(run_uuid),
            "student": {
                "checkpoint": str(checkpoint),
                "sha256": sha256_file(checkpoint),
                "iteration": int(actor_payload.get("iteration", 0)),
            },
            "replay_contract": contract.to_dict(),
            "asset": physics,
            "randomization": (
                "disabled; sampled command and executed action are recorded; "
                f"action_noise_std={args.action_noise_std}"
            ),
            "collection": {
                "steps": args.steps,
                "num_envs": 1,
                "max_episode_steps": args.max_episode_steps,
                "seed": args.seed,
                "split_seed": args.split_seed,
                "action_noise_std": args.action_noise_std,
                "command_ranges": command_ranges,
            },
            "reward_usage": "diagnostic_only_not_world_model_loss",
        }
        writer = ChunkWriter(
            output,
            INFANTRY_SPEC,
            chunk_steps=args.chunk_steps,
            split_seed=args.split_seed,
            metadata=metadata,
        )
        noise_rng = torch.Generator(device=device).manual_seed(args.seed + 1_000_003)
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                for _ in range(args.steps):
                    current, history, _ = env.observation()
                    student_action = bounded_policy_action(
                        actor(history), action_limit
                    ).clamp(*contract.action_clip)
                    executed_action = student_action
                    if args.action_noise_std:
                        executed_action = executed_action + args.action_noise_std * torch.randn(
                            executed_action.shape,
                            device=device,
                            generator=noise_rng,
                        )
                    transition = env.step(executed_action, current, history)
                    state = env.codec.unflatten(transition["state"])
                    next_state = env.codec.unflatten(transition["next_state"])
                    previous_action = history[:, -1, -INFANTRY_SPEC.policy_action_dim :]
                    previous_previous_action = history[
                        :, -2, -INFANTRY_SPEC.policy_action_dim :
                    ]
                    reward, _ = flat_reward(
                        state,
                        next_state,
                        transition["command"],
                        transition["executed_action"],
                        previous_action,
                        previous_previous_action,
                        transition["applied_control"][:, :6],
                        transition["context"],
                        joint_lower=joint_lower,
                        joint_upper=joint_upper,
                        weights=reward_weights,
                    )
                    fell_over = next_state[1][:, 2, 2] < 0.5
                    terminated = transition["terminated"] | fell_over
                    done = terminated | transition["truncated"]
                    episode_id = env.episodes.clone()
                    vertical_velocity = transition["next_state"][:, 9]
                    phase = torch.where(
                        vertical_velocity > 0.15,
                        1,
                        torch.where(vertical_velocity < -0.15, 2, 0),
                    )
                    writer.append(
                        transition
                        | {
                            "student_action": student_action,
                            "reward": reward,
                            "terminated": terminated,
                            "dt": torch.full(
                                (1,), INFANTRY_SPEC.control_dt, device=device
                            ),
                            "teacher_id": torch.full(
                                (1,), -1, dtype=torch.long, device=device
                            ),
                            "task_id": torch.zeros(1, dtype=torch.long, device=device),
                            "run_id": torch.full(
                                (1,), run_id, dtype=torch.long, device=device
                            ),
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
                    if device.type == "cuda"
                    else 0
                ),
            }
            writer.close()


def main() -> None:
    preliminary, _ = make_parser().parse_known_args()
    validate_args(preliminary)
    config = load_asset_config(preliminary.config)
    args = make_parser(with_launcher=True).parse_args()
    collect(args, config)


if __name__ == "__main__":
    main()
