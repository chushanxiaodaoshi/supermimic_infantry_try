"""Evaluate a trained flat Actor in Isaac Lab 3 + Newton/MuJoCo-Warp."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import yaml

from supermimic.core.networks import UnifiedActor
from supermimic.infantry.flat_mdp import (
    FlatRewardWeights,
    bounded_policy_action,
    flat_reward,
)
from supermimic.infantry.replay_contract import ReplayContract, extract_mjcf_physics
from supermimic.infantry.spec import INFANTRY_SPEC


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--vx", type=float)
    parser.add_argument("--yaw", type=float)
    parser.add_argument("--height", type=float)
    parser.add_argument("--report", type=Path)
    from isaaclab.app import add_launcher_args

    add_launcher_args(parser)
    return parser


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_asset_config(path: Path) -> dict[str, Any]:
    config_path = path.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("collection config must be a mapping")
    base = config_path.parent
    asset = config.get("asset")
    if not isinstance(asset, dict):
        raise TypeError("collection config asset must be a mapping")
    for name in ("replay_source", "mjcf", "usd"):
        if not asset.get(name):
            raise ValueError(f"asset.{name} is required")
        asset[name] = _resolve(asset[name], base)
        if not asset[name].is_file():
            raise FileNotFoundError(f"asset.{name}: {asset[name]}")
    commands = config.get("commands")
    if not isinstance(commands, dict):
        raise TypeError("collection config commands must be a mapping")
    return config


def load_actor(path: Path, device: torch.device) -> tuple[UnifiedActor, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("robot_spec") != asdict(INFANTRY_SPEC):
        raise ValueError("policy checkpoint uses an incompatible robot spec")
    config = payload.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("actor"), dict):
        raise TypeError("policy checkpoint has no resolved actor config")
    actor_cfg = config["actor"]
    actor = UnifiedActor(
        INFANTRY_SPEC.actor_obs_dim,
        INFANTRY_SPEC.policy_action_dim,
        history=int(actor_cfg.get("history", 5)),
        hidden=tuple(int(value) for value in actor_cfg.get("hidden", (512, 256, 128))),
    ).to(device)
    actor.load_state_dict(payload["actor"], strict=True)
    actor.eval()
    return actor, payload


def main() -> None:
    args = make_parser().parse_args()
    if args.steps <= 0 or args.num_envs <= 0:
        raise ValueError("steps and num-envs must be positive")
    config = load_asset_config(args.config)
    asset = config["asset"]
    contract = ReplayContract.from_replay(asset["replay_source"])
    physics = extract_mjcf_physics(asset["mjcf"], contract)
    device = torch.device(config.get("collection", {}).get("device", "cuda:0"))
    torch.manual_seed(args.seed)
    actor, checkpoint = load_actor(args.checkpoint.expanduser().resolve(), device)
    action_limit_value = checkpoint["config"]["actor"].get("output_limit")
    action_limit = (
        None if action_limit_value is None else float(action_limit_value)
    )
    policy_reward = checkpoint["config"].get("reward", {})
    reward_weights = FlatRewardWeights(**policy_reward.get("weights", {}))
    joint_lower = torch.tensor(
        policy_reward.get("joint_lower", [-100.0] * 4), device=device
    )
    joint_upper = torch.tensor(
        policy_reward.get("joint_upper", [100.0] * 4), device=device
    )

    from isaaclab.app import launch_simulation

    from supermimic.infantry.isaaclab_runtime import InfantryCollectionSim, make_sim_cfg

    command_ranges = {key: tuple(value) for key, value in config["commands"].items()}
    overrides = {
        "lin_vel_x": args.vx,
        "ang_vel_z": args.yaw,
        "height": args.height,
    }
    for name, value in overrides.items():
        if value is not None:
            command_ranges[name] = (float(value), float(value))

    sim_cfg = make_sim_cfg(str(device), INFANTRY_SPEC.control_dt)
    with launch_simulation(sim_cfg, args):
        env = InfantryCollectionSim(
            sim_cfg,
            num_envs=args.num_envs,
            usd_path=asset["usd"],
            contract=contract,
            context=torch.tensor(physics["context"], dtype=torch.float32),
            controller=physics["controller"],
            command_ranges=command_ranges,
            max_episode_steps=int(config.get("collection", {}).get("max_episode_steps", 1000)),
            seed=args.seed,
        )
        episode_return = torch.zeros(args.num_envs, device=device)
        episode_length = torch.zeros(args.num_envs, dtype=torch.long, device=device)
        previous_action = torch.zeros(args.num_envs, 6, device=device)
        previous_previous_action = torch.zeros_like(previous_action)
        completed_returns: list[torch.Tensor] = []
        completed_lengths: list[torch.Tensor] = []
        completed_falls = 0
        forward_errors: list[torch.Tensor] = []
        yaw_errors: list[torch.Tensor] = []
        start = time.perf_counter()

        with torch.inference_mode():
            for _ in range(args.steps):
                current, history, _ = env.observation()
                action = bounded_policy_action(
                    actor(history), action_limit
                ).clamp(*contract.action_clip)
                transition = env.step(action, current, history)
                state = env.codec.unflatten(transition["state"])
                next_state = env.codec.unflatten(transition["next_state"])
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
                rotation_t = next_state[1].transpose(-1, -2)
                body_linear = (
                    rotation_t @ next_state[2].unsqueeze(-1)
                ).squeeze(-1)
                body_angular = (
                    rotation_t @ next_state[3].unsqueeze(-1)
                ).squeeze(-1)
                forward_errors.append(
                    (body_linear[:, 0] - transition["command"][:, 0]).abs()
                )
                yaw_errors.append(
                    (body_angular[:, 2] - transition["command"][:, 1]).abs()
                )
                episode_return += reward
                episode_length += 1
                fell_over = next_state[1][:, 2, 2] < 0.5
                done = transition["terminated"] | transition["truncated"] | fell_over
                done_ids = done.nonzero(as_tuple=False).flatten()
                if len(done_ids):
                    completed_returns.append(episode_return[done_ids].cpu())
                    completed_lengths.append(episode_length[done_ids].cpu())
                    completed_falls += int(
                        (transition["terminated"][done_ids] | fell_over[done_ids]).sum()
                    )
                    env.episodes[done_ids] += 1
                    env.reset(done_ids)
                    episode_return[done_ids] = 0.0
                    episode_length[done_ids] = 0
                    previous_action[done_ids] = 0.0
                    previous_previous_action[done_ids] = 0.0
                active = ~done
                previous_previous_action[active] = previous_action[active]
                previous_action[active] = transition["executed_action"][active]

        duration = time.perf_counter() - start
        returns = torch.cat(completed_returns) if completed_returns else torch.empty(0)
        lengths = torch.cat(completed_lengths) if completed_lengths else torch.empty(0)
        episodes = len(returns)
        report = {
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "checkpoint_iteration": int(checkpoint["iteration"]),
            "backend": "isaaclab3-newton-mujoco-warp",
            "steps_per_environment": args.steps,
            "num_environments": args.num_envs,
            "completed_episodes": episodes,
            "fall_episodes": completed_falls,
            "fall_fraction": completed_falls / max(episodes, 1),
            "mean_episode_return": float(returns.mean()) if episodes else None,
            "mean_episode_length": float(lengths.float().mean()) if episodes else None,
            "forward_velocity_mae": float(torch.cat(forward_errors).mean()),
            "yaw_velocity_mae": float(torch.cat(yaw_errors).mean()),
            "wall_seconds": duration,
            "simulated_transitions_per_second": args.steps * args.num_envs / duration,
        }
        text = json.dumps(report, ensure_ascii=False, indent=2)
        print(text, flush=True)
        if args.report is not None:
            report_path = args.report.expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
