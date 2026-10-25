"""Collect real mjlab teacher transitions into the shared dataset contract.

This tool intentionally runs inside the existing mjlab environment.  It does
not import or modify mjlab source files and disables auto-reset so terminal
states are recorded before an episode is reset.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch

from supermimic.data.chunks import ChunkWriter
from supermimic.infantry.spec import (
    GAS_SPRING_TENDON_NAMES,
    INFANTRY_SPEC,
    POLICY_ACTION_NAMES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjlab-root", type=Path, required=True)
    parser.add_argument("--task", choices=("wheelleg-flat",), default="wheelleg-flat")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--chunk-steps", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def configure_collection(env_cfg: Any, *, seed: int) -> dict[str, Any]:
    """Constrain collection to dynamics represented by the current WM inputs."""
    if env_cfg.scene.terrain.terrain_type != "plane":
        raise ValueError("the current world model has no terrain input; use a plane")
    env_cfg.seed = seed
    disabled = ("push_robots", "body_torque")
    for name in disabled:
        env_cfg.events.pop(name, None)
    # The context contains base mass/inertia but not independently varying limb
    # inertia. Keep base added-mass randomization, fix all per-body scale factors.
    env_cfg.events["body_mass_inertia"].params["inertia_scale_range"] = (1.0, 1.0)
    wheels = ("Left_wheel_collision", "Right_wheel_collision")
    for name in ("wheel_friction", "restitution"):
        env_cfg.events[name].params["asset_cfg"].geom_names = wheels
    return {
        "profile": "flat_observed_dynamics_v1",
        "disabled_events": list(disabled),
        "inertia_scale_range": [1.0, 1.0],
        "randomized_contact_geoms": list(wheels),
        "seed": seed,
    }


def _state(raw_env, robot) -> torch.Tensor:
    pose = robot.data.root_link_pose_w
    velocity = robot.data.root_link_vel_w
    position = pose[:, :3] - raw_env.scene.env_origins
    return torch.cat(
        (
            position,
            pose[:, 3:7],
            velocity[:, :3],
            velocity[:, 3:6],
            robot.data.joint_pos,
            robot.data.joint_vel,
        ),
        dim=-1,
    )


def frame_major_history(value: torch.Tensor, history: int = 5) -> torch.Tensor:
    """Convert mjlab's term-major history to (batch, oldest-first time, obs)."""
    widths = (3, 3, 3, 4, 6, 6)
    if value.shape[-1] != history * sum(widths):
        raise ValueError("unexpected infantry observation history width")
    terms = value.split(tuple(history * width for width in widths), dim=-1)
    return torch.cat(
        [
            term.reshape(value.shape[0], history, width)
            for term, width in zip(terms, widths, strict=True)
        ],
        dim=-1,
    )


class PhysicsContext:
    """Resolve model indices once, then read exact per-environment parameters."""

    def __init__(self, raw_env, robot):
        self.raw_env = raw_env
        self.robot = robot
        device = raw_env.device
        wheel_local, _ = robot.find_geoms(
            ("Left_wheel_collision", "Right_wheel_collision"), preserve_order=True
        )
        base_local, _ = robot.find_bodies(("base_link",), preserve_order=True)
        actuator_local, _ = robot.find_actuators(
            POLICY_ACTION_NAMES, preserve_order=True
        )
        joint_local, _ = robot.find_joints((".*",), preserve_order=True)
        tendon_local, _ = robot.find_tendons(
            GAS_SPRING_TENDON_NAMES, preserve_order=True
        )
        self.wheel_geoms = robot.indexing.geom_ids[
            torch.tensor(wheel_local, device=device)
        ]
        self.base_body = robot.indexing.body_ids[
            torch.tensor(base_local, device=device)
        ]
        self.policy_ctrl = robot.indexing.ctrl_ids[
            torch.tensor(actuator_local, device=device)
        ]
        self.joint_dofs = robot.indexing.joint_v_adr[
            torch.tensor(joint_local, device=device)
        ]
        self.tendon_local = torch.tensor(tendon_local, device=device)

    def read(self) -> torch.Tensor:
        model = self.raw_env.sim.model
        friction = model.geom_friction[:, self.wheel_geoms, 0]
        mass = model.body_mass[:, self.base_body]
        inertia = model.body_inertia[:, self.base_body].flatten(start_dim=1)
        ipos = model.body_ipos[:, self.base_body].flatten(start_dim=1)
        gain = model.actuator_gainprm[:, self.policy_ctrl, 0]
        position_bias = model.actuator_biasprm[:, self.policy_ctrl, 1]
        velocity_bias = model.actuator_biasprm[:, self.policy_ctrl, 2]
        force_min = model.actuator_forcerange[:, self.policy_ctrl, 0]
        force_max = model.actuator_forcerange[:, self.policy_ctrl, 1]
        spring = self.robot.data.tendon_effort_target[:, self.tendon_local]
        damping = model.dof_damping[:, self.joint_dofs]
        frictionloss = model.dof_frictionloss[:, self.joint_dofs]
        solref = model.geom_solref[:, self.wheel_geoms, :2].flatten(start_dim=1)
        context = torch.cat(
            (
                friction,
                mass,
                inertia,
                ipos,
                gain,
                position_bias,
                velocity_bias,
                force_min,
                force_max,
                spring,
                damping,
                frictionloss,
                solref,
            ),
            dim=-1,
        )
        if context.shape[-1] != INFANTRY_SPEC.context_dim:
            raise RuntimeError(
                f"context width {context.shape[-1]} != {INFANTRY_SPEC.context_dim}"
            )
        return context


def _applied_control(raw_env, robot) -> torch.Tensor:
    processed = []
    for name in raw_env.action_manager.active_terms:
        term = raw_env.action_manager.get_term(name)
        processed.append(term._processed_actions)  # exact post-scale/offset value
    tendon_ids, _ = robot.find_tendons(GAS_SPRING_TENDON_NAMES, preserve_order=True)
    spring = robot.data.tendon_effort_target[:, tendon_ids]
    control = torch.cat((*processed, spring), dim=-1)
    if control.shape[-1] != INFANTRY_SPEC.control_dim:
        raise RuntimeError("unexpected applied-control layout")
    return control


def main() -> None:
    args = parse_args()
    mjlab_root = args.mjlab_root.expanduser().resolve()
    sys.path.insert(0, str(mjlab_root / "src"))
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"dataset destination must be empty: {args.output}")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks import wheelleg  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    torch.manual_seed(args.seed)
    env_cfg = load_env_cfg(args.task, play=False)
    collection_profile = configure_collection(env_cfg, seed=args.seed)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.auto_reset = False
    agent_cfg = load_rl_cfg(args.task)
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=args.device,
    )
    policy = runner.get_inference_policy(device=args.device)
    robot = raw_env.scene["robot"]
    if tuple(robot.joint_names) != INFANTRY_SPEC.joint_names:
        raise RuntimeError(f"joint order mismatch: {tuple(robot.joint_names)}")
    context_view = PhysicsContext(raw_env, robot)
    writer = ChunkWriter(
        args.output,
        INFANTRY_SPEC,
        chunk_steps=args.chunk_steps,
        split_seed=args.seed,
        metadata={
            "source": "mjlab",
            "task": args.task,
            "checkpoint": str(checkpoint),
            "mjlab_commit": subprocess.check_output(
                ["git", "-C", str(mjlab_root), "rev-parse", "HEAD"], text=True
            ).strip(),
            "control_dt": raw_env.step_dt,
            "collection_profile": collection_profile,
            "observation_history_layout": "frame_major_oldest_first",
        },
    )
    episode_count = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)
    env_index = torch.arange(args.num_envs, dtype=torch.long, device=args.device)
    elapsed = torch.zeros(args.num_envs, device=args.device)

    try:
        with torch.inference_mode():
            observations = env.get_observations()
            for _ in range(args.steps):
                state = _state(raw_env, robot)
                command = raw_env.command_manager.get_command("twist")[:, (0, 2, 3)]
                action = policy(observations)
                executed_action = action
                if agent_cfg.clip_actions is not None:
                    executed_action = action.clamp(
                        -agent_cfg.clip_actions, agent_cfg.clip_actions
                    )
                if torch.count_nonzero(robot.data.body_external_wrench):
                    raise RuntimeError(
                        "unrecorded external wrench in collection environment"
                    )
                context = context_view.read()
                _, reward, terminated, truncated, _ = raw_env.step(executed_action)
                next_state = _state(raw_env, robot)
                episode_id = env_index + args.num_envs * episode_count
                writer.append(
                    {
                        "state": state,
                        "next_state": next_state,
                        "actor_observation": observations["actor"],
                        "observation_history": frame_major_history(
                            observations["actor_history"]
                        ),
                        "command": command,
                        "teacher_action": action,
                        "applied_control": _applied_control(raw_env, robot),
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "context": context,
                        "timestamp": elapsed,
                        "dt": torch.full_like(elapsed, raw_env.step_dt),
                        "task_id": torch.zeros_like(episode_id),
                        "episode_id": episode_id,
                    }
                )
                elapsed += raw_env.step_dt
                done = terminated | truncated
                done_ids = done.nonzero(as_tuple=False).flatten()
                if len(done_ids):
                    episode_count[done_ids] += 1
                    elapsed[done_ids] = 0.0
                    raw_env.reset(env_ids=done_ids)
                observations = env.get_observations()
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    main()
