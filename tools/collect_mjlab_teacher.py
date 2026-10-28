"""Collect PPO Teacher transitions from the real mjlab simulator.

The collector reuses mjlab's registered task, replay-time checkpoint loader,
ActionManager, and actuator implementation. It never reconstructs a controller
in this repository. Auto-reset is disabled so terminal states precede reset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import yaml

from supermimic.data.chunks import ChunkWriter
from supermimic.infantry.spec import (
    GAS_SPRING_TENDON_NAMES,
    INFANTRY_SPEC,
    POLICY_ACTION_NAMES,
)


@dataclass(frozen=True)
class CollectionSettings:
    mjlab_root: Path
    checkpoint: Path
    output: Path | None
    task: str
    teacher_name: str
    teacher_id: int
    task_id: int
    num_envs: int
    steps: int
    chunk_steps: int
    device: str
    seed: int
    split_seed: int
    action_noise_std: float
    command_override: dict[str, Any]
    validate_only: bool
    config_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mjlab-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--task", choices=("wheelleg-flat", "wheelleg-jump", "wheelleg-rough")
    )
    parser.add_argument("--teacher-name")
    parser.add_argument("--teacher-id", type=int)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--chunk-steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--action-noise-std", type=float)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    return value


def _pick(cli: object, section: dict[str, Any], key: str, default: Any = None) -> Any:
    return cli if cli is not None else section.get(key, default)


def load_settings(args: argparse.Namespace) -> CollectionSettings:
    config_path = args.config.expanduser().resolve() if args.config else None
    config: dict[str, Any] = {}
    if config_path is not None:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = _mapping(loaded, "collection config")
    teacher = _mapping(config.get("teacher"), "teacher")
    collection = _mapping(config.get("collection"), "collection")
    command_override = _mapping(config.get("command_override"), "command_override")

    mjlab_root = _pick(args.mjlab_root, config, "mjlab_root")
    checkpoint = _pick(args.checkpoint, teacher, "checkpoint")
    output = _pick(args.output, collection, "output")
    if mjlab_root is None or checkpoint is None:
        raise ValueError("mjlab_root and teacher.checkpoint are required")
    validate_only = bool(args.validate_only or collection.get("validate_only", False))
    if output is None and not validate_only:
        raise ValueError("collection.output is required unless validate_only is set")
    task = str(_pick(args.task, teacher, "task", "wheelleg-flat"))
    if task not in {"wheelleg-flat", "wheelleg-jump", "wheelleg-rough"}:
        raise ValueError(f"unsupported task: {task}")
    action_noise_std = float(
        _pick(args.action_noise_std, collection, "action_noise_std", 0.0)
    )
    if action_noise_std < 0:
        raise ValueError("action_noise_std cannot be negative")
    return CollectionSettings(
        mjlab_root=Path(mjlab_root).expanduser().resolve(),
        checkpoint=Path(checkpoint).expanduser().resolve(),
        output=Path(output).expanduser().resolve() if output is not None else None,
        task=task,
        teacher_name=str(_pick(args.teacher_name, teacher, "name", task)),
        teacher_id=int(_pick(args.teacher_id, teacher, "id", 0)),
        task_id=int(_pick(args.task_id, teacher, "task_id", 0)),
        num_envs=int(_pick(args.num_envs, collection, "num_envs", 64)),
        steps=int(_pick(args.steps, collection, "steps", 1000)),
        chunk_steps=int(_pick(args.chunk_steps, collection, "chunk_steps", 128)),
        device=str(_pick(args.device, collection, "device", "cuda:0")),
        seed=int(_pick(args.seed, collection, "seed", 42)),
        split_seed=int(_pick(args.split_seed, collection, "split_seed", 42)),
        action_noise_std=action_noise_std,
        command_override=command_override,
        validate_only=validate_only,
        config_path=config_path,
    )


def _command_contract(command_cfg: Any) -> dict[str, Any]:
    return {
        "lin_vel_x": list(command_cfg.ranges.lin_vel_x),
        "lin_vel_y": list(command_cfg.ranges.lin_vel_y),
        "ang_vel_z": list(command_cfg.ranges.ang_vel_z),
        "height_range": list(command_cfg.height_range),
        "default_height": command_cfg.default_height,
        "rel_standing_envs": command_cfg.rel_standing_envs,
        "rel_forward_envs": command_cfg.rel_forward_envs,
    }


def _apply_command_override(env_cfg: Any, override: dict[str, Any]) -> None:
    if not override:
        return
    command_cfg = env_cfg.commands["twist"]
    range_names = {"lin_vel_x", "lin_vel_y", "ang_vel_z"}
    scalar_names = {"rel_standing_envs", "rel_forward_envs", "default_height"}
    allowed = range_names | scalar_names | {"height_range"}
    unknown = set(override) - allowed
    if unknown:
        raise ValueError(f"unknown command override fields: {sorted(unknown)}")
    for name in range_names:
        if name in override:
            setattr(command_cfg.ranges, name, tuple(float(x) for x in override[name]))
    if "height_range" in override:
        command_cfg.height_range = tuple(float(x) for x in override["height_range"])
    for name in scalar_names:
        if name in override:
            setattr(command_cfg, name, float(override[name]))


def configure_collection(
    env_cfg: Any,
    *,
    seed: int,
    task: str = "wheelleg-flat",
    command_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Constrain only dynamics absent from the World Model input contract."""
    if env_cfg.scene.terrain.terrain_type != "plane":
        raise ValueError(
            "the current world model has no terrain input; rough terrain collection "
            "is intentionally blocked"
        )
    if task not in {"wheelleg-flat", "wheelleg-jump"}:
        raise ValueError(f"task {task!r} has not been adapted to the world-model inputs")
    env_cfg.seed = seed
    commands = getattr(env_cfg, "commands", None)
    original_command = _command_contract(commands["twist"]) if commands else None
    disabled: list[str] = []
    if task == "wheelleg-flat":
        for name in ("push_robots", "body_torque"):
            if name in env_cfg.events:
                env_cfg.events.pop(name)
                disabled.append(name)
        event = env_cfg.events.get("body_mass_inertia")
        if event is None:
            raise ValueError("flat task is missing body_mass_inertia randomization")
        event.params["inertia_scale_range"] = (1.0, 1.0)
        wheels = ("Left_wheel_collision", "Right_wheel_collision")
        for name in ("wheel_friction", "restitution"):
            if name not in env_cfg.events:
                raise ValueError(f"flat task is missing {name} randomization")
            env_cfg.events[name].params["asset_cfg"].geom_names = wheels
        profile = "flat_observed_dynamics_v2"
    else:
        unexpected = set(env_cfg.events) - {
            "reset_base",
            "reset_joints",
            "gas_spring_force",
            "wheel_friction",
        }
        if unexpected:
            raise ValueError(f"jump task has unreviewed dynamics events: {sorted(unexpected)}")
        wheels = ("Left_wheel_collision", "Right_wheel_collision")
        profile = "jump_v8_observed_dynamics_v1"
    if command_override and not commands:
        raise ValueError("command overrides require an environment command manager")
    _apply_command_override(env_cfg, command_override or {})
    return {
        "profile": profile,
        "disabled_events": disabled,
        "inertia_scale_range": [1.0, 1.0] if task == "wheelleg-flat" else None,
        "randomized_contact_geoms": list(wheels),
        "original_command": original_command,
        "effective_command": _command_contract(commands["twist"]) if commands else None,
        "seed": seed,
    }


def _state(raw_env: Any, robot: Any) -> torch.Tensor:
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

    def __init__(self, raw_env: Any, robot: Any):
        self.raw_env = raw_env
        self.robot = robot
        device = raw_env.device
        wheel_local, _ = robot.find_geoms(
            ("Left_wheel_collision", "Right_wheel_collision"), preserve_order=True
        )
        base_local, _ = robot.find_bodies(("base_link",), preserve_order=True)
        actuator_local, _ = robot.find_actuators(POLICY_ACTION_NAMES, preserve_order=True)
        joint_local, _ = robot.find_joints((".*",), preserve_order=True)
        tendon_local, _ = robot.find_tendons(
            GAS_SPRING_TENDON_NAMES, preserve_order=True
        )
        self.wheel_geoms = robot.indexing.geom_ids[torch.tensor(wheel_local, device=device)]
        self.base_body = robot.indexing.body_ids[torch.tensor(base_local, device=device)]
        self.policy_ctrl = robot.indexing.ctrl_ids[
            torch.tensor(actuator_local, device=device)
        ]
        self.joint_dofs = robot.indexing.joint_v_adr[
            torch.tensor(joint_local, device=device)
        ]
        self.tendon_local = torch.tensor(tendon_local, device=device)

    def read(self) -> torch.Tensor:
        model = self.raw_env.sim.model
        context = torch.cat(
            (
                model.geom_friction[:, self.wheel_geoms, 0],
                model.body_mass[:, self.base_body],
                model.body_inertia[:, self.base_body].flatten(start_dim=1),
                model.body_ipos[:, self.base_body].flatten(start_dim=1),
                model.actuator_gainprm[:, self.policy_ctrl, 0],
                model.actuator_biasprm[:, self.policy_ctrl, 1],
                model.actuator_biasprm[:, self.policy_ctrl, 2],
                model.actuator_forcerange[:, self.policy_ctrl, 0],
                model.actuator_forcerange[:, self.policy_ctrl, 1],
                self.robot.data.tendon_effort_target[:, self.tendon_local],
                model.dof_damping[:, self.joint_dofs],
                model.dof_frictionloss[:, self.joint_dofs],
                model.geom_solref[:, self.wheel_geoms, :2].flatten(start_dim=1),
            ),
            dim=-1,
        )
        if context.shape[-1] != INFANTRY_SPEC.context_dim:
            raise RuntimeError(
                f"context width {context.shape[-1]} != {INFANTRY_SPEC.context_dim}"
            )
        return context


def _applied_control(raw_env: Any, robot: Any) -> torch.Tensor:
    processed = [
        raw_env.action_manager.get_term(name)._processed_actions
        for name in raw_env.action_manager.active_terms
    ]
    tendon_ids, _ = robot.find_tendons(GAS_SPRING_TENDON_NAMES, preserve_order=True)
    spring = robot.data.tendon_effort_target[:, tendon_ids]
    control = torch.cat((*processed, spring), dim=-1)
    if control.shape[-1] != INFANTRY_SPEC.control_dim:
        raise RuntimeError("unexpected applied-control layout")
    return control


def _motion_phase(
    raw_env: Any, state: torch.Tensor, was_airborne: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return contact/takeoff/ascent/descent/landing labels (0..4)."""
    found = raw_env.scene["wheel_ground_contact"].data.found
    if found is None:
        raise RuntimeError("wheel contact sensor does not expose the found field")
    contact = found.gt(0).any(dim=-1)
    vertical_velocity = state[:, 9]
    phase = torch.zeros_like(vertical_velocity, dtype=torch.long)
    phase[contact & (vertical_velocity > 0.15)] = 1
    phase[(~contact) & (vertical_velocity >= 0.0)] = 2
    phase[(~contact) & (vertical_velocity < 0.0)] = 3
    phase[contact & was_airborne] = 4
    return phase, ~contact


def _git_commit(directory: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _environment_contract(
    raw_env: Any, agent_cfg: Any, observations: dict[str, torch.Tensor]
) -> dict[str, Any]:
    actor_cfg = agent_cfg.actor
    return {
        "physics_dt": raw_env.physics_dt,
        "control_dt": raw_env.step_dt,
        "decimation": raw_env.cfg.decimation,
        "terrain_type": raw_env.cfg.scene.terrain.terrain_type,
        "action_terms": list(raw_env.action_manager.active_terms),
        "policy_action_names": list(INFANTRY_SPEC.policy_action_names),
        "applied_control_names": list(INFANTRY_SPEC.control_names),
        "joint_names": list(INFANTRY_SPEC.joint_names),
        "observation_shapes": {
            name: list(value.shape[1:]) for name, value in observations.items()
        },
        "actor": {
            "class_name": actor_cfg.class_name,
            "hidden_dims": list(actor_cfg.hidden_dims),
            "history_encoder_hidden_dims": list(actor_cfg.history_encoder_hidden_dims),
            "history_latent_dim": actor_cfg.history_latent_dim,
            "activation": actor_cfg.activation,
            "obs_normalization": actor_cfg.obs_normalization,
        },
        "clip_actions": agent_cfg.clip_actions,
        "events": sorted(raw_env.cfg.events),
    }


def main() -> None:
    settings = load_settings(parse_args())
    if not settings.checkpoint.is_file():
        raise FileNotFoundError(settings.checkpoint)
    if (
        settings.output is not None
        and settings.output.exists()
        and any(settings.output.iterdir())
    ):
        raise FileExistsError(f"dataset destination must be empty: {settings.output}")
    sys.path.insert(0, str(settings.mjlab_root / "src"))

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.scripts.play import _checkpoint_obs_dims
    from mjlab.tasks import wheelleg  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    torch.manual_seed(settings.seed)
    env_cfg = load_env_cfg(settings.task, play=False)
    collection_profile = configure_collection(
        env_cfg,
        seed=settings.seed,
        task=settings.task,
        command_override=settings.command_override,
    )
    env_cfg.scene.num_envs = settings.num_envs
    env_cfg.auto_reset = False
    agent_cfg = load_rl_cfg(settings.task)
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=settings.device, render_mode=None)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    try:
        runner_cls = load_runner_cls(settings.task) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=settings.device)
        runner.load(
            str(settings.checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=settings.device,
        )
        policy = runner.get_inference_policy(device=settings.device)
        robot = raw_env.scene["robot"]
        if tuple(robot.joint_names) != INFANTRY_SPEC.joint_names:
            raise RuntimeError(f"joint order mismatch: {tuple(robot.joint_names)}")
        observations = env.get_observations()
        checkpoint_actor_dim, _ = _checkpoint_obs_dims(settings.checkpoint)
        actual_actor_dim = (
            observations["actor"].shape[-1]
            + observations["actor_history"].shape[-1]
        )
        if checkpoint_actor_dim != actual_actor_dim:
            raise RuntimeError(
                f"checkpoint actor width {checkpoint_actor_dim} != environment width "
                f"{actual_actor_dim}"
            )
        with torch.inference_mode():
            probe_action = policy(observations)
        if probe_action.shape != (
            settings.num_envs,
            INFANTRY_SPEC.policy_action_dim,
        ):
            raise RuntimeError(f"unexpected policy action shape: {tuple(probe_action.shape)}")
        contract = _environment_contract(raw_env, agent_cfg, observations)
        print(json.dumps({"status": "contract_valid", "contract": contract}), flush=True)
        if settings.validate_only:
            return

        assert settings.output is not None
        run_uuid = uuid.uuid4()
        run_id_value = run_uuid.int & 0x7FFF_FFFF_FFFF_FFFF
        metadata: dict[str, Any] = {
            "source": "mjlab",
            "collection_run_uuid": str(run_uuid),
            "teacher_name": settings.teacher_name,
            "teacher_id": settings.teacher_id,
            "task": settings.task,
            "task_id": settings.task_id,
            "checkpoint": str(settings.checkpoint),
            "checkpoint_sha256": _sha256(settings.checkpoint),
            "supermimic_commit": _git_commit(REPO),
            "mjlab_commit": _git_commit(settings.mjlab_root),
            "config_path": str(settings.config_path) if settings.config_path else None,
            "seed": settings.seed,
            "split_seed": settings.split_seed,
            "action_noise_std": settings.action_noise_std,
            "collection_profile": collection_profile,
            "environment_contract": contract,
            "observation_history_layout": "frame_major_oldest_first",
            "motion_phase_labels": {
                "0": "contact",
                "1": "takeoff_contact",
                "2": "airborne_ascent",
                "3": "airborne_descent",
                "4": "landing_contact",
            },
        }
        writer = ChunkWriter(
            settings.output,
            INFANTRY_SPEC,
            chunk_steps=settings.chunk_steps,
            split_seed=settings.split_seed,
            metadata=metadata,
        )
        context_view = PhysicsContext(raw_env, robot)
        episode_count = torch.zeros(
            settings.num_envs, dtype=torch.long, device=settings.device
        )
        env_index = torch.arange(
            settings.num_envs, dtype=torch.long, device=settings.device
        )
        elapsed = torch.zeros(settings.num_envs, device=settings.device)
        was_airborne = torch.zeros(
            settings.num_envs, dtype=torch.bool, device=settings.device
        )
        noise_rng = torch.Generator(device=settings.device).manual_seed(
            settings.seed + 1_000_003
        )
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                for _ in range(settings.steps):
                    state = _state(raw_env, robot)
                    phase, was_airborne = _motion_phase(
                        raw_env, state, was_airborne
                    )
                    command = raw_env.command_manager.get_command("twist")[
                        :, (0, 2, 3)
                    ]
                    teacher_action = policy(observations)
                    executed_action = teacher_action
                    if settings.action_noise_std:
                        noise = torch.randn(
                            teacher_action.shape,
                            generator=noise_rng,
                            device=teacher_action.device,
                            dtype=teacher_action.dtype,
                        )
                        executed_action = (
                            executed_action + settings.action_noise_std * noise
                        )
                    if agent_cfg.clip_actions is not None:
                        executed_action = executed_action.clamp(
                            -agent_cfg.clip_actions, agent_cfg.clip_actions
                        )
                    if torch.count_nonzero(robot.data.body_external_wrench):
                        raise RuntimeError(
                            "unrecorded external wrench in collection environment"
                        )
                    context = context_view.read()
                    _, reward, terminated, truncated, _ = raw_env.step(
                        executed_action
                    )
                    next_state = _state(raw_env, robot)
                    episode_id = env_index + settings.num_envs * episode_count
                    writer.append(
                        {
                            "state": state,
                            "next_state": next_state,
                            "actor_observation": observations["actor"],
                            "observation_history": frame_major_history(
                                observations["actor_history"]
                            ),
                            "command": command,
                            "teacher_action": teacher_action,
                            "executed_action": executed_action,
                            "applied_control": _applied_control(raw_env, robot),
                            "reward": reward,
                            "terminated": terminated,
                            "truncated": truncated,
                            "context": context,
                            "timestamp": elapsed,
                            "dt": torch.full_like(elapsed, raw_env.step_dt),
                            "task_id": torch.full_like(
                                episode_id, settings.task_id
                            ),
                            "teacher_id": torch.full_like(
                                episode_id, settings.teacher_id
                            ),
                            "run_id": torch.full_like(
                                episode_id, run_id_value
                            ),
                            "episode_id": episode_id,
                            "motion_phase": phase,
                        }
                    )
                    elapsed += raw_env.step_dt
                    done = terminated | truncated
                    done_ids = done.nonzero(as_tuple=False).flatten()
                    if len(done_ids):
                        episode_count[done_ids] += 1
                        elapsed[done_ids] = 0.0
                        was_airborne[done_ids] = False
                        raw_env.reset(env_ids=done_ids)
                    observations = env.get_observations()
        finally:
            duration = time.perf_counter() - started
            metadata["collection_runtime"] = {
                "duration_seconds": duration,
                "transitions_per_second": writer.transitions
                / max(duration, 1.0e-9),
                "transitions": writer.transitions,
            }
            writer.close()
    finally:
        env.close()


if __name__ == "__main__":
    main()
