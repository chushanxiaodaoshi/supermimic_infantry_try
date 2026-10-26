"""Isaac Lab 3/MuJoCo-Warp runtime for infantry Teacher collection.

This is an adapter layer: the imported MJCF/USD keeps its native closed loops,
servo gains and spatial gas-spring tendons.  The only backend override bypasses
Isaac Lab's current rejection of spatial tendons and routes their native MuJoCo
actuators through the existing tendon-control bridge.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import isaaclab.sim as sim_utils
import torch
import warp as wp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass, instantiate
from isaaclab_newton.assets.articulation.articulation import Articulation
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from supermimic.core.spec import StateCodec
from supermimic.infantry.control import policy_action_to_control
from supermimic.infantry.replay_contract import ReplayContract
from supermimic.infantry.spec import (
    GAS_SPRING_TENDON_NAMES,
    INFANTRY_JOINT_NAMES,
    INFANTRY_SPEC,
    INITIAL_JOINT_POS,
    LOOP_EQUALITY_NAMES,
    POLICY_ACTION_NAMES,
)


def _short_labels(values) -> set[str]:
    if hasattr(values, "values"):
        values = values.values
    if hasattr(values, "numpy"):
        values = values.numpy()
    return {str(value).rsplit("/", 1)[-1] for value in values}


class SpatialTendonArticulation(Articulation):
    """Keep native spatial tendons while using Isaac Lab's MJWarp bridge.

    Pinned Isaac Lab 3 rejects every spatial tendon in ``_process_tendons``.
    The MJWarp bridge itself writes a native actuator control and is agnostic to
    the tendon path type, so this narrowly replaces the type gate.  Startup
    checks make an importer/backend drift a hard error instead of dropping the
    gas springs.
    """

    def _process_tendons(self) -> None:
        if self._root_view.tendon_count == 0:
            raise RuntimeError("infantry asset lost both gas-spring tendons")
        model = self._backend_manager.get_model()
        types = wp.to_torch(self._root_view.get_attribute("mujoco.tendon_type", model))
        if types.shape[-1] != 2 or not torch.all(types != 0):
            raise RuntimeError(
                f"expected exactly two native spatial tendons, got types {types[0, 0].tolist()}"
            )
        names = _short_labels(self._root_view.tendon_names)
        missing = set(GAS_SPRING_TENDON_NAMES) - names
        if missing:
            raise RuntimeError(f"imported asset is missing gas-spring tendons: {sorted(missing)}")
        self._fixed_tendon_control = self._backend_manager.create_fixed_tendon_control(self)
        if self._fixed_tendon_control is None:
            raise RuntimeError("spatial tendons exist but have no native MuJoCo actuators")

        attributes = getattr(model, "custom_attributes", {})
        equality_labels = attributes.get("mujoco:equality_constraint_label")
        if equality_labels is None:
            raise RuntimeError("backend exposes no MuJoCo equality constraints")
        missing = set(LOOP_EQUALITY_NAMES) - _short_labels(equality_labels)
        if missing:
            raise RuntimeError(f"imported asset lost loop equalities: {sorted(missing)}")


def make_sim_cfg(device: str, control_dt: float):
    physics = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=500,
            nconmax=128,
            cone="pyramidal",
            integrator="implicitfast",
            use_mujoco_contacts=True,
        ),
        num_substeps=4,
    )
    return sim_utils.SimulationCfg(dt=control_dt, device=device, physics=physics)


def make_robot_cfg(usd_path: Path, contract: ReplayContract) -> ArticulationCfg:
    effort_by_joint = dict(zip(contract.action_targets, contract.torque_limits, strict=True))
    return ArticulationCfg(
        class_type=SpatialTendonArticulation,
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            variants={"Physics": "mujoco"},
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, linear_damping=0.0, angular_damping=0.0
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.35),
            joint_pos=INITIAL_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        actuators={
            "replay_servos": ImplicitActuatorCfg(
                joint_names_expr=list(contract.policy_joints),
                stiffness=None,
                damping=None,
                joint_effort_limit=effort_by_joint,
            )
        },
    )


def make_scene_cfg(num_envs: int, usd_path: Path, contract: ReplayContract):
    robot_cfg = make_robot_cfg(usd_path, contract)

    @configclass
    class InfantrySceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        robot = robot_cfg

    return InfantrySceneCfg(num_envs=num_envs, env_spacing=2.0, replicate_physics=True)


class ReplayObservation:
    def __init__(self, num_envs: int, contract: ReplayContract, device: str):
        self.contract = contract
        self.history = {
            name: torch.zeros(num_envs, contract.history_length, width, device=device)
            for name, width in zip(
                contract.observation_terms, contract.observation_term_dims, strict=True
            )
        }
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.initialized[env_ids] = False

    def build(self, terms: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        missing = set(self.history) - set(terms)
        if missing:
            raise ValueError(f"missing replay observation terms: {sorted(missing)}")
        uninitialized = (~self.initialized).nonzero(as_tuple=False).flatten()
        for name, history in self.history.items():
            value = terms[name]
            history[:] = torch.roll(history, shifts=-1, dims=1)
            history[:, -1] = value
            if len(uninitialized):
                history[uninitialized] = value[uninitialized, None, :]
        self.initialized[:] = True
        current = torch.cat([terms[name] for name in self.history], dim=-1)
        term_major = torch.cat([value.flatten(1) for value in self.history.values()], dim=-1)
        return current, torch.cat((current, term_major), dim=-1)


class InfantryCollectionSim:
    """Batched simulator with replay-compatible observation and controls."""

    def __init__(
        self,
        sim_cfg,
        *,
        num_envs: int,
        usd_path: Path,
        contract: ReplayContract,
        context: torch.Tensor,
        command_ranges: Mapping[str, tuple[float, float]],
        max_episode_steps: int,
        seed: int,
    ):
        self.device = sim_cfg.device
        self.sim = SimulationContext(sim_cfg)
        self.scene = instantiate(make_scene_cfg(num_envs, usd_path, contract))
        self.sim.reset()
        self.robot = self.scene["robot"]
        if tuple(self.robot.joint_names) != INFANTRY_JOINT_NAMES:
            raise RuntimeError(f"joint order mismatch: {self.robot.joint_names}")
        self.contract = contract
        self.codec = StateCodec(INFANTRY_SPEC)
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.context = context.to(self.device).expand(num_envs, -1).clone()
        self.command_ranges = command_ranges
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.episodes = torch.zeros_like(self.steps)
        self.elapsed = torch.zeros(num_envs, device=self.device)
        self.last_action = torch.zeros(num_envs, 6, device=self.device)
        self.command = torch.zeros(num_envs, 3, device=self.device)
        self.observations = ReplayObservation(num_envs, contract, self.device)
        self.policy_joint_ids = torch.tensor(
            [self.robot.joint_names.index(name) for name in POLICY_ACTION_NAMES],
            device=self.device,
        )
        tendon_ids, tendon_names = self.robot.find_fixed_tendons(
            list(GAS_SPRING_TENDON_NAMES), preserve_order=True
        )
        if tuple(tendon_names) != GAS_SPRING_TENDON_NAMES:
            raise RuntimeError(f"gas-spring tendon order mismatch: {tendon_names}")
        self.tendon_ids = tendon_ids
        self.reset(torch.arange(num_envs, device=self.device))

    def state(self) -> torch.Tensor:
        data = self.robot.data
        pose = data.root_link_pose_w.torch
        velocity = data.root_link_vel_w.torch
        position = pose[:, :3] - self.scene.env_origins
        quat_wxyz = torch.cat((pose[:, 6:7], pose[:, 3:6]), dim=-1)
        return torch.cat(
            (
                position,
                quat_wxyz,
                velocity[:, :3],
                velocity[:, 3:6],
                data.joint_pos.torch,
                data.joint_vel.torch,
            ),
            dim=-1,
        )

    def observation(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.codec.unflatten(self.state())
        _, rotation, _, angular_velocity, q, qd = state
        rotation_t = rotation.transpose(-1, -2)
        body_ang = (rotation_t @ angular_velocity.unsqueeze(-1)).squeeze(-1)
        gravity = torch.tensor((0.0, 0.0, -1.0), device=self.device).expand(self.num_envs, -1)
        projected_gravity = (rotation_t @ gravity.unsqueeze(-1)).squeeze(-1)
        leg_ids = [self.robot.joint_names.index(name) for name in self.contract.leg_joints]
        policy_ids = [self.robot.joint_names.index(name) for name in self.contract.policy_joints]
        terms = {
            "base_ang_vel": body_ang.clamp(-20.0, 20.0) * 0.25,
            "projected_gravity": projected_gravity,
            "command": self.command * self.command.new_tensor((3.0, 0.25, 5.0)),
            "leg_joint_pos": q[:, leg_ids],
            "joint_vel": qd[:, policy_ids].clamp(-200.0, 200.0) * 0.05,
            "last_action": self.last_action.clamp(-10.0, 10.0),
        }
        current, teacher_input = self.observations.build(terms)
        frame_major = torch.cat(list(self.observations.history.values()), dim=-1)
        return current, frame_major, teacher_input

    def reset(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        root_velocity = self.robot.data.default_root_vel.torch[env_ids].clone()
        q = self.robot.data.default_joint_pos.torch[env_ids].clone()
        qd = self.robot.data.default_joint_vel.torch[env_ids].clone()
        self.robot.write_root_link_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim_index(
            root_velocity=root_velocity, env_ids=env_ids
        )
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=qd, env_ids=env_ids)
        for column, name in enumerate(("lin_vel_x", "ang_vel_z", "height")):
            low, high = self.command_ranges[name]
            self.command[env_ids, column].uniform_(low, high, generator=self.rng)
        self.last_action[env_ids] = 0.0
        self.steps[env_ids] = 0
        self.elapsed[env_ids] = 0.0
        self.observations.reset(env_ids)
        self.scene.reset(env_ids)
        self.sim.forward()
        self.scene.update(0.0)

    def step(
        self,
        teacher_action: torch.Tensor,
        actor_observation: torch.Tensor,
        observation_history: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        state = self.state()
        action = teacher_action.clamp(*self.contract.action_clip)
        zeros = torch.zeros_like(action)
        springs = action.new_tensor(self.contract.gas_spring_targets).expand(self.num_envs, -1)
        control = policy_action_to_control(
            action,
            default_motor_targets=zeros,
            gas_spring_forces=springs,
            action_scales=action.new_tensor(self.contract.action_scales),
            action_targets=self.contract.action_targets,
        )
        target_position = self.robot.data.joint_pos.torch.clone()
        target_velocity = torch.zeros_like(target_position)
        target_position[:, self.policy_joint_ids[[0, 1, 3, 4]]] = control[:, [0, 1, 3, 4]]
        target_velocity[:, self.policy_joint_ids[[2, 5]]] = control[:, [2, 5]]
        self.robot.set_joint_position_target_index(target=target_position)
        self.robot.set_joint_velocity_target_index(target=target_velocity)
        self.robot.set_fixed_tendon_position_target_index(
            target=springs, fixed_tendon_ids=self.tendon_ids
        )
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(INFANTRY_SPEC.control_dt)
        next_state = self.state()
        self.steps += 1
        self.elapsed += INFANTRY_SPEC.control_dt
        finite = torch.isfinite(next_state).all(dim=-1)
        terminated = (~finite) | (next_state[:, 2] < 0.12)
        truncated = self.steps >= self.max_episode_steps
        result = {
            "state": state,
            "next_state": next_state,
            "actor_observation": actor_observation,
            "observation_history": observation_history,
            "command": self.command.clone(),
            "executed_action": action,
            "applied_control": control,
            "terminated": terminated,
            "truncated": truncated,
            "context": self.context,
            "timestamp": self.elapsed - INFANTRY_SPEC.control_dt,
        }
        self.last_action = action
        return result
