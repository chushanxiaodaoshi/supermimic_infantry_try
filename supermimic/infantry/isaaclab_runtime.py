"""Isaac Lab 3/MuJoCo-Warp runtime for infantry Teacher collection.

This is an adapter layer: the robot USD keeps its native closed loops and
spatial gas-spring tendons, while servo gains come from the replay MJCF
contract. The only backend override bypasses Isaac Lab's current rejection of
spatial tendons and routes their native MuJoCo actuators through the existing
tendon-control bridge.
"""

from __future__ import annotations

import os
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
from newton._src.sim.builder import ModelBuilder

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


# ---------------------------------------------------------------------------
# Newton 1.6.1 compatibility patch
#
# Newton imports:
#
#   mujoco:tendon   = 2
#   mujoco:actuator = 8
#
# but the imported gas-spring rows currently end up with:
#
#   tendon owners:
#       [-1, -1]
#
#   actuator owners:
#       [0, 0, 0, 0, 0, 0, -1, -1]
#
# The ArticulationView filters its custom-frequency rows by these ownership
# arrays, therefore it sees zero tendons and only six actuators.
#
# Patch the FINAL materialized Model ownership metadata. This is intentionally
# restricted to this infantry model: one articulation, two tendons and eight
# actuators.
# ---------------------------------------------------------------------------

_ORIGINAL_FINALIZE_CUSTOM_FREQUENCY_METADATA = (
    ModelBuilder._finalize_custom_frequency_metadata
)


def _finalize_infantry_custom_frequency_metadata(
    self,
    model,
    device,
) -> None:
    _ORIGINAL_FINALIZE_CUSTOM_FREQUENCY_METADATA(
        self,
        model,
        device,
    )

    if model.articulation_count != 1:
        return

    tendon_count = model.custom_frequency_counts.get(
        "mujoco:tendon",
        0,
    )

    actuator_count = model.custom_frequency_counts.get(
        "mujoco:actuator",
        0,
    )

    if tendon_count != 2:
        raise RuntimeError(
            f"expected 2 MuJoCo tendons, got {tendon_count}"
        )

    if actuator_count != 8:
        raise RuntimeError(
            f"expected 8 MuJoCo actuators, got {actuator_count}"
        )

    tendon_owners = model.custom_frequency_articulation.get(
        "mujoco:tendon"
    )

    actuator_owners = model.custom_frequency_articulation.get(
        "mujoco:actuator"
    )

    if tendon_owners is None:
        raise RuntimeError(
            "Newton model has no mujoco:tendon ownership array"
        )

    if actuator_owners is None:
        raise RuntimeError(
            "Newton model has no mujoco:actuator ownership array"
        )

    tendon_types = getattr(model.mujoco, "tendon_type", None)
    if tendon_types is None:
        raise RuntimeError("Newton model has no mujoco.tendon_type array")

    tendon_type_np = tendon_types.numpy().copy()
    if tendon_type_np.shape != (2,):
        raise RuntimeError(
            "expected two MuJoCo tendon type rows, "
            f"got shape {tendon_type_np.shape}"
        )
    if not (tendon_type_np == 0).all():
        raise RuntimeError(
            "unexpected pre-patch tendon types: "
            f"{tendon_type_np.tolist()}"
        )

    # Newton's USD importer creates the four MjcSite shapes, but in 1.6.1 it
    # does not materialize the ``mjc:path`` relationships as
    # mujoco:tendon_wrap rows.  Reconstruct exactly the two-site paths authored
    # by scene.xml.  The suffix match is needed because USD shape labels retain
    # their full prim path.
    gas_spring_site_names = (
        "right_lf1_qitanhuang",
        "right_lf0_qitanhuang",
        "left_lf1_qitanhuang",
        "left_lf0_qitanhuang",
    )
    gas_spring_site_indices = []
    for site_name in gas_spring_site_names:
        matches = [
            index
            for index, label in enumerate(model.shape_label)
            if label == site_name or label.endswith(f"/{site_name}")
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one imported gas-spring site {site_name!r}, "
                f"got shape indices {matches}"
            )
        gas_spring_site_indices.append(matches[0])

    wrap_values = {
        "tendon_wrap_type": [0, 0, 0, 0],  # site, site; site, site
        "tendon_wrap_shape": gas_spring_site_indices,
        "tendon_wrap_sidesite": [-1, -1, -1, -1],
        "tendon_wrap_prm": [0.0, 0.0, 0.0, 0.0],
        "tendon_wrap_articulation": [0, 0, 0, 0],
    }
    for attribute_name, values in wrap_values.items():
        old_values = getattr(model.mujoco, attribute_name, None)
        if old_values is not None and len(old_values) != 0:
            raise RuntimeError(
                f"expected empty Newton USD {attribute_name}, got "
                f"{old_values.numpy().tolist()}"
            )
        attribute = self.custom_attributes[f"mujoco:{attribute_name}"]
        setattr(
            model.mujoco,
            attribute_name,
            wp.array(values, dtype=attribute.dtype, device=device),
        )

    for attribute_name, values in {
        "tendon_wrap_adr": [0, 2],
        "tendon_wrap_num": [2, 2],
    }.items():
        old_values = getattr(model.mujoco, attribute_name)
        if old_values.numpy().tolist() != [0, 0]:
            raise RuntimeError(
                f"unexpected pre-patch {attribute_name}: "
                f"{old_values.numpy().tolist()}"
            )
        setattr(
            model.mujoco,
            attribute_name,
            wp.array(values, dtype=old_values.dtype, device=old_values.device),
        )

    model.custom_frequency_counts["mujoco:tendon_wrap"] = 4
    model.custom_frequency_articulation["mujoco:tendon_wrap"] = (
        model.mujoco.tendon_wrap_articulation
    )

    tendon_np = tendon_owners.numpy().copy()
    actuator_np = actuator_owners.numpy().copy()

    tendon_np[tendon_np < 0] = 0
    actuator_np[actuator_np < 0] = 0

    model.custom_frequency_articulation[
        "mujoco:tendon"
    ] = wp.array(
        tendon_np,
        dtype=tendon_owners.dtype,
        device=tendon_owners.device,
    )

    model.custom_frequency_articulation[
        "mujoco:actuator"
    ] = wp.array(
        actuator_np,
        dtype=actuator_owners.dtype,
        device=actuator_owners.device,
    )

    # The USD authors both rows as MjcTendon type="spatial", but Newton 1.6.1
    # materializes them as type 0 (fixed).  Its MJWarp solver then skips them
    # because they have no fixed-tendon joint wraps.  The path rows were
    # reconstructed above; correct the remaining discriminator before the
    # solver builds its MuJoCo spec (0=fixed, 1=spatial).
    model.mujoco.tendon_type = wp.array(
        tendon_type_np + 1,
        dtype=tendon_types.dtype,
        device=tendon_types.device,
    )

    print(
        "INFANTRY PATCH ownership:",
        "tendon=",
        model.custom_frequency_articulation[
            "mujoco:tendon"
        ].numpy().tolist(),
        "actuator=",
        model.custom_frequency_articulation[
            "mujoco:actuator"
        ].numpy().tolist(),
        "type=",
        model.mujoco.tendon_type.numpy().tolist(),
        "wrap_shape=",
        model.mujoco.tendon_wrap_shape.numpy().tolist(),
    )


ModelBuilder._finalize_custom_frequency_metadata = (
    _finalize_infantry_custom_frequency_metadata
)


class SpatialTendonArticulation(Articulation):
    """Keep native spatial tendons while using Isaac Lab's MJWarp bridge.

    Pinned Isaac Lab 3 rejects every spatial tendon in ``_process_tendons``.
    The MJWarp bridge itself writes a native actuator control and is agnostic to
    the tendon path type, so this narrowly replaces the type gate. Startup
    checks make an importer/backend drift a hard error instead of dropping the
    gas springs.
    """

    def _process_tendons(self) -> None:
        manager = SimulationContext.instance().physics_manager
        model = manager.get_model()

        if self._root_view.tendon_count != 2:
            raise RuntimeError(
                "expected exactly two gas-spring tendons, "
                f"got {self._root_view.tendon_count}"
            )

        names = _short_labels(
            self._root_view.tendon_names
        )

        missing = (
            set(GAS_SPRING_TENDON_NAMES)
            - names
        )

        if missing:
            raise RuntimeError(
                "imported asset is missing gas-spring tendons: "
                f"{sorted(missing)}; "
                f"found={sorted(names)}"
            )

        # Diagnostic only.
        #
        # Do NOT use tendon_type as a hard spatial/fixed gate here.
        # In the current Newton import this attribute is [0, 0],
        # even though these are the two native gas-spring tendon rows.
        try:
            tendon_types = wp.to_torch(
                self._root_view.get_attribute(
                    "mujoco.tendon_type",
                    model,
                )
            )

            print(
                "INFANTRY tendon types:",
                tendon_types.tolist(),
            )

        except Exception as exc:
            print(
                "INFANTRY tendon type diagnostic unavailable:",
                repr(exc),
            )

        self._fixed_tendon_control = (
            manager.create_fixed_tendon_control(
                self
            )
        )

        if self._fixed_tendon_control is None:
            raise RuntimeError(
                "gas-spring tendons exist but "
                "MJWarp created no tendon actuator control"
            )

        mujoco = getattr(
            model,
            "mujoco",
            None,
        )

        if mujoco is None:
            raise RuntimeError(
                "Newton Model exposes no mujoco namespace"
            )

        equality_labels = getattr(
            mujoco,
            "equality_constraint_label",
            None,
        )

        if equality_labels is None:
            raise RuntimeError(
                "backend exposes no MuJoCo equality constraints"
            )

        missing = (
            set(LOOP_EQUALITY_NAMES)
            - _short_labels(equality_labels)
        )

        if missing:
            raise RuntimeError(
                "imported asset lost loop equalities: "
                f"{sorted(missing)}"
            )


def make_sim_cfg(
    device: str,
    control_dt: float,
):
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

    return sim_utils.SimulationCfg(
        dt=control_dt,
        device=device,
        physics=physics,
    )


def make_robot_cfg(
    usd_path: Path,
    contract: ReplayContract,
    controller: Mapping[
        str,
        Mapping[str, float | str],
    ],
) -> ArticulationCfg:
    effort_by_joint = dict(
        zip(
            contract.action_targets,
            contract.torque_limits,
            strict=True,
        )
    )

    stiffness = {
        name: float(controller[name]["stiffness"])
        for name in contract.policy_joints
    }

    damping = {
        name: float(controller[name]["damping"])
        for name in contract.policy_joints
    }

    return ArticulationCfg(
        class_type=SpatialTendonArticulation,
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            variants={"Physics": "mujoco"},
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                # scene.xml disables robot/robot contacts while preserving
                # robot/terrain contacts.  Enabling articulation self collision
                # here changes the deployment dynamics before the policy sees
                # its first observation.
                enabled_self_collisions=False
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # run_policy.py loads initial_state.npz, whose free-joint position
            # is exactly (0, 0, 0.2).  Starting 15 cm higher creates a drop that
            # is absent from the reference replay.
            pos=(0.0, 0.0, 0.2),
            joint_pos=INITIAL_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        actuators={
            "replay_servos": ImplicitActuatorCfg(
                joint_names_expr=list(
                    contract.policy_joints
                ),
                stiffness=stiffness,
                damping=damping,
                joint_effort_limit=effort_by_joint,
            )
        },
    )


def make_scene_cfg(
    num_envs: int,
    usd_path: Path,
    contract: ReplayContract,
    controller: Mapping[
        str,
        Mapping[str, float | str],
    ],
):
    robot_cfg = make_robot_cfg(
        usd_path,
        contract,
        controller,
    )

    @configclass
    class InfantrySceneCfg(
        InteractiveSceneCfg
    ):
        ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(),
        )

        robot = robot_cfg

    return InfantrySceneCfg(
        num_envs=num_envs,
        env_spacing=2.0,
        replicate_physics=True,
    )


class ReplayObservation:
    def __init__(
        self,
        num_envs: int,
        contract: ReplayContract,
        device: str,
    ):
        self.contract = contract

        self.history = {
            name: torch.zeros(
                num_envs,
                contract.history_length,
                width,
                device=device,
            )
            for name, width in zip(
                contract.observation_terms,
                contract.observation_term_dims,
                strict=True,
            )
        }

        self.initialized = torch.zeros(
            num_envs,
            dtype=torch.bool,
            device=device,
        )

    def reset(
        self,
        env_ids: torch.Tensor,
    ) -> None:
        self.initialized[
            env_ids
        ] = False

    def build(
        self,
        terms: Mapping[
            str,
            torch.Tensor,
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        missing = (
            set(self.history)
            - set(terms)
        )

        if missing:
            raise ValueError(
                "missing replay observation terms: "
                f"{sorted(missing)}"
            )

        uninitialized = (
            (~self.initialized)
            .nonzero(
                as_tuple=False
            )
            .flatten()
        )

        for name, history in self.history.items():
            value = terms[name]

            history[:] = torch.roll(
                history,
                shifts=-1,
                dims=1,
            )

            history[:, -1] = value

            if len(uninitialized):
                history[
                    uninitialized
                ] = value[
                    uninitialized,
                    None,
                    :,
                ]

        self.initialized[:] = True

        current = torch.cat(
            [
                terms[name]
                for name in self.history
            ],
            dim=-1,
        )

        term_major = torch.cat(
            [
                value.flatten(1)
                for value in self.history.values()
            ],
            dim=-1,
        )

        return (
            current,
            torch.cat(
                (
                    current,
                    term_major,
                ),
                dim=-1,
            ),
        )


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
        controller: Mapping[
            str,
            Mapping[str, float | str],
        ],
        command_ranges: Mapping[
            str,
            tuple[float, float],
        ],
        max_episode_steps: int,
        seed: int,
    ):
        self.device = sim_cfg.device

        self.sim = SimulationContext(
            sim_cfg
        )

        self.scene = instantiate(
            make_scene_cfg(
                num_envs,
                usd_path,
                contract,
                controller,
            )
        )

        self.sim.reset()

        self.robot = self.scene["robot"]

        actual_joint_names = tuple(self.robot.joint_names)
        expected_joint_names = tuple(INFANTRY_JOINT_NAMES)

        if set(actual_joint_names) != set(expected_joint_names):
            missing = set(expected_joint_names) - set(actual_joint_names)
            extra = set(actual_joint_names) - set(expected_joint_names)

            raise RuntimeError(
                "joint name mismatch: "
                f"missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )

        # Isaac Lab/Newton joint order does not necessarily match the replay/state
        # contract order. Reorder simulator joint state into INFANTRY_JOINT_NAMES.
        self.state_joint_ids = torch.tensor(
            [
                actual_joint_names.index(name)
                for name in expected_joint_names
            ],
            device=self.device,
            dtype=torch.long,
        )

        # state() below has already reordered q/qd into INFANTRY_JOINT_NAMES.
        # Observation indices must therefore be expressed in that contract
        # order, not in the backend's joint order.
        self.observation_leg_ids = torch.tensor(
            [expected_joint_names.index(name) for name in contract.leg_joints],
            device=self.device,
            dtype=torch.long,
        )
        self.observation_policy_ids = torch.tensor(
            [expected_joint_names.index(name) for name in contract.policy_joints],
            device=self.device,
            dtype=torch.long,
        )

        self.contract = contract
        self.codec = StateCodec(
            INFANTRY_SPEC
        )

        self.num_envs = num_envs
        self.max_episode_steps = (
            max_episode_steps
        )

        self.context = (
            context
            .to(self.device)
            .expand(
                num_envs,
                -1,
            )
            .clone()
        )

        self.command_ranges = (
            command_ranges
        )

        self.rng = (
            torch.Generator(
                device=self.device
            )
            .manual_seed(seed)
        )

        self.steps = torch.zeros(
            num_envs,
            dtype=torch.long,
            device=self.device,
        )

        self.episodes = (
            torch.zeros_like(
                self.steps
            )
        )

        self.elapsed = torch.zeros(
            num_envs,
            device=self.device,
        )

        self.last_action = (
            torch.zeros(
                num_envs,
                6,
                device=self.device,
            )
        )

        self.command = torch.zeros(
            num_envs,
            3,
            device=self.device,
        )

        self.observations = (
            ReplayObservation(
                num_envs,
                contract,
                self.device,
            )
        )

        self.policy_joint_ids = (
            torch.tensor(
                [
                    self.robot
                    .joint_names
                    .index(name)
                    for name in POLICY_ACTION_NAMES
                ],
                device=self.device,
            )
        )

        (
            tendon_ids,
            tendon_names,
        ) = self.robot.find_fixed_tendons(
            list(
                GAS_SPRING_TENDON_NAMES
            ),
            preserve_order=True,
        )

        if (
            tuple(tendon_names)
            != GAS_SPRING_TENDON_NAMES
        ):
            raise RuntimeError(
                "gas-spring tendon order mismatch: "
                f"{tendon_names}"
            )

        self.tendon_ids = tendon_ids

        self.reset(
            torch.arange(
                num_envs,
                device=self.device,
            )
        )

    def state(
        self,
    ) -> torch.Tensor:
        data = self.robot.data

        pose = (
            data.root_link_pose_w.torch
        )

        velocity = (
            data.root_link_vel_w.torch
        )

        position = (
            pose[:, :3]
            - self.scene.env_origins
        )

        quat_wxyz = torch.cat(
            (
                pose[:, 6:7],
                pose[:, 3:6],
            ),
            dim=-1,
        )

        return torch.cat(
            (
                position,
                quat_wxyz,
                velocity[:, :3],
                velocity[:, 3:6],
                data.joint_pos.torch[:, self.state_joint_ids],
                data.joint_vel.torch[:, self.state_joint_ids],
            ),
            dim=-1,
        )

    def observation(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        state = (
            self.codec.unflatten(
                self.state()
            )
        )

        (
            _,
            rotation,
            _,
            angular_velocity,
            q,
            qd,
        ) = state

        rotation_t = (
            rotation.transpose(
                -1,
                -2,
            )
        )

        body_ang = (
            rotation_t
            @ angular_velocity.unsqueeze(-1)
        ).squeeze(-1)

        gravity = torch.tensor(
            (
                0.0,
                0.0,
                -1.0,
            ),
            device=self.device,
        ).expand(
            self.num_envs,
            -1,
        )

        projected_gravity = (
            rotation_t
            @ gravity.unsqueeze(-1)
        ).squeeze(-1)

        terms = {
            "base_ang_vel":
                body_ang.clamp(
                    -20.0,
                    20.0,
                )
                * 0.25,

            "projected_gravity":
                projected_gravity,

            "command":
                self.command
                * self.command.new_tensor(
                    (
                        3.0,
                        0.25,
                        5.0,
                    )
                ),

            "leg_joint_pos":
                q[:, self.observation_leg_ids],

            "joint_vel":
                qd[:, self.observation_policy_ids]
                .clamp(
                    -200.0,
                    200.0,
                )
                * 0.05,

            "last_action":
                self.last_action
                .clamp(
                    -10.0,
                    10.0,
                ),
        }

        (
            current,
            teacher_input,
        ) = self.observations.build(
            terms
        )

        frame_major = torch.cat(
            list(
                self.observations
                .history
                .values()
            ),
            dim=-1,
        )

        return (
            current,
            frame_major,
            teacher_input,
        )

    def resample_commands(self, env_ids: torch.Tensor) -> None:
        """Sample new commands without resetting physical state or history."""
        if len(env_ids) == 0:
            return
        for column, name in enumerate(("lin_vel_x", "ang_vel_z", "height")):
            low, high = self.command_ranges[name]
            sampled = torch.empty(
                len(env_ids),
                device=self.device,
                dtype=self.command.dtype,
            ).uniform_(float(low), float(high), generator=self.rng)
            self.command[env_ids, column] = sampled

    def reset(
        self,
        env_ids: torch.Tensor,
    ) -> None:
        if len(env_ids) == 0:
            return

        root_pose = (
            self.robot
            .data
            .default_root_pose
            .torch[
                env_ids
            ]
            .clone()
        )

        root_pose[:, :3] += (
            self.scene
            .env_origins[
                env_ids
            ]
        )

        root_velocity = (
            self.robot
            .data
            .default_root_vel
            .torch[
                env_ids
            ]
            .clone()
        )

        q = (
            self.robot
            .data
            .default_joint_pos
            .torch[
                env_ids
            ]
            .clone()
        )

        qd = (
            self.robot
            .data
            .default_joint_vel
            .torch[
                env_ids
            ]
            .clone()
        )

        self.robot.write_root_link_pose_to_sim_index(
            root_pose=root_pose,
            env_ids=env_ids,
        )

        self.robot.write_root_link_velocity_to_sim_index(
            root_velocity=root_velocity,
            env_ids=env_ids,
        )

        self.robot.write_joint_position_to_sim_index(
            position=q,
            env_ids=env_ids,
        )

        self.robot.write_joint_velocity_to_sim_index(
            velocity=qd,
            env_ids=env_ids,
        )

        self.resample_commands(env_ids)

        self.last_action[
            env_ids
        ] = 0.0

        self.steps[
            env_ids
        ] = 0

        self.elapsed[
            env_ids
        ] = 0.0

        self.observations.reset(
            env_ids
        )

        self.scene.reset(
            env_ids
        )

        self.sim.forward()

        self.scene.update(
            0.0
        )

    def step(
        self,
        teacher_action: torch.Tensor,
        actor_observation: torch.Tensor,
        observation_history: torch.Tensor,
    ) -> dict[
        str,
        torch.Tensor,
    ]:
        state = self.state()

        action = (
            teacher_action.clamp(
                *self.contract.action_clip
            )
        )

        zeros = torch.zeros_like(
            action
        )

        spring_scale = float(
            os.environ.get(
                "SUPERMIMIC_GAS_SPRING_SCALE",
                "1.0",
            )
        )

        springs = (
            action.new_tensor(
                self.contract.gas_spring_targets
            )
            .expand(
                self.num_envs,
                -1,
            )
            * spring_scale
        )

        control = (
            policy_action_to_control(
                action,
                default_motor_targets=zeros,
                gas_spring_forces=springs,
                action_scales=action.new_tensor(
                    self.contract.action_scales
                ),
                action_targets=(
                    self.contract.action_targets
                ),
            )
        )

        target_position = (
            self.robot
            .data
            .joint_pos
            .torch
            .clone()
        )

        target_velocity = (
            torch.zeros_like(
                target_position
            )
        )

        target_position[
            :,
            self.policy_joint_ids[
                [
                    0,
                    1,
                    3,
                    4,
                ]
            ],
        ] = control[
            :,
            [
                0,
                1,
                3,
                4,
            ],
        ]

        target_velocity[
            :,
            self.policy_joint_ids[
                [
                    2,
                    5,
                ]
            ],
        ] = control[
            :,
            [
                2,
                5,
            ],
        ]

        self.robot.set_joint_position_target_index(
            target=target_position
        )

        self.robot.set_joint_velocity_target_index(
            target=target_velocity
        )

        self.robot.set_fixed_tendon_position_target_index(
            target=springs,
            fixed_tendon_ids=self.tendon_ids,
        )

        self.scene.write_data_to_sim()

        self.sim.step(
            render=True
        )

        self.scene.update(
            INFANTRY_SPEC.control_dt
        )

        next_state = self.state()

        self.steps += 1

        self.elapsed += (
            INFANTRY_SPEC.control_dt
        )

        finite = torch.isfinite(
            next_state
        ).all(
            dim=-1
        )

        terminated = (
            (~finite)
            | (
                next_state[:, 2]
                < 0.08
            )
        )

        truncated = (
            self.steps
            >= self.max_episode_steps
        )

        result = {
            "state":
                state,

            "next_state":
                next_state,

            "actor_observation":
                actor_observation,

            "observation_history":
                observation_history,

            "command":
                self.command.clone(),

            "executed_action":
                action,

            "applied_control":
                control,

            "terminated":
                terminated,

            "truncated":
                truncated,

            "context":
                self.context,

            "timestamp":
                self.elapsed
                - INFANTRY_SPEC.control_dt,
        }

        self.last_action = action

        return result
