"""Compile and compare the task-scene and extracted infantry MJCF models."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np


def object_names(
    model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int
) -> list[str]:
    return [
        name
        for index in range(count)
        if (name := mujoco.mj_id2name(model, object_type, index)) is not None
    ]


def named_rows(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    names: list[str],
    values: np.ndarray,
    prefix: str = "",
) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(values[mujoco.mj_name2id(model, object_type, prefix + name)])
        for name in names
    }


def compare_named(
    label: str, left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> None:
    if left.keys() != right.keys():
        raise ValueError(f"{label} names differ")
    for name, value in left.items():
        np.testing.assert_array_equal(value, right[name], err_msg=f"{label}: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--robot", required=True, type=Path)
    parser.add_argument("--source-prefix", default="")
    args = parser.parse_args()
    source = mujoco.MjModel.from_xml_path(str(args.source.resolve()))
    robot = mujoco.MjModel.from_xml_path(str(args.robot.resolve()))

    joint_names = object_names(robot, mujoco.mjtObj.mjOBJ_JOINT, robot.njnt)
    body_names = object_names(robot, mujoco.mjtObj.mjOBJ_BODY, robot.nbody)
    geom_names = [
        name
        for name in object_names(robot, mujoco.mjtObj.mjOBJ_GEOM, robot.ngeom)
        if name != "base_link_visual"
    ]
    visual_id = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_GEOM, "base_link_visual")
    if visual_id < 0:
        raise ValueError("robot model has no separate base_link_visual geom")
    if robot.geom_contype[visual_id] or robot.geom_conaffinity[visual_id]:
        raise ValueError("base_link_visual must not participate in collisions")
    for name in joint_names:
        source_id = mujoco.mj_name2id(
            source, mujoco.mjtObj.mjOBJ_JOINT, args.source_prefix + name
        )
        robot_id = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_JOINT, name)
        source_dof = source.jnt_dofadr[source_id]
        robot_dof = robot.jnt_dofadr[robot_id]
        np.testing.assert_array_equal(
            source.jnt_range[source_id], robot.jnt_range[robot_id]
        )
        np.testing.assert_array_equal(
            source.dof_damping[source_dof], robot.dof_damping[robot_dof]
        )
        np.testing.assert_array_equal(
            source.dof_frictionloss[source_dof], robot.dof_frictionloss[robot_dof]
        )
    compare_named(
        "body inertial",
        named_rows(
            source,
            mujoco.mjtObj.mjOBJ_BODY,
            body_names,
            np.column_stack((source.body_mass, source.body_inertia, source.body_ipos)),
            args.source_prefix,
        ),
        named_rows(
            robot,
            mujoco.mjtObj.mjOBJ_BODY,
            body_names,
            np.column_stack((robot.body_mass, robot.body_inertia, robot.body_ipos)),
        ),
    )
    compare_named(
        "geom properties",
        named_rows(
            source,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_names,
            np.column_stack(
                (source.geom_friction, source.geom_solref, source.geom_solimp)
            ),
            args.source_prefix,
        ),
        named_rows(
            robot,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_names,
            np.column_stack(
                (robot.geom_friction, robot.geom_solref, robot.geom_solimp)
            ),
        ),
    )
    actuator_names = object_names(robot, mujoco.mjtObj.mjOBJ_ACTUATOR, robot.nu)
    compare_named(
        "actuator parameters",
        named_rows(
            source,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_names,
            np.column_stack(
                (
                    source.actuator_ctrlrange,
                    source.actuator_forcerange,
                    source.actuator_gainprm,
                    source.actuator_biasprm,
                )
            ),
            args.source_prefix,
        ),
        named_rows(
            robot,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_names,
            np.column_stack(
                (
                    robot.actuator_ctrlrange,
                    robot.actuator_forcerange,
                    robot.actuator_gainprm,
                    robot.actuator_biasprm,
                )
            ),
        ),
    )

    summary = {
        "bodies": robot.nbody,
        "joints_including_free": robot.njnt,
        "position_dofs": robot.nq,
        "velocity_dofs": robot.nv,
        "actuators": actuator_names,
        "equalities": object_names(robot, mujoco.mjtObj.mjOBJ_EQUALITY, robot.neq),
        "tendons": object_names(robot, mujoco.mjtObj.mjOBJ_TENDON, robot.ntendon),
        "meshes": robot.nmesh,
        "geoms": robot.ngeom,
    }
    print(summary)
    print("Robot dynamics parameters match the source model exactly by name.")


if __name__ == "__main__":
    main()
