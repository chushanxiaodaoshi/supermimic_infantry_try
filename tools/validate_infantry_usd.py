"""Offline structural validation for the converted infantry USD package."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import isaacsim.asset.importer.utils.impl as importer_utils
from pxr import Plug, Usd, UsdPhysics, UsdUtils

JOINT_NAMES = {
    "Left_rear_joint",
    "Left_rear_child1_joint",
    "Left_Wheel_joint",
    "Right_rear_joint",
    "Right_rear_child1_joint",
    "Right_Wheel_joint",
    "Left_front_joint",
    "Left_front_child1_joint",
    "Left_front_child2_joint",
    "Left_front_child3_joint",
    "Right_front_joint",
    "Right_front_child1_joint",
    "Right_front_child2_joint",
    "Right_front_joint3_joint",
}
EQUALITY_NAMES = {"Left_loop1", "Left_loop2", "Right_loop1", "Right_loop2"}
TENDON_NAMES = {"Left_gas_spring_tendon", "Right_gas_spring_tendon"}


def require_names(label: str, actual: set[str], expected: set[str]) -> None:
    if actual != expected:
        raise ValueError(
            f"unexpected {label}: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("usd", type=Path)
    args = parser.parse_args()
    usd = args.usd.expanduser().resolve()
    package_root = usd.parent
    if not usd.is_file():
        raise FileNotFoundError(usd)

    Plug.Registry().RegisterPlugins(
        os.path.join(os.path.dirname(importer_utils.__file__), "schemas")
    )
    stage = Usd.Stage.Open(str(usd))
    if stage is None:
        raise ValueError(f"USD could not be opened: {usd}")
    root = stage.GetDefaultPrim()
    if not root:
        raise ValueError("USD has no default prim")
    physics = root.GetVariantSets().GetVariantSet("Physics")
    if "mujoco" not in physics.GetVariantNames():
        raise ValueError("USD has no Physics=mujoco variant")
    physics.SetVariantSelection("mujoco")
    prims = list(stage.Traverse())
    articulation_roots = [
        prim for prim in prims if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if len(articulation_roots) != 1:
        raise ValueError(f"expected one articulation root, got {articulation_roots}")
    joints = {
        prim.GetName() for prim in prims if prim.GetTypeName() == "PhysicsRevoluteJoint"
    }
    equalities = {
        prim.GetName()
        for prim in prims
        if prim.GetTypeName() == "PhysicsSphericalJoint"
    }
    tendons = {prim.GetName() for prim in prims if prim.GetTypeName() == "MjcTendon"}
    actuators = [
        prim.GetName() for prim in prims if prim.GetTypeName() == "MjcActuator"
    ]
    require_names("revolute joints", joints, JOINT_NAMES)
    require_names("closed-loop equalities", equalities, EQUALITY_NAMES)
    require_names("spatial tendons", tendons, TENDON_NAMES)
    if len(actuators) != 8:
        raise ValueError(f"expected eight MjcActuator prims, got {actuators}")

    rigid_bodies = [prim for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    masses: dict[str, float] = {}
    for prim in rigid_bodies:
        mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
        if mass is not None:
            mass = float(mass)
            if not (0.0 < mass < 1000.0):
                raise ValueError(f"invalid mass {mass} on {prim.GetPath()}")
            masses[str(prim.GetPath())] = mass
    if len(masses) != 15:
        raise ValueError(f"expected masses for 15 robot bodies, got {len(masses)}")

    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(usd))
    if unresolved:
        raise ValueError(f"unresolved USD dependencies: {unresolved}")
    external = []
    for layer in layers:
        path = Path(layer.realPath).resolve()
        if not path.is_relative_to(package_root):
            external.append(str(path))
    for asset in assets:
        path = Path(str(asset)).resolve()
        if not path.is_relative_to(package_root):
            external.append(str(path))
    if external:
        raise ValueError(f"dependencies outside package: {external}")

    result = {
        "usd": str(usd),
        "default_prim": root.GetPath().pathString,
        "physics_variants": physics.GetVariantNames(),
        "articulation_root": str(articulation_roots[0].GetPath()),
        "revolute_joints": sorted(joints),
        "closed_loop_equalities": sorted(equalities),
        "spatial_tendons": sorted(tendons),
        "actuators": sorted(actuators),
        "rigid_body_mass_count": len(masses),
        "dependency_layers": len(layers),
        "dependency_assets": len(assets),
        "unresolved_dependencies": 0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
