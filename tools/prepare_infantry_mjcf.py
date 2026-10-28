"""Extract the infantry robot from its mjlab robot-description MJCF.

The source asset also contains optional task-course bodies. This tool retains
only the robot description and refuses replay-scene namespaced models so that
scene/controller configuration cannot leak into the USD asset.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

JOINT_NAMES = (
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
)
EQUALITY_NAMES = ("Left_loop1", "Left_loop2", "Right_loop1", "Right_loop2")
TENDON_NAMES = ("Left_gas_spring_tendon", "Right_gas_spring_tendon")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def names(elements: list[ET.Element]) -> tuple[str, ...]:
    return tuple(element.attrib["name"] for element in elements)


def require_equal(
    label: str, actual: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    if actual != expected:
        raise ValueError(f"unexpected {label}: expected {expected}, got {actual}")


def prepare(source: Path, destination: Path, report_path: Path) -> None:
    root = ET.parse(source).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no worldbody")
    robot = worldbody.find("./body[@name='base_link']")
    if robot is None:
        raise ValueError("robot-description MJCF has no unprefixed base_link body")
    require_equal("robot joint order", names(robot.findall(".//joint")), JOINT_NAMES)

    equality = root.find("equality")
    if equality is None:
        raise ValueError("MJCF has no equality section")
    equalities = {element.attrib.get("name"): element for element in list(equality)}
    missing = [name for name in EQUALITY_NAMES if name not in equalities]
    if missing:
        raise ValueError(f"missing closed-loop equalities: {missing}")
    tendon = root.find("tendon")
    if tendon is None:
        raise ValueError("MJCF has no tendon section")
    require_equal(
        "spatial tendon order", names(tendon.findall("spatial")), TENDON_NAMES
    )
    actuator = root.find("actuator")
    if actuator is None or len(list(actuator)) != 8:
        raise ValueError("expected exactly eight robot actuators")

    referenced_meshes = {
        geom.attrib["mesh"]
        for geom in robot.findall(".//geom")
        if "mesh" in geom.attrib
    }
    asset = root.find("asset")
    if asset is None:
        raise ValueError("MJCF has no asset section")
    meshes = {element.attrib.get("name"): element for element in asset.findall("mesh")}
    missing_meshes = sorted(referenced_meshes - meshes.keys())
    if missing_meshes:
        raise ValueError(f"robot references missing meshes: {missing_meshes}")
    procedural = sorted(
        name for name in referenced_meshes if "file" not in meshes[name].attrib
    )
    if procedural:
        raise ValueError(
            f"robot references procedural meshes without files: {procedural}"
        )

    output = ET.Element(root.tag, root.attrib)
    for tag in ("compiler", "option", "size", "default", "visual"):
        element = root.find(tag)
        if element is not None:
            output.append(copy.deepcopy(element))
    output_asset = ET.SubElement(output, "asset")
    for element in list(asset):
        if element.tag == "mesh" and element.attrib.get("name") in referenced_meshes:
            output_asset.append(copy.deepcopy(element))
    output_worldbody = ET.SubElement(output, "worldbody")
    output_worldbody.append(copy.deepcopy(robot))
    output_equality = ET.SubElement(output, "equality")
    for name in EQUALITY_NAMES:
        output_equality.append(copy.deepcopy(equalities[name]))
    for tag in ("tendon", "actuator", "sensor"):
        element = root.find(tag)
        if element is not None:
            output.append(copy.deepcopy(element))

    destination.parent.mkdir(parents=True, exist_ok=True)
    output_tree = ET.ElementTree(output)
    ET.indent(output_tree, space="  ")
    output_tree.write(destination, encoding="utf-8", xml_declaration=True)

    copied_assets: list[str] = []
    for name in sorted(referenced_meshes):
        relative = Path(meshes[name].attrib["file"])
        source_asset = source.parent / relative
        destination_asset = destination.parent / relative
        if not source_asset.is_file():
            raise FileNotFoundError(source_asset)
        destination_asset.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_asset, destination_asset)
        copied_assets.append(relative.as_posix())

    contact = root.find("contact")
    report = {
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(destination),
        "output_sha256": sha256(destination),
        "preserved": {
            "root_body": "base_link",
            "joints": list(JOINT_NAMES),
            "closed_loop_equalities": list(EQUALITY_NAMES),
            "spatial_tendons": list(TENDON_NAMES),
            "actuator_count": len(list(actuator)),
            "mesh_count": len(referenced_meshes),
            "mesh_files": copied_assets,
        },
        "removed_scene_content": {
            "top_level_bodies": [
                body.attrib.get("name", "<unnamed>")
                for body in worldbody.findall("body")
                if body is not robot
            ],
            "meshes": sorted(set(meshes) - referenced_meshes),
            "contact_entries": len(list(contact)) if contact is not None else 0,
            "equalities": sorted(set(equalities) - set(EQUALITY_NAMES)),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    prepare(
        args.src.expanduser().resolve(),
        args.dst.expanduser().resolve(),
        args.report.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
