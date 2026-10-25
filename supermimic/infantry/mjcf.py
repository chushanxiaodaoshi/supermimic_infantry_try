"""Static and MuJoCo-backed validation for the infantry MJCF asset."""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .spec import (
    GAS_SPRING_TENDON_NAMES,
    INFANTRY_JOINT_NAMES,
    LOOP_EQUALITY_NAMES,
    POLICY_ACTION_NAMES,
)


@dataclass(frozen=True)
class MjcfAudit:
    path: str
    nq: int
    nv: int
    nu: int
    nbody: int
    njnt: int
    neq: int
    ntendon: int
    timestep: float
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    equality_names: tuple[str, ...]
    tendon_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_mjcf(path: str | Path) -> MjcfAudit:
    """Compile the source MJCF and fail if required mechanics are absent."""
    try:
        mujoco: Any = importlib.import_module("mujoco")
    except ImportError as exc:  # pragma: no cover - exercised in deployment setup
        raise RuntimeError("MuJoCo is required to audit the infantry asset") from exc

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    model = mujoco.MjModel.from_xml_path(str(source))

    def names(object_type: Any, count: int) -> tuple[str, ...]:
        return tuple(
            name
            for index in range(count)
            if (name := mujoco.mj_id2name(model, object_type, index)) is not None
        )

    result = MjcfAudit(
        path=str(source),
        nq=model.nq,
        nv=model.nv,
        nu=model.nu,
        nbody=model.nbody,
        njnt=model.njnt,
        neq=model.neq,
        ntendon=model.ntendon,
        timestep=float(model.opt.timestep),
        joint_names=names(mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
        actuator_names=names(mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),
        equality_names=names(mujoco.mjtObj.mjOBJ_EQUALITY, model.neq),
        tendon_names=names(mujoco.mjtObj.mjOBJ_TENDON, model.ntendon),
    )
    missing_joints = set(INFANTRY_JOINT_NAMES) - set(result.joint_names)
    missing_actions = set(POLICY_ACTION_NAMES) - set(result.joint_names)
    missing_equalities = set(LOOP_EQUALITY_NAMES) - set(result.equality_names)
    missing_tendons = set(GAS_SPRING_TENDON_NAMES) - set(result.tendon_names)
    failures = {
        "joints": sorted(missing_joints),
        "policy joints": sorted(missing_actions),
        "loop equalities": sorted(missing_equalities),
        "gas-spring tendons": sorted(missing_tendons),
    }
    failures = {label: values for label, values in failures.items() if values}
    if failures:
        raise ValueError(f"MJCF is missing required infantry mechanics: {failures}")
    return result
