"""Read the ONNX deployment contract from mjlab's replay source.

The replay program remains the source of truth.  Parsing its constants avoids
maintaining a second actuator configuration in this repository.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_REQUIRED = (
    "LEG_JOINTS",
    "POLICY_JOINTS",
    "ACTION_TARGETS",
    "ACTION_SCALES",
    "POLICY_TORQUE_LIMITS",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Call):
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "np"
            and func.attr == "array"
            and len(node.args) == 1
            and not node.keywords
        ):
            return ast.literal_eval(node.args[0])
        raise ValueError("only literal np.array(...) calls are accepted")
    return ast.literal_eval(node)


def extract_replay_constants(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, Any] = {}
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if statement.value is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in _REQUIRED:
                values[target.id] = _literal(statement.value)
    missing = sorted(set(_REQUIRED) - values.keys())
    if missing:
        raise ValueError(f"replay source is missing required constants: {missing}")
    return values


@dataclass(frozen=True)
class ReplayContract:
    source: str
    source_sha256: str
    leg_joints: tuple[str, ...]
    policy_joints: tuple[str, ...]
    action_targets: tuple[str, ...]
    action_scales: tuple[float, ...]
    torque_limits: tuple[float, ...]
    observation_terms: tuple[str, ...] = (
        "base_ang_vel",
        "projected_gravity",
        "command",
        "leg_joint_pos",
        "joint_vel",
        "last_action",
    )
    observation_term_dims: tuple[int, ...] = (3, 3, 3, 4, 6, 6)
    current_observation_dim: int = 25
    history_length: int = 5
    history_layout: str = "term-major"
    action_clip: tuple[float, float] = (-100.0, 100.0)
    gas_spring_targets: tuple[float, float] = (300.0, 300.0)

    @property
    def observation_dim(self) -> int:
        return self.current_observation_dim * (self.history_length + 1)

    @classmethod
    def from_replay(cls, path: str | Path) -> ReplayContract:
        source = Path(path).expanduser().resolve()
        constants = extract_replay_constants(source)
        contract = cls(
            source=str(source),
            source_sha256=sha256_file(source),
            leg_joints=tuple(constants["LEG_JOINTS"]),
            policy_joints=tuple(constants["POLICY_JOINTS"]),
            action_targets=tuple(constants["ACTION_TARGETS"]),
            action_scales=tuple(float(v) for v in constants["ACTION_SCALES"]),
            torque_limits=tuple(float(v) for v in constants["POLICY_TORQUE_LIMITS"]),
        )
        contract.validate()
        return contract

    def validate(self) -> None:
        if len(self.action_targets) != 6 or len(set(self.action_targets)) != 6:
            raise ValueError("replay contract must define six unique action targets")
        for name, values in (
            ("action_scales", self.action_scales),
            ("torque_limits", self.torque_limits),
        ):
            if len(values) != 6 or not all(value > 0.0 for value in values):
                raise ValueError(f"{name} must contain six positive values")
        if sum(self.observation_term_dims) != self.current_observation_dim:
            raise ValueError("observation term dimensions do not sum to current width")
        if self.observation_dim != 150:
            raise ValueError(f"expected replay observation width 150, got {self.observation_dim}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"observation_dim": self.observation_dim}

    def write_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def validate_onnx_contract(path: str | Path, contract: ReplayContract) -> dict[str, Any]:
    """Check model dimensions and identity without starting ONNX Runtime."""
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - optional development dependency
        raise RuntimeError("install onnx to validate Teacher files") from exc

    model_path = Path(path).expanduser().resolve()
    model = onnx.load(str(model_path), load_external_data=False)
    initializers = {value.name for value in model.graph.initializer}
    inputs = [value for value in model.graph.input if value.name not in initializers]
    outputs = list(model.graph.output)
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(f"{model_path}: expected one input and one output")

    def shape(value: Any) -> list[int | str | None]:
        result: list[int | str | None] = []
        for dim in value.type.tensor_type.shape.dim:
            result.append(dim.dim_value or dim.dim_param or None)
        return result

    input_shape, output_shape = shape(inputs[0]), shape(outputs[0])
    if len(input_shape) != 2 or input_shape[-1] != contract.observation_dim:
        raise ValueError(
            f"{model_path}: Teacher input must be [batch,{contract.observation_dim}], "
            f"got {input_shape}"
        )
    if len(output_shape) != 2 or output_shape[-1] != len(contract.action_targets):
        raise ValueError(f"{model_path}: Teacher output must end in 6, got {output_shape}")
    return {
        "path": str(model_path),
        "sha256": sha256_file(model_path),
        "input_name": inputs[0].name,
        "input_shape": input_shape,
        "output_name": outputs[0].name,
        "output_shape": output_shape,
        "metadata": {item.key: item.value for item in model.metadata_props},
    }
