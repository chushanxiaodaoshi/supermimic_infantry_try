"""Shared data structures and coordinate transforms for PACE identification."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


UINT32_MOD = 1 << 32
PHI_SWEEP = 4
LENGTH_SWEEP = 7
SWEEP_PHASES = (PHI_SWEEP, LENGTH_SWEEP)
PHASE_NAMES = {
  0: "idle",
  1: "auto_align_down",
  2: "auto_move_length",
  3: "pre_sweep_settle",
  4: "phi_sweep",
  5: "intermission",
  6: "fault",
  7: "length_sweep",
  8: "final_hold",
  9: "auto_move_phi",
}

REQUIRED_COLUMNS = (
  "sequence",
  "device_time_us",
  "phase",
  "flags",
  "fault",
  "workpoint",
  "kinematics_valid",
  "output_saturated",
  "target_leg_length_m",
  "target_leg_phi_rad",
  "solver_q1_rad",
  "solver_q2_rad",
  "measured_leg_length_m",
  "measured_leg_phi_rad",
  "measured_leg_length_rate_m_s",
  "measured_leg_phi_rate_rad_s",
  "virtual_force_command_n",
  "virtual_torque_command_nm",
  "can3_command_nm",
  "can3_feedback_nm",
  "can3_policy_angle_rad",
  "can3_speed_rad_s",
  "can2_command_nm",
  "can2_feedback_nm",
  "can2_policy_angle_rad",
  "can2_speed_rad_s",
)

FLOAT_COLUMNS = REQUIRED_COLUMNS[8:]
INT_COLUMNS = REQUIRED_COLUMNS[:8]


@dataclass(frozen=True)
class Mapping:
  """Verified physical-ID mapping expressed without left/right assumptions."""

  model_side: str = "Left"
  can3_joint_suffix: str = "front_joint"
  can2_joint_suffix: str = "rear_joint"
  can3_qpos_scale: float = -1.0
  can3_qpos_offset: float = 1.707
  can2_qpos_scale: float = -1.0
  can2_qpos_offset: float = -0.0169
  can3_torque_sign: float = 1.0
  can2_torque_sign: float = 1.0

  @property
  def can3_joint(self) -> str:
    return f"robot/{self.model_side}_{self.can3_joint_suffix}"

  @property
  def can2_joint(self) -> str:
    return f"robot/{self.model_side}_{self.can2_joint_suffix}"

  def qpos_from_solver(
    self, q1: np.ndarray | float, q2: np.ndarray | float
  ) -> tuple[np.ndarray, np.ndarray]:
    q_can3 = self.can3_qpos_scale * np.asarray(q1) + self.can3_qpos_offset
    q_can2 = self.can2_qpos_scale * np.asarray(q2) + self.can2_qpos_offset
    return wrap_angle(q_can3), wrap_angle(q_can2)

  def as_dict(self) -> dict[str, object]:
    return {
      "physical_mapping": {
        "CAN3": "solver q1/tau1",
        "CAN2": "solver q2/tau2",
      },
      "model_side_label": self.model_side,
      "model_label_warning": (
        "The XML side label is a model coordinate label, not a hardware ID."
      ),
      "can3_model_joint": self.can3_joint,
      "can2_model_joint": self.can2_joint,
      "qpos_from_solver": {
        "CAN3": {
          "scale": self.can3_qpos_scale,
          "offset_rad": self.can3_qpos_offset,
        },
        "CAN2": {
          "scale": self.can2_qpos_scale,
          "offset_rad": self.can2_qpos_offset,
        },
      },
      "torque_sign_to_model": {
        "CAN3": self.can3_torque_sign,
        "CAN2": self.can2_torque_sign,
      },
    }


@dataclass(frozen=True)
class Sample:
  row: int
  sequence: int
  device_time_us: int
  phase: int
  flags: int
  fault: int
  workpoint: int
  kinematics_valid: int
  output_saturated: int
  values: tuple[float, ...]

  def value(self, name: str) -> float:
    return self.values[FLOAT_COLUMNS.index(name)]


@dataclass(frozen=True)
class LoadedSamples:
  """Dense arrays used only after the streaming audit has accepted a log."""

  sequence: np.ndarray
  device_time_us: np.ndarray
  phase: np.ndarray
  flags: np.ndarray
  fault: np.ndarray
  workpoint: np.ndarray
  kinematics_valid: np.ndarray
  output_saturated: np.ndarray
  fields: dict[str, np.ndarray]

  def __len__(self) -> int:
    return int(self.sequence.size)


def wrap_angle(value: np.ndarray | float) -> np.ndarray:
  array = np.asarray(value, dtype=np.float64)
  return (array + math.pi) % (2.0 * math.pi) - math.pi


def uint32_delta(current: int, previous: int) -> int:
  return (current - previous) % UINT32_MOD


def forward_kinematics(
  q1: np.ndarray | float,
  q2: np.ndarray | float,
  upper_m: float = 0.1747,
  lower_m: float = 0.2080,
) -> tuple[np.ndarray, np.ndarray]:
  """Match the firmware LegSolver forward-kinematics branch."""
  q1_array = np.asarray(q1, dtype=np.float64)
  q2_array = np.asarray(q2, dtype=np.float64)
  x_a = upper_m * np.cos(q1_array)
  y_a = upper_m * np.sin(q1_array)
  x_b = upper_m * np.cos(q2_array)
  y_b = upper_m * np.sin(q2_array)
  dx = x_b - x_a
  dy = y_b - y_a
  a = 2.0 * lower_m * dx
  b = 2.0 * lower_m * dy
  c = dx * dx + dy * dy
  root = np.sqrt(np.maximum(a * a + b * b - c * c, 0.0))
  phi_a = 2.0 * np.arctan2(b + root, a + c)
  x_p = x_a + lower_m * np.cos(phi_a)
  y_p = y_a + lower_m * np.sin(phi_a)
  length = np.sqrt(x_p * x_p + y_p * y_p)
  phi = wrap_angle(np.arctan2(y_p, x_p) - 0.5 * math.pi)
  return length, phi


def _data_lines(stream: Iterator[str]) -> Iterator[str]:
  for line in stream:
    if not line.startswith("#"):
      yield line


def iter_samples(path: Path) -> Iterator[Sample]:
  with path.open(newline="", encoding="utf-8") as stream:
    reader = csv.DictReader(_data_lines(iter(stream)))
    if reader.fieldnames is None:
      raise ValueError(f"PACE CSV has no data header: {path}")
    missing = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames))
    if missing:
      raise ValueError(f"PACE CSV is missing columns: {', '.join(missing)}")
    for row_index, row in enumerate(reader, start=1):
      try:
        integers = tuple(int(row[name]) for name in INT_COLUMNS)
        values = tuple(float(row[name]) for name in FLOAT_COLUMNS)
      except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric value at data row {row_index}") from error
      yield Sample(row_index, *integers, values)


def load_samples(path: Path, phases: Sequence[int] = SWEEP_PHASES) -> LoadedSamples:
  integer_values: list[list[int]] = [[] for _ in INT_COLUMNS]
  float_values: dict[str, list[float]] = {name: [] for name in FLOAT_COLUMNS}
  selected = set(phases)
  for sample in iter_samples(path):
    if sample.phase not in selected:
      continue
    for target, value in zip(integer_values, (
      sample.sequence,
      sample.device_time_us,
      sample.phase,
      sample.flags,
      sample.fault,
      sample.workpoint,
      sample.kinematics_valid,
      sample.output_saturated,
    ), strict=True):
      target.append(value)
    for name, value in zip(FLOAT_COLUMNS, sample.values, strict=True):
      float_values[name].append(value)
  if not integer_values[0]:
    raise ValueError("PACE CSV has no sweep samples")
  integers = [np.asarray(values, dtype=np.int64) for values in integer_values]
  fields = {
    name: np.asarray(values, dtype=np.float64)
    for name, values in float_values.items()
  }
  return LoadedSamples(*integers, fields)


def file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
  path.write_text(
    json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    encoding="utf-8",
  )


def create_output_dir(path: Path) -> Path:
  resolved = path.resolve()
  resolved.mkdir(parents=True, exist_ok=False)
  return resolved
