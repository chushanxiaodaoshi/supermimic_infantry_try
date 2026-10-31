"""Focused tests for PACE audit/replay invariants."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from pace_id.core import (
  FLOAT_COLUMNS,
  LoadedSamples,
  Mapping,
  forward_kinematics,
  uint32_delta,
  wrap_angle,
)
from pace_id.diagnose import _failure_kind
from pace_id.model import ModelAdapter, _remove_free_joint_and_keyframe
from pace_id.replay import split_sweep_runs


ROOT = Path(__file__).resolve().parents[1]


def test_wrap_angle_handles_pi_boundary() -> None:
  wrapped = wrap_angle(np.array((math.pi + 0.1, -math.pi - 0.1)))
  np.testing.assert_allclose(wrapped, (-math.pi + 0.1, math.pi - 0.1))


def test_uint32_delta_accepts_counter_wrap() -> None:
  assert uint32_delta(3, (1 << 32) - 2) == 5


def test_mapping_uses_solver_coordinates_and_matches_policy_check() -> None:
  mapping = Mapping()
  q1 = np.array((2.3952019214630127, 1.948998212814331))
  q2 = np.array((1.3880798816680908, 0.9370825290679932))
  can3, can2 = mapping.qpos_from_solver(q1, q2)
  np.testing.assert_allclose(can3, (-0.6882019, -0.2419982), atol=1e-6)
  np.testing.assert_allclose(can2, (-1.4049799, -0.9539825), atol=1e-6)


def test_solver_forward_kinematics_matches_known_log_sample() -> None:
  length, phi = forward_kinematics(0.6607222557067871, -0.603632926940918)
  assert math.isclose(float(length), 0.32151320576667786, abs_tol=2e-7)
  assert math.isclose(float(phi), -1.542251706123352, abs_tol=2e-7)


def test_fixed_xml_transform_requires_exact_base_joint() -> None:
  xml = '<mujoco><worldbody><joint name="robot/base_free_joint" '
  xml += 'type="free" /></worldbody><keyframe><key /></keyframe></mujoco>'
  fixed = _remove_free_joint_and_keyframe(xml)
  assert "base_free_joint" not in fixed
  assert "keyframe" not in fixed


def test_model_has_no_duplicate_compensation_and_keeps_wheels_free() -> None:
  adapter = ModelAdapter.create(ROOT / "mjcf/scene.xml", Mapping())
  validation = adapter.validation
  assert validation["base_fixed"]
  assert validation["wheel_joints_free"]
  assert validation["policy_actuators_disabled"]
  assert validation["duplicate_compensation_check_passed"]
  assert validation["gas_spring"]["implementation"] == (
    "constant tendon actuator force"
  )
  assert validation["gas_spring"]["native_tendon_stiffness"] == [0.0, 0.0]


def test_closure_svd_excludes_structural_zeros_from_condition() -> None:
  adapter = ModelAdapter.create(ROOT / "mjcf/scene.xml", Mapping())
  data = adapter.new_data()
  adapter.initialize(data, -0.6882019, -1.4049799, 0.0, 0.0)
  result = adapter.closure_diagnostics(data)
  assert result["effective_rank"] > 0
  assert result["structural_zero_count"] > 0
  assert result["effective_rank"] + result["structural_zero_count"] == 6
  assert math.isfinite(result["effective_condition_number"])


def test_time_reset_is_a_distinct_numerical_failure() -> None:
  adapter = ModelAdapter.create(ROOT / "mjcf/scene.xml", Mapping())
  data = adapter.new_data()
  kind, details = _failure_kind(data, 1.0, 0.0025)
  assert kind == "automatic_reset_or_time_jump"
  assert details["automatic_time_reset_or_jump"]


def test_sequence_gap_splits_sweep_instead_of_concatenating() -> None:
  integer = np.array((1, 2, 4, 5), dtype=np.int64)
  samples = LoadedSamples(
    sequence=integer,
    device_time_us=np.array((10, 20, 30, 40), dtype=np.int64),
    phase=np.full(4, 4, dtype=np.int64),
    flags=np.full(4, 0x17, dtype=np.int64),
    fault=np.zeros(4, dtype=np.int64),
    workpoint=np.zeros(4, dtype=np.int64),
    kinematics_valid=np.ones(4, dtype=np.int64),
    output_saturated=np.zeros(4, dtype=np.int64),
    fields={name: np.zeros(4) for name in FLOAT_COLUMNS},
  )
  runs = split_sweep_runs(samples)
  assert [run.tolist() for run in runs] == [[0, 1], [2, 3]]
