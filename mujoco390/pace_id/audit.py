"""Streaming integrity and excitation audit for PACE CSV logs."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import numpy as np

from .core import (
  LENGTH_SWEEP,
  Mapping,
  PHASE_NAMES,
  PHI_SWEEP,
  SWEEP_PHASES,
  Sample,
  file_sha256,
  forward_kinematics,
  iter_samples,
  uint32_delta,
  wrap_angle,
)


@dataclass
class Range:
  minimum: float = math.inf
  maximum: float = -math.inf

  def update(self, value: float) -> None:
    self.minimum = min(self.minimum, value)
    self.maximum = max(self.maximum, value)

  @property
  def ptp(self) -> float:
    if not math.isfinite(self.minimum):
      return math.nan
    return self.maximum - self.minimum


@dataclass
class SweepRun:
  workpoint: int
  phase: int
  start_row: int
  end_row: int
  rows: int = 0
  duration_s: float = 0.0
  faults: int = 0
  saturated: int = 0
  invalid_kinematics: int = 0
  nonfinite: int = 0
  work_false: int = 0
  can_offline: int = 0
  target: Range = field(default_factory=Range)
  measured: Range = field(default_factory=Range)
  cross: Range = field(default_factory=Range)
  can3_command: Range = field(default_factory=Range)
  can2_command: Range = field(default_factory=Range)

  def update(self, sample: Sample, dt_s: float) -> None:
    self.end_row = sample.row
    self.rows += 1
    self.duration_s += dt_s
    self.faults += int(sample.fault != 0 or bool(sample.flags & 0x08))
    self.saturated += int(bool(sample.output_saturated or sample.flags & 0x20))
    self.invalid_kinematics += int(not sample.kinematics_valid)
    self.work_false += int(not bool(sample.flags & 0x01))
    self.can_offline += int((sample.flags & 0x06) != 0x06)
    values = np.asarray(sample.values)
    if not np.isfinite(values).all():
      self.nonfinite += 1
      return
    if self.phase == PHI_SWEEP:
      target_name = "target_leg_phi_rad"
      measured_name = "measured_leg_phi_rad"
      cross_name = "measured_leg_length_m"
    else:
      target_name = "target_leg_length_m"
      measured_name = "measured_leg_length_m"
      cross_name = "measured_leg_phi_rad"
    self.target.update(sample.value(target_name))
    self.measured.update(sample.value(measured_name))
    self.cross.update(sample.value(cross_name))
    self.can3_command.update(sample.value("can3_command_nm"))
    self.can2_command.update(sample.value("can2_command_nm"))

  def as_dict(
    self,
    minimum_duration_s: float,
    minimum_phi_motion_rad: float,
    minimum_length_motion_m: float,
  ) -> dict[str, Any]:
    motion_limit = (
      minimum_phi_motion_rad if self.phase == PHI_SWEEP
      else minimum_length_motion_m
    )
    clean = not any((
      self.faults,
      self.saturated,
      self.invalid_kinematics,
      self.nonfinite,
      self.work_false,
      self.can_offline,
    ))
    complete = (
      self.duration_s >= minimum_duration_s
      and self.measured.ptp >= motion_limit
      and clean
    )
    return {
      "workpoint": self.workpoint,
      "phase": self.phase,
      "phase_name": PHASE_NAMES.get(self.phase, f"unknown_{self.phase}"),
      "start_row": self.start_row,
      "end_row": self.end_row,
      "rows": self.rows,
      "duration_s": self.duration_s,
      "fault_rows": self.faults,
      "saturated_rows": self.saturated,
      "invalid_kinematics_rows": self.invalid_kinematics,
      "nonfinite_rows": self.nonfinite,
      "work_false_rows": self.work_false,
      "can_offline_rows": self.can_offline,
      "target_ptp": self.target.ptp,
      "measured_ptp": self.measured.ptp,
      "cross_coordinate_ptp": self.cross.ptp,
      "can3_command_ptp_nm": self.can3_command.ptp,
      "can2_command_ptp_nm": self.can2_command.ptp,
      "clean_and_complete": complete,
    }


@dataclass
class MappingErrors:
  rows: int = 0
  policy_sq: float = 0.0
  policy_max: float = 0.0
  length_sq: float = 0.0
  length_max: float = 0.0
  phi_sq: float = 0.0
  phi_max: float = 0.0

  def update(self, sample: Sample, mapping: Mapping) -> None:
    if not sample.kinematics_valid:
      return
    names = (
      "solver_q1_rad",
      "solver_q2_rad",
      "can3_policy_angle_rad",
      "can2_policy_angle_rad",
      "measured_leg_length_m",
      "measured_leg_phi_rad",
    )
    values = [sample.value(name) for name in names]
    if not np.isfinite(values).all():
      return
    q1, q2, policy3, policy2, measured_l, measured_phi = values
    q3_model, q2_model = mapping.qpos_from_solver(q1, q2)
    policy_errors = np.abs(wrap_angle((q3_model - policy3, q2_model - policy2)))
    predicted_l, predicted_phi = forward_kinematics(q1, q2)
    length_error = abs(float(predicted_l) - measured_l)
    phi_error = abs(float(wrap_angle(float(predicted_phi) - measured_phi)))
    self.rows += 1
    self.policy_sq += float(np.sum(policy_errors**2))
    self.policy_max = max(self.policy_max, float(np.max(policy_errors)))
    self.length_sq += length_error**2
    self.length_max = max(self.length_max, length_error)
    self.phi_sq += phi_error**2
    self.phi_max = max(self.phi_max, phi_error)

  def as_dict(self) -> dict[str, float | int]:
    count = max(self.rows, 1)
    return {
      "rows_checked": self.rows,
      "solver_to_policy_qpos_rmse_rad": math.sqrt(self.policy_sq / (2 * count)),
      "solver_to_policy_qpos_max_abs_rad": self.policy_max,
      "solver_fk_length_rmse_m": math.sqrt(self.length_sq / count),
      "solver_fk_length_max_abs_m": self.length_max,
      "solver_fk_phi_rmse_rad": math.sqrt(self.phi_sq / count),
      "solver_fk_phi_max_abs_rad": self.phi_max,
    }


def audit_csv(
  path: Path,
  expected_workpoints: int = 30,
  maximum_continuous_dt_s: float = 0.05,
  minimum_sweep_duration_s: float = 19.0,
  minimum_phi_motion_rad: float = 0.02,
  minimum_length_motion_m: float = 0.001,
) -> dict[str, Any]:
  mapping = Mapping()
  mapping_errors = MappingErrors()
  rows = 0
  nonfinite_rows = 0
  fault_rows = 0
  saturated_rows = 0
  sequence_gap_events = 0
  estimated_missing_frames = 0
  reset_or_break_events = 0
  max_dt_s = 0.0
  segment_count = 0
  segment_start_row = 0
  segments: list[dict[str, Any]] = []
  runs: list[SweepRun] = []
  current_run: SweepRun | None = None
  previous: Sample | None = None
  previous_unwrapped_us = 0
  current_unwrapped_us = 0

  def close_segment(end_row: int, reason: str) -> None:
    nonlocal segment_count, segment_start_row
    if segment_start_row:
      segments.append({
        "index": segment_count,
        "start_row": segment_start_row,
        "end_row": end_row,
        "reason_ended": reason,
      })
    segment_count += 1
    segment_start_row = 0

  def close_run() -> None:
    nonlocal current_run
    if current_run is not None:
      runs.append(current_run)
      current_run = None

  for sample in iter_samples(path):
    rows += 1
    mapping_errors.update(sample, mapping)
    finite = np.isfinite(np.asarray(sample.values)).all()
    nonfinite_rows += int(not finite)
    fault_rows += int(sample.fault != 0 or bool(sample.flags & 0x08))
    saturated_rows += int(bool(sample.output_saturated or sample.flags & 0x20))
    continuous = previous is not None
    dt_s = 0.0
    break_reason = ""
    if previous is not None:
      sequence_delta = uint32_delta(sample.sequence, previous.sequence)
      time_delta_us = uint32_delta(sample.device_time_us, previous.device_time_us)
      dt_s = time_delta_us * 1.0e-6
      max_dt_s = max(max_dt_s, dt_s if dt_s < 1000.0 else 0.0)
      if sequence_delta != 1:
        sequence_gap_events += 1
        if 1 < sequence_delta < 1_000_000:
          estimated_missing_frames += sequence_delta - 1
          break_reason = "sequence_gap"
        else:
          break_reason = "sequence_reset_or_reverse"
      if time_delta_us == 0 or dt_s > maximum_continuous_dt_s:
        break_reason = break_reason or "device_time_reset_or_gap"
      continuous = not break_reason
      if continuous:
        current_unwrapped_us = previous_unwrapped_us + time_delta_us
      else:
        reset_or_break_events += 1
        close_run()
        close_segment(previous.row, break_reason)
        current_unwrapped_us = 0
    if not segment_start_row:
      segment_start_row = sample.row

    run_key = (sample.workpoint, sample.phase)
    current_key = None if current_run is None else (
      current_run.workpoint,
      current_run.phase,
    )
    if sample.phase in SWEEP_PHASES:
      if current_key != run_key or not continuous:
        close_run()
        current_run = SweepRun(
          workpoint=sample.workpoint,
          phase=sample.phase,
          start_row=sample.row,
          end_row=sample.row,
        )
      current_run.update(sample, dt_s if continuous else 0.0)
    else:
      close_run()
    previous = sample
    previous_unwrapped_us = current_unwrapped_us

  close_run()
  if previous is not None:
    close_segment(previous.row, "end_of_file")

  run_reports = [
    run.as_dict(
      minimum_sweep_duration_s,
      minimum_phi_motion_rad,
      minimum_length_motion_m,
    )
    for run in runs
  ]
  complete_keys = {
    (run["workpoint"], run["phase"])
    for run in run_reports
    if run["clean_and_complete"]
  }
  expected_keys = {
    (workpoint, phase)
    for workpoint in range(expected_workpoints)
    for phase in SWEEP_PHASES
  }
  mapping_report = mapping_errors.as_dict()
  mapping_ok = (
    mapping_report["solver_to_policy_qpos_max_abs_rad"] <= 0.002
    and mapping_report["solver_fk_length_max_abs_m"] <= 0.00005
    and mapping_report["solver_fk_phi_max_abs_rad"] <= 0.0005
  )
  complete = expected_keys <= complete_keys
  overall_ok = bool(
    rows
    and nonfinite_rows == 0
    and fault_rows == 0
    and saturated_rows == 0
    and reset_or_break_events == 0
    and mapping_ok
    and complete
  )
  return {
    "schema": "pace-audit-v1",
    "source": {
      "path": str(path.resolve()),
      "sha256": file_sha256(path),
    },
    "thresholds": {
      "expected_workpoints": expected_workpoints,
      "maximum_continuous_dt_s": maximum_continuous_dt_s,
      "minimum_sweep_duration_s": minimum_sweep_duration_s,
      "minimum_phi_motion_rad": minimum_phi_motion_rad,
      "minimum_length_motion_m": minimum_length_motion_m,
    },
    "integrity": {
      "rows": rows,
      "continuous_segments": len(segments),
      "sequence_gap_events": sequence_gap_events,
      "estimated_missing_frames": estimated_missing_frames,
      "reset_or_break_events": reset_or_break_events,
      "max_device_dt_s": max_dt_s,
      "nonfinite_rows": nonfinite_rows,
      "fault_rows": fault_rows,
      "saturated_rows": saturated_rows,
    },
    "segments": segments,
    "mapping": {
      **mapping.as_dict(),
      "errors": mapping_report,
      "hard_check_passed": mapping_ok,
    },
    "sweeps": run_reports,
    "completeness": {
      "expected_sweeps": len(expected_keys),
      "clean_complete_sweeps": len(expected_keys & complete_keys),
      "missing_or_rejected": [
        {"workpoint": wp, "phase": phase, "phase_name": PHASE_NAMES[phase]}
        for wp, phase in sorted(expected_keys - complete_keys)
      ],
    },
    "accepted_for_baseline_replay": overall_ok,
    "interpretation_notes": [
      "Peak-to-peak motion is an excitation check, not an FRF or SNR estimate.",
      "Motor feedback torque may be current-derived and is not an independent "
      "torque calibration.",
      "A timestamp or sequence discontinuity starts a new segment; segments "
      "must not be concatenated.",
    ],
  }


def format_audit_report(report: dict[str, Any]) -> str:
  integrity = report["integrity"]
  completeness = report["completeness"]
  mapping = report["mapping"]
  errors = mapping["errors"]
  status = "PASS" if report["accepted_for_baseline_replay"] else "FAIL"
  lines = [
    f"PACE audit: {status}",
    f"source: {report['source']['path']}",
    f"sha256: {report['source']['sha256']}",
    (
      "rows={rows}, segments={continuous_segments}, gaps={sequence_gap_events}, "
      "missing={estimated_missing_frames}, breaks={reset_or_break_events}"
    ).format(**integrity),
    (
      "fault_rows={fault_rows}, saturated_rows={saturated_rows}, "
      "nonfinite_rows={nonfinite_rows}, max_dt_ms={max_dt:.3f}"
    ).format(max_dt=1000.0 * integrity["max_device_dt_s"], **integrity),
    (
      f"complete sweeps: {completeness['clean_complete_sweeps']}/"
      f"{completeness['expected_sweeps']}"
    ),
    (
      "mapping check: {} (q max={:.3g} rad, L max={:.3g} m, "
      "phi max={:.3g} rad)"
    ).format(
      "PASS" if mapping["hard_check_passed"] else "FAIL",
      errors["solver_to_policy_qpos_max_abs_rad"],
      errors["solver_fk_length_max_abs_m"],
      errors["solver_fk_phi_max_abs_rad"],
    ),
  ]
  if completeness["missing_or_rejected"]:
    missing = ", ".join(
      f"wp{item['workpoint']}:{item['phase_name']}"
      for item in completeness["missing_or_rejected"]
    )
    lines.append(f"missing/rejected: {missing}")
  lines.extend((
    "Note: peak-to-peak values are motion checks, not FRF/SNR estimates.",
    "Note: torque feedback is not treated as an independent torque sensor.",
  ))
  return "\n".join(lines) + "\n"
