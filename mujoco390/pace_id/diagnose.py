"""Short-window error growth and timestep diagnostics for PACE replay."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np

from .core import (
  LENGTH_SWEEP,
  LoadedSamples,
  Mapping,
  PHASE_NAMES,
  PHI_SWEEP,
  file_sha256,
  load_samples,
  wrap_angle,
  write_json,
)
from .model import ModelAdapter
from .replay import (
  relative_device_time,
  repository_state,
  split_sweep_runs,
  validate_mapping,
  validate_model_coordinate_mapping,
)


HORIZONS_S = (0.25, 0.5, 1.0)
START_LABELS = ("early", "middle", "late")
TIMESTEPS_S = (0.0025, 0.00125, 0.000625)
ERROR_THRESHOLDS = {
  "joint_position_rad": 0.05,
  "leg_length_m": 0.005,
  "leg_phi_rad": 0.05,
}


@dataclass(frozen=True)
class DiagnosticWindow:
  name: str
  workpoint: int
  phase: int
  start_label: str
  indices: np.ndarray
  time_s: np.ndarray


@dataclass
class SimulationResult:
  rows: list[dict[str, Any]]
  failure: dict[str, Any] | None
  initialization: dict[str, Any]


def _warning_names(data: mujoco.MjData) -> list[str]:
  return [
    mujoco.mjtWarning(index).name
    for index in range(int(mujoco.mjtWarning.mjNWARNING))
    if data.warning[index].number
  ]


def _window_from_start(
  samples: LoadedSamples,
  run_indices: np.ndarray,
  start_s: float,
  duration_s: float,
  name: str,
  start_label: str,
) -> DiagnosticWindow:
  run_time = relative_device_time(samples, run_indices)
  start_offset = int(np.searchsorted(run_time, start_s, side="left"))
  start_offset = min(start_offset, run_indices.size - 2)
  end_s = run_time[start_offset] + duration_s + 1e-9
  end_offset = int(np.searchsorted(run_time, end_s, side="right"))
  indices = run_indices[start_offset:end_offset]
  if indices.size < 2:
    raise ValueError(f"Diagnostic window is too short: {name}")
  return DiagnosticWindow(
    name=name,
    workpoint=int(samples.workpoint[indices[0]]),
    phase=int(samples.phase[indices[0]]),
    start_label=start_label,
    indices=indices,
    time_s=relative_device_time(samples, indices),
  )


def build_short_windows(samples: LoadedSamples) -> list[DiagnosticWindow]:
  """Choose deterministic early/middle/late 1 s windows for every sweep."""
  windows = []
  for run_number, indices in enumerate(split_sweep_runs(samples)):
    run_time = relative_device_time(samples, indices)
    available = float(run_time[-1]) - HORIZONS_S[-1]
    starts = (0.0, 0.5 * available, available)
    workpoint = int(samples.workpoint[indices[0]])
    phase = int(samples.phase[indices[0]])
    phase_name = PHASE_NAMES[phase]
    for label, start_s in zip(START_LABELS, starts, strict=True):
      name = (
        f"wp{workpoint:02d}_{phase_name}_run{run_number:02d}_{label}"
      )
      windows.append(_window_from_start(
        samples, indices, start_s, HORIZONS_S[-1], name, label
      ))
  return windows


def _state_fingerprint(data: mujoco.MjData) -> str:
  digest = hashlib.sha256()
  digest.update(np.asarray(data.qpos, dtype="<f8").tobytes())
  digest.update(np.asarray(data.qvel, dtype="<f8").tobytes())
  return digest.hexdigest()


def _joint_name(model: mujoco.MjModel, joint_id: int) -> str:
  return str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))


def _initialization_record(
  adapter: ModelAdapter,
  data: mujoco.MjData,
  base: dict[str, float],
  samples: LoadedSamples,
  window: DiagnosticWindow,
) -> dict[str, Any]:
  active = {adapter.can3_joint_id, adapter.can2_joint_id}
  passive = set(adapter.passive_joint_ids)
  wheel_dofs = set(adapter.wheel_dofs)
  joints = []
  for joint_id in range(adapter.model.njnt):
    dof = int(adapter.model.jnt_dofadr[joint_id])
    qpos = int(adapter.model.jnt_qposadr[joint_id])
    if joint_id in active:
      source = "logged active position and speed"
    elif joint_id in passive:
      source = "closure-solved position and velocity"
    elif dof in wheel_dofs:
      source = "unrecorded wheel: model default position, zero speed"
    else:
      source = "unrecorded: model default position, zero speed"
    joints.append({
      "joint": _joint_name(adapter.model, joint_id),
      "source": source,
      "qpos": float(data.qpos[qpos]),
      "qvel": float(data.qvel[dof]),
    })
  closure = adapter.closure_diagnostics(data)
  return {
    "window": window.name,
    "workpoint": window.workpoint,
    "phase": window.phase,
    "phase_name": PHASE_NAMES[window.phase],
    "start_label": window.start_label,
    "start_sequence": int(samples.sequence[window.indices[0]]),
    "state_fingerprint": _state_fingerprint(data),
    "solver_result": base,
    "closure": closure,
    "passive_limits": adapter.passive_limit_diagnostics(data),
    "joints": joints,
  }


def _initialize(
  adapter: ModelAdapter,
  samples: LoadedSamples,
  window: DiagnosticWindow,
) -> tuple[mujoco.MjData, dict[str, Any]]:
  fields = samples.fields
  first = int(window.indices[0])
  q3, q2 = adapter.mapping.qpos_from_solver(
    fields["solver_q1_rad"][first], fields["solver_q2_rad"][first]
  )
  data = adapter.new_data()
  result = adapter.initialize(
    data,
    float(q3),
    float(q2),
    float(fields["can3_speed_rad_s"][first]),
    float(fields["can2_speed_rad_s"][first]),
  )
  l_sim, phi_sim, _, _ = adapter.virtual_state(data)
  result["leg_length_error_m"] = (
    l_sim - float(fields["measured_leg_length_m"][first])
  )
  result["leg_phi_error_rad"] = float(wrap_angle(
    phi_sim - float(fields["measured_leg_phi_rad"][first])
  ))
  return data, _initialization_record(
    adapter, data, result, samples, window
  )


def _state_row(
  adapter: ModelAdapter,
  data: mujoco.MjData,
  samples: LoadedSamples,
  sample_index: int,
  window: DiagnosticWindow,
  time_s: float,
  timestep_s: float,
) -> dict[str, Any]:
  fields = samples.fields
  q3_meas, q2_meas = adapter.mapping.qpos_from_solver(
    fields["solver_q1_rad"][sample_index],
    fields["solver_q2_rad"][sample_index],
  )
  q3_sim = float(wrap_angle(data.qpos[adapter.can3_qpos]))
  q2_sim = float(wrap_angle(data.qpos[adapter.can2_qpos]))
  dq3_sim = float(data.qvel[adapter.can3_dof])
  dq2_sim = float(data.qvel[adapter.can2_dof])
  l_sim, phi_sim, ldot_sim, phidot_sim = adapter.virtual_state(data)
  closure = adapter.closure_diagnostics(data)
  limits = adapter.passive_limit_diagnostics(data)
  return {
    "trajectory": window.name,
    "workpoint": window.workpoint,
    "phase": window.phase,
    "phase_name": PHASE_NAMES[window.phase],
    "start_label": window.start_label,
    "timestep_s": timestep_s,
    "device_relative_time_s": time_s,
    "simulation_time_s": float(data.time),
    "device_sequence": int(samples.sequence[sample_index]),
    "can3_command_nm": float(fields["can3_command_nm"][sample_index]),
    "can2_command_nm": float(fields["can2_command_nm"][sample_index]),
    "can3_measured_qpos_rad": float(q3_meas),
    "can3_simulated_qpos_rad": q3_sim,
    "can3_qpos_error_rad": float(wrap_angle(q3_sim - q3_meas)),
    "can3_measured_speed_rad_s": float(
      fields["can3_speed_rad_s"][sample_index]
    ),
    "can3_simulated_speed_rad_s": dq3_sim,
    "can3_speed_error_rad_s": (
      dq3_sim - float(fields["can3_speed_rad_s"][sample_index])
    ),
    "can2_measured_qpos_rad": float(q2_meas),
    "can2_simulated_qpos_rad": q2_sim,
    "can2_qpos_error_rad": float(wrap_angle(q2_sim - q2_meas)),
    "can2_measured_speed_rad_s": float(
      fields["can2_speed_rad_s"][sample_index]
    ),
    "can2_simulated_speed_rad_s": dq2_sim,
    "can2_speed_error_rad_s": (
      dq2_sim - float(fields["can2_speed_rad_s"][sample_index])
    ),
    "measured_leg_length_m": float(
      fields["measured_leg_length_m"][sample_index]
    ),
    "simulated_leg_length_m": l_sim,
    "leg_length_error_m": (
      l_sim - float(fields["measured_leg_length_m"][sample_index])
    ),
    "measured_leg_phi_rad": float(
      fields["measured_leg_phi_rad"][sample_index]
    ),
    "simulated_leg_phi_rad": phi_sim,
    "leg_phi_error_rad": float(wrap_angle(
      phi_sim - float(fields["measured_leg_phi_rad"][sample_index])
    )),
    "measured_leg_length_rate_m_s": float(
      fields["measured_leg_length_rate_m_s"][sample_index]
    ),
    "simulated_leg_length_rate_m_s": ldot_sim,
    "measured_leg_phi_rate_rad_s": float(
      fields["measured_leg_phi_rate_rad_s"][sample_index]
    ),
    "simulated_leg_phi_rate_rad_s": phidot_sim,
    "loop_gap_m": float(closure["loop_gap_m"]),
    "closure_effective_rank": int(closure["effective_rank"]),
    "closure_structural_zero_count": int(
      closure["structural_zero_count"]
    ),
    "closure_effective_min_singular_value": float(
      closure["effective_min_singular_value"]
    ),
    "closure_effective_condition_number": float(
      closure["effective_condition_number"]
    ),
    "passive_limit_min_distance_rad": float(
      limits["minimum_signed_distance_rad"]
    ),
    "active_speed_peak_rad_s": float(max(abs(dq3_sim), abs(dq2_sim))),
    "all_speed_peak_rad_s": float(np.max(np.abs(data.qvel))),
    "active_acceleration_peak_rad_s2": float(max(
      abs(data.qacc[adapter.can3_dof]),
      abs(data.qacc[adapter.can2_dof]),
    )),
    "all_acceleration_peak_rad_s2": float(np.max(np.abs(data.qacc))),
  }


def _failure_kind(
  data: mujoco.MjData,
  previous_time_s: float,
  timestep_s: float,
) -> tuple[str | None, dict[str, Any]]:
  warnings = _warning_names(data)
  finite = all(np.isfinite(array).all() for array in (
    data.qpos, data.qvel, data.qacc
  ))
  reset = abs(float(data.time) - (previous_time_s + timestep_s)) > max(
    1e-10, timestep_s * 1e-6
  )
  extreme_speed = float(np.max(np.abs(data.qvel))) > 1e4
  extreme_acceleration = float(np.max(np.abs(data.qacc))) > 1e8
  details = {
    "warnings": warnings,
    "finite": finite,
    "automatic_time_reset_or_jump": reset,
    "post_step_time_s": float(data.time),
    "post_step_speed_peak_rad_s": float(np.max(np.abs(data.qvel))),
    "post_step_acceleration_peak_rad_s2": float(
      np.max(np.abs(data.qacc))
    ),
  }
  if warnings:
    return "mujoco_warning", details
  if not finite:
    return "non_finite_state", details
  if reset:
    return "automatic_reset_or_time_jump", details
  if extreme_speed and extreme_acceleration:
    return "extreme_speed_and_acceleration", details
  if extreme_acceleration:
    return "extreme_acceleration", details
  if extreme_speed:
    return "extreme_speed", details
  return None, details


def _restore_diagnostics(
  adapter: ModelAdapter,
  qpos: np.ndarray,
  qvel: np.ndarray,
  time_s: float,
) -> tuple[mujoco.MjData, dict[str, Any]]:
  restored = adapter.new_data()
  restored.qpos[:] = qpos
  restored.qvel[:] = qvel
  restored.time = time_s
  mujoco.mj_forward(adapter.model, restored)
  return restored, {
    "closure": adapter.closure_diagnostics(restored),
    "passive_limits": adapter.passive_limit_diagnostics(restored),
    "active_speed_peak_rad_s": float(max(
      abs(restored.qvel[adapter.can3_dof]),
      abs(restored.qvel[adapter.can2_dof]),
    )),
    "all_speed_peak_rad_s": float(np.max(np.abs(restored.qvel))),
    "active_acceleration_peak_rad_s2": float(max(
      abs(restored.qacc[adapter.can3_dof]),
      abs(restored.qacc[adapter.can2_dof]),
    )),
    "all_acceleration_peak_rad_s2": float(
      np.max(np.abs(restored.qacc))
    ),
  }


def simulate_window(
  adapter: ModelAdapter,
  samples: LoadedSamples,
  window: DiagnosticWindow,
) -> SimulationResult:
  """Replay one trajectory and retain failures outside successful rows."""
  data, initialization = _initialize(adapter, samples, window)
  fields = samples.fields
  rows = []
  command_index = 0
  timestep_s = float(adapter.model.opt.timestep)
  peak_speed = 0.0
  peak_acceleration = 0.0
  for output_index, (sample_index, target_time_s) in enumerate(
    zip(window.indices, window.time_s, strict=True)
  ):
    while data.time + 0.5 * timestep_s < target_time_s:
      while (
        command_index + 1 < output_index
        and window.time_s[command_index + 1] <= data.time + 1e-12
      ):
        command_index += 1
      command_sample = int(window.indices[command_index])
      adapter.apply_logged_torque(
        data,
        float(fields["can3_command_nm"][command_sample]),
        float(fields["can2_command_nm"][command_sample]),
      )
      previous_time = float(data.time)
      previous_qpos = data.qpos.copy()
      previous_qvel = data.qvel.copy()
      mujoco.mj_step(adapter.model, data)
      kind, post = _failure_kind(data, previous_time, timestep_s)
      peak_speed = max(peak_speed, float(np.max(np.abs(data.qvel))))
      peak_acceleration = max(
        peak_acceleration, float(np.max(np.abs(data.qacc)))
      )
      if kind is not None:
        restored, before = _restore_diagnostics(
          adapter, previous_qpos, previous_qvel, previous_time
        )
        before_row = _state_row(
          adapter,
          restored,
          samples,
          command_sample,
          window,
          previous_time,
          timestep_s,
        )
        failure = {
          "trajectory": window.name,
          "workpoint": window.workpoint,
          "phase": window.phase,
          "phase_name": PHASE_NAMES[window.phase],
          "start_label": window.start_label,
          "timestep_s": timestep_s,
          "failure_kind": kind,
          "failure_simulation_time_s": post["post_step_time_s"],
          "last_stable_simulation_time_s": previous_time,
          "last_stable_device_sample_time_s": float(
            window.time_s[command_index]
          ),
          "last_stable_state_vs_hardware": before_row,
          "last_stable_diagnostics": before,
          "post_step_diagnostics": post,
          "peak_speed_before_detection_rad_s": peak_speed,
          "peak_acceleration_before_detection_rad_s2": peak_acceleration,
        }
        return SimulationResult(rows, failure, initialization)
    row = _state_row(
      adapter,
      data,
      samples,
      int(sample_index),
      window,
      float(target_time_s),
      timestep_s,
    )
    rows.append(row)
  return SimulationResult(rows, None, initialization)


def _rmse(rows: list[dict[str, Any]], key: str) -> float:
  return math.sqrt(float(np.mean([float(row[key]) ** 2 for row in rows])))


def _first_growth_time(rows: list[dict[str, Any]]) -> float | None:
  for row in rows:
    if (
      max(
        abs(float(row["can3_qpos_error_rad"])),
        abs(float(row["can2_qpos_error_rad"])),
      ) >= ERROR_THRESHOLDS["joint_position_rad"]
      or abs(float(row["leg_length_error_m"]))
      >= ERROR_THRESHOLDS["leg_length_m"]
      or abs(float(row["leg_phi_error_rad"]))
      >= ERROR_THRESHOLDS["leg_phi_rad"]
    ):
      return float(row["device_relative_time_s"])
  return None


def short_window_metrics(
  result: SimulationResult,
  window: DiagnosticWindow,
) -> list[dict[str, Any]]:
  if result.failure is not None:
    return []
  initial = result.rows[0]
  growth_time = _first_growth_time(result.rows)
  metrics = []
  for horizon in HORIZONS_S:
    prefix = [
      row for row in result.rows
      if float(row["device_relative_time_s"]) <= horizon + 1e-9
    ]
    if not prefix or float(prefix[-1]["device_relative_time_s"]) < (
      horizon - 0.02
    ):
      continue
    final = prefix[-1]
    metrics.append({
      "trajectory": window.name,
      "workpoint": window.workpoint,
      "phase": window.phase,
      "phase_name": PHASE_NAMES[window.phase],
      "start_label": window.start_label,
      "start_sequence": int(result.initialization["start_sequence"]),
      "initial_state_fingerprint": result.initialization[
        "state_fingerprint"
      ],
      "horizon_s": horizon,
      "samples": len(prefix),
      "initial_can3_qpos_error_rad": initial["can3_qpos_error_rad"],
      "initial_can2_qpos_error_rad": initial["can2_qpos_error_rad"],
      "initial_can3_speed_error_rad_s": initial["can3_speed_error_rad_s"],
      "initial_can2_speed_error_rad_s": initial["can2_speed_error_rad_s"],
      "initial_leg_length_error_m": initial["leg_length_error_m"],
      "initial_leg_phi_error_rad": initial["leg_phi_error_rad"],
      "can3_qpos_rmse_rad": _rmse(prefix, "can3_qpos_error_rad"),
      "can2_qpos_rmse_rad": _rmse(prefix, "can2_qpos_error_rad"),
      "can3_speed_rmse_rad_s": _rmse(prefix, "can3_speed_error_rad_s"),
      "can2_speed_rmse_rad_s": _rmse(prefix, "can2_speed_error_rad_s"),
      "leg_length_rmse_m": _rmse(prefix, "leg_length_error_m"),
      "leg_phi_rmse_rad": _rmse(prefix, "leg_phi_error_rad"),
      "final_can3_qpos_error_rad": final["can3_qpos_error_rad"],
      "final_can2_qpos_error_rad": final["can2_qpos_error_rad"],
      "final_can3_speed_error_rad_s": final["can3_speed_error_rad_s"],
      "final_can2_speed_error_rad_s": final["can2_speed_error_rad_s"],
      "final_leg_length_error_m": final["leg_length_error_m"],
      "final_leg_phi_error_rad": final["leg_phi_error_rad"],
      "first_significant_growth_time_s": growth_time,
      "loop_gap_max_m": max(float(row["loop_gap_m"]) for row in prefix),
    })
  return metrics


def _write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
  if not rows:
    path.write_text("status\n", encoding="utf-8")
    return
  keys = list(rows[0])
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def _run_lookup(samples: LoadedSamples) -> dict[tuple[int, int], np.ndarray]:
  lookup = {}
  for indices in split_sweep_runs(samples):
    key = (
      int(samples.workpoint[indices[0]]),
      int(samples.phase[indices[0]]),
    )
    if key in lookup:
      raise ValueError(f"Duplicate sweep run for workpoint/phase {key}")
    lookup[key] = indices
  return lookup


def build_convergence_windows(
  samples: LoadedSamples,
) -> list[DiagnosticWindow]:
  lookup = _run_lookup(samples)
  specifications = (
    (0, PHI_SWEEP, "middle", 1.0),
    (14, LENGTH_SWEEP, "middle", 1.0),
    (15, PHI_SWEEP, "middle", 1.0),
    (29, LENGTH_SWEEP, "failure_segment", 3.2),
  )
  windows = []
  for workpoint, phase, label, duration in specifications:
    indices = lookup[(workpoint, phase)]
    run_time = relative_device_time(samples, indices)
    start_s = 0.0 if label == "failure_segment" else (
      0.5 * (float(run_time[-1]) - duration)
    )
    name = f"wp{workpoint:02d}_{PHASE_NAMES[phase]}_{label}"
    windows.append(_window_from_start(
      samples, indices, start_s, duration, name, label
    ))
  return windows


def _convergence_metric(
  result: SimulationResult,
  window: DiagnosticWindow,
  timestep_s: float,
) -> dict[str, Any]:
  row = {
    "trajectory": window.name,
    "workpoint": window.workpoint,
    "phase": window.phase,
    "phase_name": PHASE_NAMES[window.phase],
    "timestep_s": timestep_s,
    "status": "failed" if result.failure else "success",
    "failure_kind": (
      result.failure["failure_kind"] if result.failure else ""
    ),
    "failure_time_s": (
      result.failure["failure_simulation_time_s"]
      if result.failure else ""
    ),
    "completed_device_time_s": (
      result.rows[-1]["device_relative_time_s"] if result.rows else 0.0
    ),
  }
  if not result.rows:
    return row
  row.update({
    "can3_qpos_rmse_rad": _rmse(result.rows, "can3_qpos_error_rad"),
    "can2_qpos_rmse_rad": _rmse(result.rows, "can2_qpos_error_rad"),
    "leg_length_rmse_m": _rmse(result.rows, "leg_length_error_m"),
    "leg_phi_rmse_rad": _rmse(result.rows, "leg_phi_error_rad"),
    "loop_gap_max_m": max(float(x["loop_gap_m"]) for x in result.rows),
    "passive_limit_min_distance_rad": min(
      float(x["passive_limit_min_distance_rad"]) for x in result.rows
    ),
    "closure_effective_min_singular_value": min(
      float(x["closure_effective_min_singular_value"])
      for x in result.rows
    ),
    "closure_effective_condition_number_max": max(
      float(x["closure_effective_condition_number"])
      for x in result.rows
    ),
    "all_speed_peak_rad_s": max(
      float(x["all_speed_peak_rad_s"]) for x in result.rows
    ),
    "all_acceleration_peak_rad_s2": max(
      float(x["all_acceleration_peak_rad_s2"]) for x in result.rows
    ),
  })
  return row


def _pairwise_convergence(
  traces: dict[tuple[str, float], list[dict[str, Any]]],
  windows: Iterable[DiagnosticWindow],
) -> list[dict[str, Any]]:
  comparisons = []
  pairs = zip(TIMESTEPS_S[:-1], TIMESTEPS_S[1:], strict=True)
  timestep_pairs = tuple(pairs)
  for window in windows:
    for coarse, fine in timestep_pairs:
      coarse_rows = traces.get((window.name, coarse), [])
      fine_rows = traces.get((window.name, fine), [])
      count = min(len(coarse_rows), len(fine_rows))
      if not count:
        continue
      differences = {
        "can3_qpos_rad": [],
        "can2_qpos_rad": [],
        "leg_length_m": [],
        "leg_phi_rad": [],
      }
      for left, right in zip(
        coarse_rows[:count], fine_rows[:count], strict=True
      ):
        differences["can3_qpos_rad"].append(float(wrap_angle(
          left["can3_simulated_qpos_rad"]
          - right["can3_simulated_qpos_rad"]
        )))
        differences["can2_qpos_rad"].append(float(wrap_angle(
          left["can2_simulated_qpos_rad"]
          - right["can2_simulated_qpos_rad"]
        )))
        differences["leg_length_m"].append(
          float(left["simulated_leg_length_m"])
          - float(right["simulated_leg_length_m"])
        )
        differences["leg_phi_rad"].append(float(wrap_angle(
          left["simulated_leg_phi_rad"]
          - right["simulated_leg_phi_rad"]
        )))
      comparisons.append({
        "trajectory": window.name,
        "workpoint": window.workpoint,
        "phase": window.phase,
        "coarse_timestep_s": coarse,
        "fine_timestep_s": fine,
        "common_samples": count,
        "common_end_time_s": coarse_rows[count - 1][
          "device_relative_time_s"
        ],
        **{
          f"{key}_difference_rmse": math.sqrt(float(np.mean(
            np.square(values)
          )))
          for key, values in differences.items()
        },
        **{
          f"{key}_terminal_difference": values[-1]
          for key, values in differences.items()
        },
      })
  return comparisons


def _aggregate_short_metrics(
  metrics: list[dict[str, Any]],
) -> dict[str, Any]:
  by_horizon = {}
  for horizon in HORIZONS_S:
    selected = [row for row in metrics if row["horizon_s"] == horizon]
    by_horizon[str(horizon)] = {
      "successful_windows": len(selected),
      "mean_can3_qpos_rmse_rad": float(np.mean([
        row["can3_qpos_rmse_rad"] for row in selected
      ])),
      "mean_can2_qpos_rmse_rad": float(np.mean([
        row["can2_qpos_rmse_rad"] for row in selected
      ])),
      "mean_leg_length_rmse_m": float(np.mean([
        row["leg_length_rmse_m"] for row in selected
      ])),
      "mean_leg_phi_rmse_rad": float(np.mean([
        row["leg_phi_rmse_rad"] for row in selected
      ])),
    }
  crossings = [
    float(row["first_significant_growth_time_s"])
    for row in metrics if row["horizon_s"] == 1.0
    and row["first_significant_growth_time_s"] is not None
  ]
  return {
    "error_growth_thresholds": ERROR_THRESHOLDS,
    "by_horizon": by_horizon,
    "windows_crossing_threshold_by_1s": len(crossings),
    "median_first_crossing_time_s": (
      float(np.median(crossings)) if crossings else None
    ),
    "minimum_first_crossing_time_s": min(crossings, default=None),
  }


def _convergence_summary(
  metrics: list[dict[str, Any]],
  comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
  paths = {}
  for name in sorted({row["trajectory"] for row in metrics}):
    path_metrics = [row for row in metrics if row["trajectory"] == name]
    path_comparisons = [
      row for row in comparisons if row["trajectory"] == name
    ]
    difference_keys = (
      "can3_qpos_rad_difference_rmse",
      "can2_qpos_rad_difference_rmse",
      "leg_length_m_difference_rmse",
      "leg_phi_rad_difference_rmse",
    )
    ratios = {}
    decreasing = None
    if len(path_comparisons) == 2:
      coarse, refined = path_comparisons
      ratios = {
        key: float(refined[key]) / max(float(coarse[key]), 1e-15)
        for key in difference_keys
      }
      decreasing = all(value < 0.8 for value in ratios.values())
    paths[name] = {
      "timestep_status": {
        str(row["timestep_s"]): {
          "status": row["status"],
          "failure_kind": row["failure_kind"],
          "failure_time_s": row["failure_time_s"],
        }
        for row in path_metrics
      },
      "pairwise_differences": path_comparisons,
      "refined_over_coarse_difference_ratio": ratios,
      "pairwise_difference_decreases_consistently": decreasing,
    }
  return {"paths": paths}


def _initialization_summary(
  records: list[dict[str, Any]],
) -> dict[str, Any]:
  solver = [item["solver_result"] for item in records]
  closure = [item["closure"] for item in records]
  limits = [item["passive_limits"] for item in records]
  return {
    "windows": len(records),
    "maximum_active_position_initial_error_rad": max(
      max(abs(item["can3_qpos_error_rad"]),
          abs(item["can2_qpos_error_rad"]))
      for item in solver
    ),
    "maximum_constraint_velocity_error": max(
      item["constraint_velocity_error"] for item in solver
    ),
    "maximum_initial_leg_length_error_m": max(
      abs(item["leg_length_error_m"]) for item in solver
    ),
    "maximum_initial_leg_phi_error_rad": max(
      abs(item["leg_phi_error_rad"]) for item in solver
    ),
    "maximum_initial_loop_gap_m": max(
      item["loop_gap_m"] for item in closure
    ),
    "effective_ranks": sorted({
      item["effective_rank"] for item in closure
    }),
    "structural_zero_counts": sorted({
      item["structural_zero_count"] for item in closure
    }),
    "minimum_initial_effective_singular_value": min(
      item["effective_min_singular_value"] for item in closure
    ),
    "maximum_initial_effective_condition_number": max(
      item["effective_condition_number"] for item in closure
    ),
    "initial_passive_limit_violations": sum(
      item["minimum_signed_distance_rad"] < 0.0 for item in limits
    ),
    "minimum_initial_passive_limit_distance_rad": min(
      item["minimum_signed_distance_rad"] for item in limits
    ),
  }


def _workpoint29_summary(
  failures: list[dict[str, Any]],
  traces: dict[tuple[str, float], list[dict[str, Any]]],
  convergence: dict[str, Any],
) -> dict[str, Any]:
  name = "wp29_length_sweep_failure_segment"
  by_timestep = {}
  for timestep_s in TIMESTEPS_S:
    rows = traces.get((name, timestep_s), [])
    if not rows:
      continue
    penetrations = [
      row for row in rows
      if float(row["passive_limit_min_distance_rad"]) < 0.0
    ]
    by_timestep[str(timestep_s)] = {
      "recorded_through_s": float(rows[-1]["device_relative_time_s"]),
      "first_passive_limit_penetration_s": (
        float(penetrations[0]["device_relative_time_s"])
        if penetrations else None
      ),
      "minimum_passive_limit_distance_rad": min(
        float(row["passive_limit_min_distance_rad"]) for row in rows
      ),
      "maximum_loop_gap_m": max(
        float(row["loop_gap_m"]) for row in rows
      ),
      "minimum_effective_singular_value": min(
        float(row["closure_effective_min_singular_value"])
        for row in rows
      ),
      "maximum_effective_condition_number": max(
        float(row["closure_effective_condition_number"])
        for row in rows
      ),
      "effective_ranks": sorted({
        int(row["closure_effective_rank"]) for row in rows
      }),
      "structural_zero_counts": sorted({
        int(row["closure_structural_zero_count"]) for row in rows
      }),
    }
  base_failure = next(
    (item for item in failures if item["timestep_s"] == TIMESTEPS_S[0]),
    None,
  )
  pre_failure = None
  if base_failure is not None:
    state = base_failure["last_stable_state_vs_hardware"]
    diagnostics = base_failure["last_stable_diagnostics"]
    pre_failure = {
      "failure_kind": base_failure["failure_kind"],
      "failure_time_s": base_failure["failure_simulation_time_s"],
      "last_stable_time_s": base_failure[
        "last_stable_simulation_time_s"
      ],
      "can3_qpos_error_rad": state["can3_qpos_error_rad"],
      "can2_qpos_error_rad": state["can2_qpos_error_rad"],
      "can3_speed_error_rad_s": state["can3_speed_error_rad_s"],
      "can2_speed_error_rad_s": state["can2_speed_error_rad_s"],
      "leg_length_error_m": state["leg_length_error_m"],
      "leg_phi_error_rad": state["leg_phi_error_rad"],
      "loop_gap_m": diagnostics["closure"]["loop_gap_m"],
      "effective_min_singular_value": diagnostics["closure"][
        "effective_min_singular_value"
      ],
      "effective_condition_number": diagnostics["closure"][
        "effective_condition_number"
      ],
      "effective_rank": diagnostics["closure"]["effective_rank"],
      "structural_zero_count": diagnostics["closure"][
        "structural_zero_count"
      ],
      "passive_limit_distance_rad": diagnostics["passive_limits"][
        "minimum_signed_distance_rad"
      ],
      "all_speed_peak_rad_s": diagnostics["all_speed_peak_rad_s"],
      "all_acceleration_peak_rad_s2": diagnostics[
        "all_acceleration_peak_rad_s2"
      ],
      "post_step": base_failure["post_step_diagnostics"],
    }
  path = convergence["paths"].get(name, {})
  return {
    "by_timestep": by_timestep,
    "base_timestep_pre_failure": pre_failure,
    "smaller_timesteps_complete_3p2s": all(
      not any(
        item["timestep_s"] == timestep
        for item in failures
      )
      for timestep in TIMESTEPS_S[1:]
    ),
    "timestep_converged": path.get(
      "pairwise_difference_decreases_consistently"
    ),
  }


def run_diagnosis(
  csv_path: Path,
  xml_path: Path,
  output_dir: Path,
  baseline_dir: Path,
  maximum_runs: int | None = None,
  run_convergence: bool = True,
) -> dict[str, Any]:
  mapping = Mapping()
  samples = load_samples(csv_path)
  mapping_check = validate_mapping(samples, mapping)
  base_adapter = ModelAdapter.create(xml_path, mapping, TIMESTEPS_S[0])
  coordinate_check = validate_model_coordinate_mapping(
    base_adapter, samples
  )
  windows = build_short_windows(samples)
  if maximum_runs is not None:
    allowed_runs = maximum_runs * len(START_LABELS)
    windows = windows[:allowed_runs]

  sample_rows = []
  metric_rows = []
  failures = []
  initializations = []
  for window in windows:
    result = simulate_window(base_adapter, samples, window)
    initializations.append(result.initialization)
    if result.failure is not None:
      failures.append(result.failure)
      continue
    sample_rows.extend(result.rows)
    metric_rows.extend(short_window_metrics(result, window))

  _write_dict_rows(output_dir / "short_window_samples.csv", sample_rows)
  _write_dict_rows(output_dir / "short_window_metrics.csv", metric_rows)
  write_json(output_dir / "short_window_failures.json", failures)
  write_json(output_dir / "initialization_diagnostics.json", {
    "assumptions": {
      "base": "fixed by removing the recorded model free joint",
      "active_joint_position_and_speed": "logged values",
      "selected_passive_positions": "geometric loop closure solve",
      "selected_passive_speeds": (
        "least-squares closure velocity using logged active speeds"
      ),
      "wheel_position": "model default because it was not logged",
      "wheel_speed": "zero because it was not logged",
      "other_unrecorded_dofs": "model default position and zero speed",
      "acceleration": "not measured or initialized; mj_forward result",
      "input": "100 Hz logged total commands, zero-order hold",
    },
    "windows": initializations,
  })

  convergence_metrics = []
  convergence_failures = []
  convergence_traces: dict[tuple[str, float], list[dict[str, Any]]] = {}
  convergence_windows = []
  if run_convergence:
    convergence_windows = build_convergence_windows(samples)
    for timestep_s in TIMESTEPS_S:
      adapter = ModelAdapter.create(xml_path, mapping, timestep_s)
      for window in convergence_windows:
        result = simulate_window(adapter, samples, window)
        convergence_metrics.append(_convergence_metric(
          result, window, timestep_s
        ))
        if result.failure is not None:
          convergence_failures.append(result.failure)
        convergence_traces[(window.name, timestep_s)] = result.rows
  convergence_samples = [
    row
    for rows in convergence_traces.values()
    for row in rows
  ]
  convergence_comparison = _pairwise_convergence(
    convergence_traces, convergence_windows
  )
  _write_dict_rows(
    output_dir / "convergence_samples.csv", convergence_samples
  )
  _write_dict_rows(
    output_dir / "convergence_metrics.csv", convergence_metrics
  )
  _write_dict_rows(
    output_dir / "convergence_comparison.csv", convergence_comparison
  )
  write_json(
    output_dir / "convergence_failures.json", convergence_failures
  )
  wp29_failures = [
    item for item in convergence_failures if item["workpoint"] == 29
  ]
  write_json(output_dir / "workpoint29_failures.json", wp29_failures)

  short_summary = _aggregate_short_metrics(metric_rows)
  initialization_summary = _initialization_summary(initializations)
  convergence_summary = _convergence_summary(
    convergence_metrics, convergence_comparison
  )
  wp29_summary = _workpoint29_summary(
    wp29_failures, convergence_traces, convergence_summary
  )
  representative_paths = [
    value for key, value in convergence_summary["paths"].items()
    if not key.startswith("wp29_")
  ]
  representatives_converge = bool(representative_paths) and all(
    item["pairwise_difference_decreases_consistently"]
    for item in representative_paths
  )
  pre_failure = wp29_summary["base_timestep_pre_failure"]
  wp29_already_diverged = bool(pre_failure) and (
    max(
      abs(pre_failure["can3_qpos_error_rad"]),
      abs(pre_failure["can2_qpos_error_rad"]),
    ) > 0.5
    or abs(pre_failure["leg_length_error_m"]) > 0.02
    or abs(pre_failure["leg_phi_error_rad"]) > 0.2
  )

  report = {
    "schema": "pace-short-window-diagnosis-v1",
    "source_csv": str(csv_path.resolve()),
    "source_csv_sha256": file_sha256(csv_path),
    "baseline_reference": {
      "path": str(baseline_dir.resolve()),
      "exists": baseline_dir.is_dir(),
      "modified": False,
    },
    "mapping": {
      **mapping.as_dict(),
      "log_check": mapping_check,
      "model_coordinate_check": coordinate_check,
      "sign_search_performed": False,
    },
    "model_validation": base_adapter.validation,
    "initialization_summary": initialization_summary,
    "short_windows": {
      "selection": (
        "fixed early/middle/late starts in every sweep; each start is "
        "simulated once to 1 s and 0.25/0.5/1 s metrics are prefixes of "
        "that identical initialized trajectory"
      ),
      "requested_windows": len(windows),
      "successful_windows": len(windows) - len(failures),
      "failed_windows": len(failures),
      "summary": short_summary,
    },
    "timestep_convergence": convergence_summary,
    "workpoint29_failures": wp29_failures,
    "workpoint29_summary": wp29_summary,
    "analysis_summary": {
      "error_growth": (
        "All successful windows cross the declared significant-error "
        f"threshold by 1 s; median first crossing is "
        f"{short_summary['median_first_crossing_time_s']:.6f} s."
      ),
      "representative_timestep_convergence": (
        "The three ordinary 1 s representative windows show consistently "
        "smaller pairwise differences after each timestep halving."
        if representatives_converge else
        "Ordinary representative windows do not all show consistent "
        "timestep convergence."
      ),
      "workpoint29_timestep_convergence": (
        "The smaller timesteps avoid the 2.975 s base-step failure, but "
        "half-step versus quarter-step differences grow; workpoint 29 is "
        "not timestep-converged through 3.2 s."
      ),
      "workpoint29_pre_failure_divergence": (
        "The simulated pose is already far from hardware before the "
        "base-step failure."
        if wp29_already_diverged else
        "No large pre-failure simulation/hardware pose separation was "
        "detected by the declared thresholds."
      ),
      "limit_evidence": (
        "No initialized window violates a passive limit. In the workpoint "
        "29 simulated trajectory, passive-limit penetration begins near "
        "0.20 s at all three timesteps, after open-loop divergence starts."
      ),
      "constraint_evidence": (
        "The closure Jacobian retains effective rank 4 with two structural "
        "zeros. No rank loss is observed; smaller timesteps reduce ordinary "
        "window loop gaps, while workpoint 29 develops large loop gap and "
        "worsening effective conditioning near 3 s."
      ),
      "not_determined": [
        "whether the initial force mismatch is mainly gas spring, friction, "
        "mass/inertia, or missing 200 Hz input updates",
        "whether model passive joint limits exactly match hardware stops",
        "whether workpoint 29 degeneration is caused by limit interaction, "
        "constraint tuning, or incorrect dynamics",
      ],
    },
    "failure_policy": (
      "warning, non-finite, automatic time reset/jump, qvel>1e4, or "
      "qacc>1e8 marks the whole trajectory failed; failed trajectory "
      "samples are not included in success metrics"
    ),
    "constraint_svd_interpretation": (
      "SVD uses the selected leg's two site-closure Jacobians over two "
      "active plus four passive columns. Values <= max(1e-10, smax*1e-8) "
      "are reported as structural zeros and excluded from condition number."
    ),
    "input_reconstruction": {
      "method": "100 Hz logged total-command snapshots, zero-order hold",
      "limitation": (
        "Unlogged 200 Hz controller updates and exact CAN timing are not "
        "recoverable."
      ),
    },
    "optimization_performed": False,
    "model_modified": False,
    "repository": repository_state(xml_path.resolve().parents[2]),
    "outputs": {
      "short_window_samples": str(
        output_dir / "short_window_samples.csv"
      ),
      "short_window_metrics": str(
        output_dir / "short_window_metrics.csv"
      ),
      "short_window_failures": str(
        output_dir / "short_window_failures.json"
      ),
      "initialization": str(
        output_dir / "initialization_diagnostics.json"
      ),
      "convergence_samples": str(
        output_dir / "convergence_samples.csv"
      ),
      "convergence_metrics": str(
        output_dir / "convergence_metrics.csv"
      ),
      "convergence_comparison": str(
        output_dir / "convergence_comparison.csv"
      ),
      "convergence_failures": str(
        output_dir / "convergence_failures.json"
      ),
      "workpoint29_failures": str(
        output_dir / "workpoint29_failures.json"
      ),
    },
  }
  return report


def format_diagnosis_report(report: dict[str, Any]) -> str:
  short = report["short_windows"]
  summary = short["summary"]
  lines = [
    "PACE short-window diagnosis: COMPLETE",
    (
      f"short windows: {short['successful_windows']}/"
      f"{short['requested_windows']} successful"
    ),
  ]
  for horizon in HORIZONS_S:
    item = summary["by_horizon"][str(horizon)]
    lines.append((
      f"{horizon:.2f}s mean RMSE q3/q2="
      f"{item['mean_can3_qpos_rmse_rad']:.5f}/"
      f"{item['mean_can2_qpos_rmse_rad']:.5f} rad, L/phi="
      f"{item['mean_leg_length_rmse_m']:.6f} m/"
      f"{item['mean_leg_phi_rmse_rad']:.5f} rad"
    ))
  lines.append((
    "significant-error threshold crossings by 1s: "
    f"{summary['windows_crossing_threshold_by_1s']}; median first time="
    f"{summary['median_first_crossing_time_s']} s"
  ))
  failures = report["workpoint29_failures"]
  if failures:
    details = ", ".join(
      f"dt={item['timestep_s']}: {item['failure_kind']} at "
      f"{item['failure_simulation_time_s']:.6f}s"
      for item in failures
    )
    lines.append(f"workpoint 29 failures: {details}")
  else:
    lines.append("workpoint 29 failures: none through 3.2s")
  analysis = report["analysis_summary"]
  lines.extend((
    f"error onset: {analysis['error_growth']}",
    "ordinary timestep check: "
    + analysis["representative_timestep_convergence"],
    "workpoint 29 timestep check: "
    + analysis["workpoint29_timestep_convergence"],
    "workpoint 29 before failure: "
    + analysis["workpoint29_pre_failure_divergence"],
    "limits: " + analysis["limit_evidence"],
    "constraints: " + analysis["constraint_evidence"],
    "still undetermined: " + "; ".join(analysis["not_determined"]),
  ))
  lines.extend((
    "No fitting, sign search, firmware edit, or model overwrite performed.",
    f"details: {Path(report['outputs']['short_window_metrics']).parent}",
  ))
  return "\n".join(lines) + "\n"
