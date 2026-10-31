"""Baseline replay of logged two-motor PACE commands in MuJoCo."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import math
from pathlib import Path
import subprocess
from typing import Any, Iterator

import mujoco
import numpy as np

from .core import (
  LENGTH_SWEEP,
  LoadedSamples,
  Mapping,
  PHASE_NAMES,
  PHI_SWEEP,
  forward_kinematics,
  load_samples,
  uint32_delta,
  wrap_angle,
)
from .model import ModelAdapter


RESULT_COLUMNS = (
  "mode",
  "trajectory",
  "workpoint",
  "phase",
  "phase_name",
  "device_time_s",
  "simulation_time_s",
  "can3_command_nm",
  "can2_command_nm",
  "can3_measured_qpos_rad",
  "can3_simulated_qpos_rad",
  "can3_measured_speed_rad_s",
  "can3_simulated_speed_rad_s",
  "can2_measured_qpos_rad",
  "can2_simulated_qpos_rad",
  "can2_measured_speed_rad_s",
  "can2_simulated_speed_rad_s",
  "measured_leg_length_m",
  "simulated_leg_length_m",
  "measured_leg_phi_rad",
  "simulated_leg_phi_rad",
  "measured_leg_length_rate_m_s",
  "simulated_leg_length_rate_m_s",
  "measured_leg_phi_rate_rad_s",
  "simulated_leg_phi_rate_rad_s",
  "can3_qpos_error_rad",
  "can2_qpos_error_rad",
  "leg_length_error_m",
  "leg_phi_error_rad",
  "loop_gap_m",
)


class ReplayNumericalError(RuntimeError):
  """Raised when MuJoCo reports or produces an invalid dynamic state."""


def _check_simulation_state(
  data: mujoco.MjData,
  trajectory: Trajectory,
  command_can3: float,
  command_can2: float,
) -> None:
  warnings = [
    mujoco.mjtWarning(index).name
    for index in range(int(mujoco.mjtWarning.mjNWARNING))
    if data.warning[index].number
  ]
  arrays = (data.qpos, data.qvel, data.qacc)
  finite = all(np.isfinite(array).all() for array in arrays)
  extreme = (
    float(np.max(np.abs(data.qvel))) > 1.0e4
    or float(np.max(np.abs(data.qacc))) > 1.0e8
  )
  if warnings or not finite or extreme:
    raise ReplayNumericalError(
      f"{trajectory.name} failed at simulation t={data.time:.6f}s; "
      f"warnings={warnings}, finite={finite}, extreme={extreme}, "
      f"CAN3/CAN2={command_can3:.6f}/{command_can2:.6f} Nm"
    )


@dataclass(frozen=True)
class Trajectory:
  mode: str
  name: str
  indices: np.ndarray
  time_s: np.ndarray


@dataclass
class ErrorAccumulator:
  count: int = 0
  sums: dict[str, float] = field(default_factory=dict)
  loop_gap_max_m: float = 0.0

  def add(self, errors: dict[str, float], loop_gap_m: float) -> None:
    self.count += 1
    for name, value in errors.items():
      self.sums[name] = self.sums.get(name, 0.0) + value * value
    self.loop_gap_max_m = max(self.loop_gap_max_m, loop_gap_m)

  def metrics(self) -> dict[str, float | int]:
    return {
      "samples": self.count,
      **{
        f"{name}_rmse": math.sqrt(total / max(self.count, 1))
        for name, total in self.sums.items()
      },
      "loop_gap_max_m": self.loop_gap_max_m,
    }


def split_sweep_runs(
  samples: LoadedSamples,
  maximum_dt_s: float = 0.05,
) -> list[np.ndarray]:
  runs: list[np.ndarray] = []
  start = 0
  for index in range(1, len(samples)):
    sequence_delta = uint32_delta(
      int(samples.sequence[index]), int(samples.sequence[index - 1])
    )
    dt_s = uint32_delta(
      int(samples.device_time_us[index]),
      int(samples.device_time_us[index - 1]),
    ) * 1.0e-6
    same_sweep = (
      samples.phase[index] == samples.phase[index - 1]
      and samples.workpoint[index] == samples.workpoint[index - 1]
    )
    if sequence_delta != 1 or not 0.0 < dt_s <= maximum_dt_s or not same_sweep:
      runs.append(np.arange(start, index, dtype=np.int64))
      start = index
  runs.append(np.arange(start, len(samples), dtype=np.int64))
  return [indices for indices in runs if indices.size]


def relative_device_time(samples: LoadedSamples, indices: np.ndarray) -> np.ndarray:
  result = np.zeros(indices.size, dtype=np.float64)
  for output_index in range(1, indices.size):
    current = int(samples.device_time_us[indices[output_index]])
    previous = int(samples.device_time_us[indices[output_index - 1]])
    result[output_index] = result[output_index - 1] + uint32_delta(
      current, previous
    ) * 1.0e-6
  return result


def build_trajectories(
  samples: LoadedSamples,
  window_s: float,
  window_stride_s: float,
  minimum_window_s: float = 2.0,
) -> list[Trajectory]:
  trajectories: list[Trajectory] = []
  for run_number, indices in enumerate(split_sweep_runs(samples)):
    time_s = relative_device_time(samples, indices)
    workpoint = int(samples.workpoint[indices[0]])
    phase = int(samples.phase[indices[0]])
    base_name = f"wp{workpoint:02d}_{PHASE_NAMES[phase]}_run{run_number:02d}"
    trajectories.append(Trajectory("full", base_name, indices, time_s))
    start_s = 0.0
    window_number = 0
    while start_s + minimum_window_s <= time_s[-1] + 1e-9:
      mask = (time_s >= start_s) & (time_s <= start_s + window_s + 1e-9)
      window_indices = indices[mask]
      if window_indices.size >= 2:
        window_time = relative_device_time(samples, window_indices)
        if window_time[-1] >= minimum_window_s:
          name = f"{base_name}_window{window_number:02d}"
          trajectories.append(
            Trajectory("window", name, window_indices, window_time)
          )
      start_s += window_stride_s
      window_number += 1
  return trajectories


def validate_mapping(samples: LoadedSamples, mapping: Mapping) -> dict[str, float]:
  derived_can3, derived_can2 = mapping.qpos_from_solver(
    samples.fields["solver_q1_rad"], samples.fields["solver_q2_rad"]
  )
  can3_error = wrap_angle(
    derived_can3 - samples.fields["can3_policy_angle_rad"]
  )
  can2_error = wrap_angle(
    derived_can2 - samples.fields["can2_policy_angle_rad"]
  )
  result = {
    "can3_solver_to_policy_max_abs_rad": float(np.max(np.abs(can3_error))),
    "can2_solver_to_policy_max_abs_rad": float(np.max(np.abs(can2_error))),
  }
  if max(result.values()) > 0.002:
    raise ValueError(f"Solver-to-model mapping check failed: {result}")
  return result


def validate_model_coordinate_mapping(
  adapter: ModelAdapter,
  samples: LoadedSamples,
  step_rad: float = 1e-5,
) -> dict[str, object]:
  first = 0
  mapping = adapter.mapping
  model_q = np.asarray(mapping.qpos_from_solver(
    samples.fields["solver_q1_rad"][first],
    samples.fields["solver_q2_rad"][first],
  ))
  model_columns = []
  firmware_columns = []
  for column in range(2):
    plus = model_q.copy()
    minus = model_q.copy()
    plus[column] += step_rad
    minus[column] -= step_rad
    model_values = []
    for pose in (plus, minus):
      data = adapter.new_data()
      adapter.initialize(data, float(pose[0]), float(pose[1]), 0.0, 0.0)
      model_values.append(np.asarray(adapter.virtual_state(data)[:2]))
    model_difference = model_values[0] - model_values[1]
    model_difference[1] = wrap_angle(model_difference[1])
    model_columns.append(model_difference / (2.0 * step_rad))

    solver_values = []
    for pose in (plus, minus):
      solver_q1 = (
        pose[0] - mapping.can3_qpos_offset
      ) / mapping.can3_qpos_scale
      solver_q2 = (
        pose[1] - mapping.can2_qpos_offset
      ) / mapping.can2_qpos_scale
      solver_values.append(np.asarray(forward_kinematics(solver_q1, solver_q2)))
    firmware_difference = solver_values[0] - solver_values[1]
    firmware_difference[1] = wrap_angle(firmware_difference[1])
    firmware_columns.append(firmware_difference / (2.0 * step_rad))
  model_jacobian = np.asarray(model_columns).T
  firmware_jacobian = np.asarray(firmware_columns).T
  maximum_error = float(np.max(np.abs(model_jacobian - firmware_jacobian)))
  passed = maximum_error <= 0.02
  result = {
    "model_d_L_phi_d_CAN3_CAN2": model_jacobian.tolist(),
    "firmware_d_L_phi_d_CAN3_CAN2": firmware_jacobian.tolist(),
    "maximum_abs_difference": maximum_error,
    "passed": passed,
    "interpretation": (
      "Matching Jacobian columns validate motor assignment, coordinate "
      "direction, and generalized-torque sign convention locally."
    ),
  }
  if not passed:
    raise ValueError(f"Model/firmware coordinate Jacobian mismatch: {result}")
  return result


def _record_state(
  adapter: ModelAdapter,
  data: mujoco.MjData,
  samples: LoadedSamples,
  sample_index: int,
  trajectory: Trajectory,
  relative_time_s: float,
) -> tuple[list[object], dict[str, float], dict[str, float]]:
  fields = samples.fields
  measured_q3, measured_q2 = adapter.mapping.qpos_from_solver(
    fields["solver_q1_rad"][sample_index],
    fields["solver_q2_rad"][sample_index],
  )
  simulated_q3 = float(wrap_angle(data.qpos[adapter.can3_qpos]))
  simulated_q2 = float(wrap_angle(data.qpos[adapter.can2_qpos]))
  simulated_dq3 = float(data.qvel[adapter.can3_dof])
  simulated_dq2 = float(data.qvel[adapter.can2_dof])
  sim_l, sim_phi, sim_ldot, sim_phidot = adapter.virtual_state(data)
  measured_l = float(fields["measured_leg_length_m"][sample_index])
  measured_phi = float(fields["measured_leg_phi_rad"][sample_index])
  measured_ldot = float(fields["measured_leg_length_rate_m_s"][sample_index])
  measured_phidot = float(fields["measured_leg_phi_rate_rad_s"][sample_index])
  measured_dq3 = float(fields["can3_speed_rad_s"][sample_index])
  measured_dq2 = float(fields["can2_speed_rad_s"][sample_index])
  errors = {
    "can3_qpos_rad": float(wrap_angle(simulated_q3 - measured_q3)),
    "can2_qpos_rad": float(wrap_angle(simulated_q2 - measured_q2)),
    "can3_speed_rad_s": simulated_dq3 - measured_dq3,
    "can2_speed_rad_s": simulated_dq2 - measured_dq2,
    "leg_length_m": sim_l - measured_l,
    "leg_phi_rad": float(wrap_angle(sim_phi - measured_phi)),
    "leg_length_rate_m_s": sim_ldot - measured_ldot,
    "leg_phi_rate_rad_s": sim_phidot - measured_phidot,
  }
  loop_gap = adapter.loop_gap_m(data)
  row = [
    trajectory.mode,
    trajectory.name,
    int(samples.workpoint[sample_index]),
    int(samples.phase[sample_index]),
    PHASE_NAMES[int(samples.phase[sample_index])],
    relative_time_s,
    float(data.time),
    float(fields["can3_command_nm"][sample_index]),
    float(fields["can2_command_nm"][sample_index]),
    float(measured_q3),
    simulated_q3,
    measured_dq3,
    simulated_dq3,
    float(measured_q2),
    simulated_q2,
    measured_dq2,
    simulated_dq2,
    measured_l,
    sim_l,
    measured_phi,
    sim_phi,
    measured_ldot,
    sim_ldot,
    measured_phidot,
    sim_phidot,
    errors["can3_qpos_rad"],
    errors["can2_qpos_rad"],
    errors["leg_length_m"],
    errors["leg_phi_rad"],
    loop_gap,
  ]
  trace = {
    "time_s": relative_time_s,
    "measured_q3": float(measured_q3),
    "simulated_q3": simulated_q3,
    "measured_q2": float(measured_q2),
    "simulated_q2": simulated_q2,
    "measured_l": measured_l,
    "simulated_l": sim_l,
    "measured_phi": measured_phi,
    "simulated_phi": sim_phi,
  }
  return row, errors, {**trace, "loop_gap_m": loop_gap}


def replay_trajectory(
  adapter: ModelAdapter,
  samples: LoadedSamples,
  trajectory: Trajectory,
  writer: csv.writer,
  keep_trace: bool,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
  indices = trajectory.indices
  fields = samples.fields
  first = int(indices[0])
  q3, q2 = adapter.mapping.qpos_from_solver(
    fields["solver_q1_rad"][first], fields["solver_q2_rad"][first]
  )
  data = adapter.new_data()
  initialization = adapter.initialize(
    data,
    float(q3),
    float(q2),
    float(fields["can3_speed_rad_s"][first]),
    float(fields["can2_speed_rad_s"][first]),
  )
  initial_l, initial_phi, _, _ = adapter.virtual_state(data)
  initialization["leg_length_error_m"] = (
    initial_l - float(fields["measured_leg_length_m"][first])
  )
  initialization["leg_phi_error_rad"] = float(wrap_angle(
    initial_phi - float(fields["measured_leg_phi_rad"][first])
  ))
  if (
    abs(initialization["leg_length_error_m"]) > 0.02
    or abs(initialization["leg_phi_error_rad"]) > 0.2
  ):
    raise RuntimeError(
      "Mapped model pose disagrees with logged virtual leg at initialization: "
      f"{initialization}"
    )
  accumulator = ErrorAccumulator()
  trace: list[dict[str, float]] = []
  command_index = 0
  for output_index, (sample_index, target_time_s) in enumerate(
    zip(indices, trajectory.time_s, strict=True)
  ):
    while data.time + 0.5 * adapter.model.opt.timestep < target_time_s:
      while (
        command_index + 1 < output_index
        and trajectory.time_s[command_index + 1] <= data.time
      ):
        command_index += 1
      command_sample = int(indices[command_index])
      adapter.apply_logged_torque(
        data,
        float(fields["can3_command_nm"][command_sample]),
        float(fields["can2_command_nm"][command_sample]),
      )
      mujoco.mj_step(adapter.model, data)
      _check_simulation_state(
        data,
        trajectory,
        float(fields["can3_command_nm"][command_sample]),
        float(fields["can2_command_nm"][command_sample]),
      )
    row, errors, trace_row = _record_state(
      adapter,
      data,
      samples,
      int(sample_index),
      trajectory,
      float(target_time_s),
    )
    writer.writerow(row)
    accumulator.add(errors, trace_row["loop_gap_m"])
    if keep_trace:
      trace.append(trace_row)
  return {
    "mode": trajectory.mode,
    "trajectory": trajectory.name,
    "workpoint": int(samples.workpoint[first]),
    "phase": int(samples.phase[first]),
    "phase_name": PHASE_NAMES[int(samples.phase[first])],
    "duration_s": float(trajectory.time_s[-1]),
    "initialization": initialization,
    **accumulator.metrics(),
  }, trace


def _write_metrics(path: Path, metrics: list[dict[str, Any]]) -> None:
  if not metrics:
    path.write_text("status\n", encoding="utf-8")
    return
  scalar_keys = [
    key for key in metrics[0]
    if key != "initialization"
  ]
  initialization_keys = sorted(metrics[0]["initialization"])
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(scalar_keys + [f"init_{key}" for key in initialization_keys])
    for metric in metrics:
      writer.writerow(
        [metric[key] for key in scalar_keys]
        + [metric["initialization"][key] for key in initialization_keys]
      )


def _plot_traces(
  output_dir: Path,
  traces: dict[str, list[dict[str, float]]],
) -> list[str]:
  if not traces:
    return []
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  plot_dir = output_dir / "plots"
  plot_dir.mkdir()
  outputs = []
  panels = (
    ("measured_q3", "simulated_q3", "CAN3 q [rad]"),
    ("measured_q2", "simulated_q2", "CAN2 q [rad]"),
    ("measured_l", "simulated_l", "L [m]"),
    ("measured_phi", "simulated_phi", "phi [rad]"),
  )
  for name, trace in traces.items():
    time_s = np.asarray([row["time_s"] for row in trace])
    figure, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for axis, (measured, simulated, label) in zip(axes, panels, strict=True):
      axis.plot(time_s, [row[measured] for row in trace], label="hardware")
      axis.plot(time_s, [row[simulated] for row in trace], label="MuJoCo")
      axis.set_ylabel(label)
      axis.grid(True, alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("device-relative time [s]")
    figure.suptitle(name)
    figure.tight_layout()
    output = plot_dir / f"{name}.png"
    figure.savefig(output, dpi=140)
    plt.close(figure)
    outputs.append(str(output))
  return outputs


def repository_state(path: Path) -> dict[str, object]:
  try:
    commit = subprocess.run(
      ("git", "rev-parse", "HEAD"),
      cwd=path,
      check=True,
      capture_output=True,
      text=True,
    ).stdout.strip()
    status = subprocess.run(
      ("git", "status", "--short"),
      cwd=path,
      check=True,
      capture_output=True,
      text=True,
    ).stdout.splitlines()
  except (OSError, subprocess.CalledProcessError):
    return {"commit": None, "dirty_paths": None}
  return {"commit": commit, "dirty_paths": status}


def run_replay(
  csv_path: Path,
  xml_path: Path,
  output_dir: Path,
  window_s: float = 5.0,
  window_stride_s: float = 5.0,
  plot_workpoints: set[int] | None = None,
) -> dict[str, Any]:
  mapping = Mapping()
  samples = load_samples(csv_path)
  mapping_check = validate_mapping(samples, mapping)
  adapter = ModelAdapter.create(xml_path, mapping)
  model_coordinate_check = validate_model_coordinate_mapping(adapter, samples)
  trajectories = build_trajectories(samples, window_s, window_stride_s)
  if not trajectories:
    raise ValueError("No replay trajectories were produced")
  plot_workpoints = plot_workpoints or set()
  metrics: list[dict[str, Any]] = []
  failures: list[dict[str, Any]] = []
  traces: dict[str, list[dict[str, float]]] = {}
  samples_path = output_dir / "replay_samples.csv"
  with samples_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(RESULT_COLUMNS)
    for trajectory in trajectories:
      first = int(trajectory.indices[0])
      workpoint = int(samples.workpoint[first])
      keep_trace = trajectory.mode == "full" and workpoint in plot_workpoints
      try:
        metric, trace = replay_trajectory(
          adapter, samples, trajectory, writer, keep_trace
        )
      except ReplayNumericalError as error:
        failures.append({
          "mode": trajectory.mode,
          "trajectory": trajectory.name,
          "workpoint": workpoint,
          "phase": int(samples.phase[first]),
          "phase_name": PHASE_NAMES[int(samples.phase[first])],
          "error": str(error),
        })
        continue
      metrics.append(metric)
      if trace:
        traces[trajectory.name] = trace
  _write_metrics(output_dir / "replay_metrics.csv", metrics)
  failure_path = output_dir / "replay_failures.csv"
  with failure_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(
      stream,
      fieldnames=(
        "mode", "trajectory", "workpoint", "phase", "phase_name", "error"
      ),
    )
    writer.writeheader()
    writer.writerows(failures)
  plots = _plot_traces(output_dir, traces)
  full_metrics = [item for item in metrics if item["mode"] == "full"]
  window_metrics = [item for item in metrics if item["mode"] == "window"]

  def aggregate(items: list[dict[str, Any]]) -> dict[str, float | int]:
    keys = (
      "can3_qpos_rad_rmse",
      "can2_qpos_rad_rmse",
      "can3_speed_rad_s_rmse",
      "can2_speed_rad_s_rmse",
      "leg_length_m_rmse",
      "leg_phi_rad_rmse",
    )
    if not items:
      return {"trajectories": 0}
    return {
      "trajectories": len(items),
      **{
        f"mean_{key}": float(np.mean([item[key] for item in items]))
        for key in keys
      },
      "maximum_loop_gap_m": max(item["loop_gap_max_m"] for item in items),
    }

  baseline_valid = not failures
  return {
    "schema": "pace-replay-v1",
    "source_csv": str(csv_path.resolve()),
    "mapping": {
      **mapping.as_dict(),
      "log_check": mapping_check,
      "model_coordinate_check": model_coordinate_check,
    },
    "model_validation": adapter.validation,
    "full_phase_baseline": aggregate(full_metrics),
    "short_window_baseline": aggregate(window_metrics),
    "numerical_failures": failures,
    "baseline_valid": baseline_valid,
    "outputs": {
      "samples_csv": str(samples_path),
      "metrics_csv": str(output_dir / "replay_metrics.csv"),
      "failures_csv": str(failure_path),
      "plots": plots,
    },
    "input_reconstruction": {
      "method": "zero-order hold of 100 Hz logged total motor commands",
      "limitation": (
        "The firmware control loop is faster than telemetry. Unlogged command "
        "updates and exact CAN scheduling are not reconstructed."
      ),
    },
    "torque_feedback_interpretation": (
      "Logged feedback may be current-derived; it was not used as an "
      "independent torque calibration."
    ),
    "optimization_performed": False,
    "repository": repository_state(xml_path.resolve().parents[2]),
  }


def format_replay_report(report: dict[str, Any]) -> str:
  full = report["full_phase_baseline"]
  windows = report["short_window_baseline"]
  model = report["model_validation"]
  valid = report["baseline_valid"]
  failure_count = len(report["numerical_failures"])
  lines = [
    "PACE MuJoCo baseline replay: "
    + ("VALID" if valid else "INVALID: NUMERICAL FAILURES"),
    f"source: {report['source_csv']}",
    f"model: {model['source_xml']}",
    (
      "model boundary: fixed base, free wheels, robot contacts disabled, "
      "policy actuators disabled"
    ),
    (
      f"gas spring: {model['gas_spring']['force_n']:.1f} N constant tendon "
      "force; firmware 370.1*L not added in simulation"
    ),
  ]
  if full["trajectories"]:
    lines.append((
      "full phases: {trajectories} trajectories, mean q RMSE CAN3/CAN2="
      "{q3:.5f}/{q2:.5f} rad, mean L/phi RMSE={length:.6f} m/"
      "{phi:.5f} rad"
    ).format(
      trajectories=full["trajectories"],
      q3=full["mean_can3_qpos_rad_rmse"],
      q2=full["mean_can2_qpos_rad_rmse"],
      length=full["mean_leg_length_m_rmse"],
      phi=full["mean_leg_phi_rad_rmse"],
    ))
  if windows["trajectories"]:
    lines.append((
      "short windows: {trajectories} trajectories, mean q RMSE CAN3/CAN2="
      "{q3:.5f}/{q2:.5f} rad, mean L/phi RMSE={length:.6f} m/"
      "{phi:.5f} rad"
    ).format(
      trajectories=windows["trajectories"],
      q3=windows["mean_can3_qpos_rad_rmse"],
      q2=windows["mean_can2_qpos_rad_rmse"],
      length=windows["mean_leg_length_m_rmse"],
      phi=windows["mean_leg_phi_rad_rmse"],
    ))
  lines.extend((
    f"numerical failures: {failure_count}",
    "Input assumption: 100 Hz total-command snapshots use zero-order hold.",
    "Torque feedback was not treated as an independent torque sensor.",
    "No parameter fitting or model overwrite was performed.",
  ))
  return "\n".join(lines) + "\n"
