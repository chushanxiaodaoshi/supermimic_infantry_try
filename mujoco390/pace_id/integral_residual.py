"""Closed-chain impulse balance over moving length-sweep windows; no fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .core import Mapping, file_sha256, load_samples, wrap_angle, write_json
from .initial_balance import tangent_basis
from .model import ModelAdapter
from .replay import relative_device_time, repository_state, split_sweep_runs


CONFIG = {
  "samples_per_window": 21, "stride_samples": 10,
  "minimum_length_displacement_m": 0.002,
  "minimum_each_joint_displacement_rad": 0.004,
  "minimum_joint_net_to_travel_ratio": 0.80,
  "maximum_pair_mean_q_distance_rad": 0.03,
  "maximum_pair_mean_length_distance_m": 0.003,
  "maximum_pair_mean_phi_distance_rad": 0.03,
  "minimum_passive_limit_clearance_rad": 0.03,
  "geometry_derivative_time_s": 1e-6,
}


def select(samples):
  q = np.array(Mapping().qpos_from_solver(
    samples.fields["solver_q1_rad"], samples.fields["solver_q2_rad"]
  )).T
  selected, coverage = [], []
  for ids in split_sweep_runs(samples):
    if samples.phase[ids[0]] != 7:
      continue
    time = relative_device_time(samples, ids)
    candidates = []
    for offset in range(0, len(ids)-20, CONFIG["stride_samples"]):
      indices = ids[offset:offset+21]
      qw = np.unwrap(q[indices], axis=0)
      delta = qw[-1] - qw[0]
      travel = np.sum(np.abs(np.diff(qw, axis=0)), axis=0)
      length = samples.fields["measured_leg_length_m"][indices]
      phi = np.unwrap(samples.fields["measured_leg_phi_rad"][indices])
      if (abs(length[-1]-length[0]) < 0.002 or delta[0]*delta[1] >= 0
          or np.min(np.abs(delta)) < 0.004
          or np.min(np.abs(delta)/np.maximum(travel, 1e-12)) < 0.8):
        continue
      candidates.append({
        "offset": offset, "direction": int(np.sign(delta[0])),
        "mean_q": np.mean(qw, axis=0), "mean_L": float(np.mean(length)),
        "mean_phi": float(np.mean(phi)), "delta": delta,
      })
    options = []
    for a in candidates:
      if a["direction"] < 0:
        continue
      for b in candidates:
        if b["direction"] > 0 or abs(a["offset"]-b["offset"]) <= 20:
          continue
        errors = np.array([
          np.max(np.abs(wrap_angle(a["mean_q"]-b["mean_q"]))),
          abs(a["mean_L"]-b["mean_L"]),
          abs(float(wrap_angle(a["mean_phi"]-b["mean_phi"]))),
        ])
        tolerance = np.array([0.03, 0.003, 0.03])
        if np.all(errors <= tolerance):
          options.append((float(np.linalg.norm(errors/tolerance)), a, b))
    wp = int(samples.workpoint[ids[0]])
    coverage.append({"workpoint": wp, "candidates": len(candidates),
                     "eligible_pairs": len(options)})
    if not options:
      continue
    score, a, b = min(options, key=lambda x: (x[0], x[1]["offset"],
                                           x[2]["offset"]))
    for label, candidate in (("positive", a), ("negative", b)):
      offset = candidate["offset"]
      ix = ids[offset:offset+21]
      selected.append({
        "workpoint": wp, "direction": label, "pair_score": score,
        "start_sequence": int(samples.sequence[ix[0]]),
        "time_s": (time[offset:offset+21]-time[offset]).tolist(),
        "q": q[ix].tolist(),
        "v": np.column_stack([
          samples.fields[f"can{can}_speed_rad_s"][ix] for can in (3, 2)
        ]).tolist(),
        "command": np.column_stack([
          samples.fields[f"can{can}_command_nm"][ix] for can in (3, 2)
        ]).tolist(),
        "L": samples.fields["measured_leg_length_m"][ix].tolist(),
        "phi": samples.fields["measured_leg_phi_rad"][ix].tolist(),
      })
  return {"windows": selected, "coverage": coverage, "config": CONFIG}


def mass_matrix(model, data):
  mass = np.zeros((model.nv, model.nv))
  mujoco.mj_fullM(model, mass, data.qM)
  return mass


def terms(adapter, data, probe, epsilon=1e-6):
  """Use derivatives of model geometry, never measured speed derivatives."""
  model = adapter.model
  basis, jac, free = tangent_basis(adapter, data)
  u = data.qvel[free].copy()
  mass = mass_matrix(model, data)
  reduced = basis.T @ mass @ basis
  values = []
  for sign in (1, -1):
    probe.qpos[:] = data.qpos
    probe.qvel[:] = 0
    mujoco.mj_integratePos(model, probe.qpos, data.qvel, sign*epsilon)
    b, _, _ = tangent_basis(adapter, probe)
    values.append((b, b.T @ mass_matrix(model, probe) @ b))
  bdot = (values[0][0]-values[1][0])/(2*epsilon)
  mdot = (values[0][1]-values[1][1])/(2*epsilon)
  curvature_force = basis.T @ mass @ (bdot @ u)
  mass_rate_force = mdot @ u
  return {
    "momentum": reduced @ u,
    "gas": basis.T @ data.qfrc_actuator,
    "passive": basis.T @ data.qfrc_passive,
    "negative_bias": -basis.T @ data.qfrc_bias,
    "curvature_force": curvature_force,
    "mass_rate_force": mass_rate_force,
    "reduced_mass": reduced,
    "friction_capacity": np.abs(basis).T @ model.dof_frictionloss,
    "gap": adapter.loop_gap_m(data),
    "velocity_closure_error": float(np.max(np.abs(jac @ data.qvel))),
    "limit_clearance": adapter.passive_limit_diagnostics(data)[
      "minimum_signed_distance_rad"
    ],
  }


def set_pose(adapter, data, q, v, first=False):
  if first:
    adapter.initialize(data, *q, *v)
  else:
    active = [adapter.can3_qpos, adapter.can2_qpos]
    data.qpos[active] += wrap_angle(np.array(q)-data.qpos[active])
    adapter._solve_passive_position(data)
    basis, _, _ = tangent_basis(adapter, data)
    data.qvel[:] = basis @ np.array([*v, 0.0])
    mujoco.mj_forward(adapter.model, data)


def integrate(time, command, states):
  time, command = np.asarray(time), np.asarray(command)
  duration = time[-1] - time[0]
  mean = lambda key: np.trapezoid(
    [s[key] for s in states], time, axis=0
  )/duration
  motor = np.r_[np.sum(command[:-1]*np.diff(time)[:, None], axis=0)/duration, 0]
  quasi = motor + mean("gas") + mean("passive") + mean("negative_bias")
  momentum_rate = (states[-1]["momentum"]-states[0]["momentum"])/duration
  inertia = momentum_rate + mean("curvature_force") - mean("mass_rate_force")
  residual = quasi - inertia
  return {
    "duration_s": float(duration),
    "mean_motor_Nm": motor.tolist(), "mean_gas_Nm": mean("gas").tolist(),
    "mean_passive_Nm": mean("passive").tolist(),
    "mean_negative_bias_Nm": mean("negative_bias").tolist(),
    "mean_quasistatic_residual_Nm": quasi.tolist(),
    "mean_inertial_torque_Nm": inertia.tolist(),
    "mean_residual_Nm": residual.tolist(),
    "momentum_change_over_duration_Nm": momentum_rate.tolist(),
    "mean_curvature_force_Nm": mean("curvature_force").tolist(),
    "mean_mass_rate_force_Nm": mean("mass_rate_force").tolist(),
    "mean_friction_capacity_Nm": mean("friction_capacity").tolist(),
  }


def evaluate(adapter, window):
  data, probe = adapter.new_data(), adapter.new_data()
  states = []
  for index, (q, v) in enumerate(zip(window["q"], window["v"], strict=True)):
    set_pose(adapter, data, q, v, first=index == 0)
    state = terms(adapter, data, probe)
    if np.any(data.warning.number) or state["limit_clearance"] < 0.03:
      return {"accepted": False, "reason": "warning_or_limit"}
    if not all(np.isfinite(value).all() for value in state.values()):
      return {"accepted": False, "reason": "nonfinite"}
    states.append(state)
  result = integrate(window["time_s"], window["command"], states)
  # Same endpoints, half-rate geometry quadrature; retain full-rate ZOH motor.
  coarse = integrate(window["time_s"][::2], window["command"][::2], states[::2])
  coarse_motor_delta = (np.array(result["mean_motor_Nm"])
                        - coarse["mean_motor_Nm"])
  q = np.unwrap(window["q"], axis=0)
  measured_delta = q[-1]-q[0]
  velocity_integral = np.trapezoid(window["v"], window["time_s"], axis=0)
  return {
    "accepted": True, **result,
    "geometry_half_rate_difference_Nm": (
      np.array(coarse["mean_residual_Nm"]) + coarse_motor_delta
      - result["mean_residual_Nm"]
    ).tolist(),
    "joint_displacement_rad": measured_delta.tolist(),
    "encoder_velocity_integral_disagreement_rad": (
      measured_delta-velocity_integral
    ).tolist(),
    "closure_gap_max_m": max(s["gap"] for s in states),
    "velocity_closure_error_max": max(s["velocity_closure_error"] for s in states),
    "limit_clearance_min_rad": min(s["limit_clearance"] for s in states),
    "momentum_time_series": [s["momentum"].tolist() for s in states],
  }


def analyze(adapter, selection):
  windows = []
  for window in selection["windows"]:
    windows.append({**window, "result": evaluate(adapter, window)})
  pairs = []
  for wp in sorted({w["workpoint"] for w in windows}):
    positive = next(w for w in windows if w["workpoint"] == wp
                    and w["direction"] == "positive")
    negative = next(w for w in windows if w["workpoint"] == wp
                    and w["direction"] == "negative")
    a, b = positive["result"], negative["result"]
    if not (a["accepted"] and b["accepted"]):
      continue
    ra, rb = np.array(a["mean_residual_Nm"]), np.array(b["mean_residual_Nm"])
    odd, even = (ra-rb)/2, (ra+rb)/2
    inertia_odd = (np.array(a["mean_inertial_torque_Nm"])
                   - b["mean_inertial_torque_Nm"])/2
    model_odd = sum(np.array(a[k])-b[k] for k in (
      "mean_gas_Nm", "mean_passive_Nm", "mean_negative_bias_Nm"
    ))/2
    pairs.append({
      "workpoint": wp, "odd_CAN3_CAN2_Nm": odd[:2].tolist(),
      "odd_along_motion_Nm": (odd[:2]*[1, -1]).tolist(),
      "even_CAN3_CAN2_Nm": even[:2].tolist(),
      "odd_inertial_correction_Nm": inertia_odd[:2].tolist(),
      "odd_known_model_terms_Nm": model_odd[:2].tolist(),
    })
  good = [w["result"] for w in windows if w["result"]["accepted"]]
  summary = {
    "selected_windows": len(windows), "accepted_windows": len(good),
    "matched_workpoints": len(pairs),
    "both_motors_resistive_pairs": sum(
      min(p["odd_along_motion_Nm"]) > 0 for p in pairs
    ),
    "median_odd_along_motion_CAN3_CAN2_Nm": (
      np.median([p["odd_along_motion_Nm"] for p in pairs], axis=0).tolist()
      if pairs else None
    ),
    "median_even_CAN3_CAN2_Nm": (
      np.median([p["even_CAN3_CAN2_Nm"] for p in pairs], axis=0).tolist()
      if pairs else None
    ),
    "max_abs_odd_inertial_correction_Nm": max((
      max(abs(x) for x in p["odd_inertial_correction_Nm"]) for p in pairs
    ), default=None),
    "max_geometry_quadrature_difference_Nm": max((
      max(abs(x) for x in w["geometry_half_rate_difference_Nm"][:2])
      for w in good
    ), default=None),
    "max_encoder_speed_integral_disagreement_rad": max((
      max(abs(x) for x in w["encoder_velocity_integral_disagreement_rad"])
      for w in good
    ), default=None),
  }
  summary["motion_characteristics"] = motion_characteristics(windows)
  return {"windows": windows, "pairs": pairs, "summary": summary}


def motion_characteristics(windows):
  good = [w for w in windows if w["result"]["accepted"]]
  if not good:
    return {}
  velocities = [np.asarray(w["v"]) for w in good]
  pauses = [float(np.mean(np.max(np.abs(v), axis=1) < 0.06))
            for v in velocities]
  return {
    "median_window_mean_abs_speed_CAN3_CAN2_rad_s": np.median([
      np.mean(np.abs(v), axis=0) for v in velocities
    ], axis=0).tolist(),
    "median_window_p90_abs_speed_CAN3_CAN2_rad_s": np.median([
      np.percentile(np.abs(v), 90, axis=0) for v in velocities
    ], axis=0).tolist(),
    "both_motors_below_0p06_rad_s_fraction_median": float(np.median(pauses)),
    "both_motors_below_0p06_rad_s_fraction_max": max(pauses),
    "median_joint_displacement_abs_CAN3_CAN2_rad": np.median([
      np.abs(w["result"]["joint_displacement_rad"]) for w in good
    ], axis=0).tolist(),
    "interpretation": "Stop/start averages, not steady sliding coefficients.",
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--csv", type=Path, required=True)
  parser.add_argument("--xml", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--select-only", action="store_true")
  parser.add_argument("--selection", type=Path)
  args = parser.parse_args()
  if args.selection:
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if (selection["config"] != CONFIG
        or selection["csv_sha256"] != file_sha256(args.csv)):
      raise ValueError("Selection config or input hash differs")
  else:
    selection = select(load_samples(args.csv, phases=(7,)))
    selection["csv_sha256"] = file_sha256(args.csv)
  args.out.mkdir(parents=True, exist_ok=False)
  write_json(args.out / "selection.json", selection)
  if args.select_only:
    print(json.dumps({"windows": len(selection["windows"]),
                      "coverage": selection["coverage"]}))
    return
  adapter = ModelAdapter.create(args.xml, Mapping())
  report = analyze(adapter, selection)
  report.update({
    "config": CONFIG, "csv_sha256": selection["csv_sha256"],
    "model": adapter.validation, "script_sha256": file_sha256(Path(__file__)),
    "repository": repository_state(args.xml.resolve().parents[2]),
    "optimization_performed": False,
    "formula": "mean residual = mean[B.T*(motor+gas+passive-bias)] "
               "- (delta(Mr*u)/dt + mean[B.T*M*Bdot*u-Mrdot*u])",
    "assumptions": [
      "Ideal position/velocity closure; unchanged physical parameters.",
      "Wheel angle remains model default and wheel speed is unrecorded: zero.",
      "All derivatives are of model geometry along velocity, not log speed.",
      "Motor input uses 100 Hz ZOH; other terms use trapezoidal quadrature.",
      "Residual includes unmodeled friction plus all other model/input errors.",
      "Known mass/inertia are assumed, not calibrated by this calculation.",
      "No measured true motor torque or reconstructed 200 Hz update exists.",
    ],
  })
  write_json(args.out / "integral_residual.json", report)
  print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
  main()
