"""Matched opposite-motion diagnostics; no parameter estimation or replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .core import (
  Mapping, file_sha256, iter_samples, load_samples, wrap_angle, write_json,
)
from .initial_balance import tangent_basis
from .model import ModelAdapter
from .replay import relative_device_time, repository_state, split_sweep_runs


CONFIG = {
  "block_samples": 10,
  "moving_min_each_speed": 0.08,
  "moving_max_each_speed": 1.0,
  "block_speed_p90_p10_max": 0.15,
  "static_speed_p90_max": 0.06,
  "pair_joint_distance_max_rad": 0.02,
  "pair_length_distance_max_m": 0.002,
  "pair_phi_distance_max_rad": 0.015,
  "opposite_speed_relative_mismatch_max": 0.25,
  "maximum_pairs_per_sweep": 2,
  "model_passive_limit_clearance_min_rad": 0.03,
}


def candidates(samples):
  q = np.array(Mapping().qpos_from_solver(
    samples.fields["solver_q1_rad"], samples.fields["solver_q2_rad"]
  )).T
  velocity = np.column_stack([
    samples.fields[f"can{can}_speed_rad_s"] for can in (3, 2)
  ])
  command = np.column_stack([
    samples.fields[f"can{can}_command_nm"] for can in (3, 2)
  ])
  result = []
  for run, indices in enumerate(split_sweep_runs(samples)):
    time = relative_device_time(samples, indices)
    for offset in range(0, len(indices) - 9, 10):
      ids = indices[offset:offset + 10]
      v = velocity[ids]
      median_v = np.median(v, axis=0)
      spread = np.percentile(v, 90, axis=0) - np.percentile(v, 10, axis=0)
      sign = np.sign(median_v)
      steady = bool(np.max(spread) <= CONFIG["block_speed_p90_p10_max"])
      moving = bool(
        steady and np.all(np.abs(median_v) >= CONFIG["moving_min_each_speed"])
        and np.all(np.abs(median_v) <= CONFIG["moving_max_each_speed"])
        and np.all(v * sign >= 0.04)
      )
      static = bool(np.max(np.percentile(np.abs(v), 90, axis=0)) <=
                    CONFIG["static_speed_p90_max"])
      if not (moving or static):
        continue
      median_q = wrap_angle(q[ids[0]] + np.median(
        wrap_angle(q[ids] - q[ids[0]]), axis=0
      ))
      phi = samples.fields["measured_leg_phi_rad"][ids]
      result.append({
        "id": len(result), "run": run,
        "workpoint": int(samples.workpoint[ids[0]]),
        "phase": int(samples.phase[ids[0]]),
        "sequence": int(samples.sequence[ids[0]]),
        "device_time_us": int(samples.device_time_us[ids[0]]),
        "sweep_time_s": float(time[offset]),
        "duration_s": float(time[offset+9] - time[offset]),
        "moving": moving, "static": static,
        "q": median_q.tolist(), "v": median_v.tolist(),
        "speed_spread": spread.tolist(),
        "command": np.median(command[ids], axis=0).tolist(),
        "L": float(np.median(samples.fields["measured_leg_length_m"][ids])),
        "phi": float(wrap_angle(phi[0] + np.median(wrap_angle(phi-phi[0])))),
      })
  return result


def match_error(a, b):
  if a["run"] != b["run"] or not (a["moving"] and b["moving"]):
    return None
  va, vb = np.array(a["v"]), np.array(b["v"])
  if not np.all(va * vb < 0):
    return None
  qerr = float(np.max(np.abs(wrap_angle(np.array(a["q"]) - b["q"]))))
  le = abs(a["L"] - b["L"])
  pe = abs(float(wrap_angle(a["phi"] - b["phi"])))
  speed = float(np.max(np.abs(va + vb) / np.maximum(np.abs(va), np.abs(vb))))
  errors = np.array([qerr, le, pe, speed])
  tolerances = np.array([
    CONFIG["pair_joint_distance_max_rad"],
    CONFIG["pair_length_distance_max_m"],
    CONFIG["pair_phi_distance_max_rad"],
    CONFIG["opposite_speed_relative_mismatch_max"],
  ])
  return float(np.linalg.norm(errors / tolerances)) if np.all(
    errors <= tolerances
  ) else None


def select_pairs(blocks):
  pairs = []
  for run in sorted({b["run"] for b in blocks}):
    moving = [b for b in blocks if b["run"] == run and b["moving"]]
    positive = [b for b in moving if b["v"][0] > 0]
    negative = [b for b in moving if b["v"][0] < 0]
    options = []
    for a in positive:
      for b in negative:
        error = match_error(a, b)
        if error is not None:
          options.append((error, a["id"], b["id"]))
    used = set()
    count = 0
    for score, ai, bi in sorted(options):
      if ai in used or bi in used:
        continue
      pairs.append({"positive": ai, "negative": bi, "score": score})
      used.update((ai, bi))
      count += 1
      if count == CONFIG["maximum_pairs_per_sweep"]:
        break
  return pairs


def evaluate(adapter, block):
  model = adapter.model
  data = adapter.new_data()
  adapter.initialize(data, *block["q"], *block["v"])
  adapter.apply_logged_torque(data, *block["command"])
  basis, jac, free = tangent_basis(adapter, data)
  clearance = adapter.passive_limit_diagnostics(data)[
    "minimum_signed_distance_rad"
  ]
  if clearance < CONFIG["model_passive_limit_clearance_min_rad"]:
    return {"accepted": False, "reason": "passive_limit_clearance"}
  if np.any(data.warning.number):
    return {"accepted": False, "reason": "mujoco_warning"}
  motor = basis.T @ data.qfrc_applied
  gas = basis.T @ data.qfrc_actuator
  passive = basis.T @ data.qfrc_passive
  bias = basis.T @ data.qfrc_bias
  mass = np.zeros((model.nv, model.nv))
  mujoco.mj_fullM(model, mass, data.qM)
  reduced_mass = basis.T @ mass @ basis
  # Friction capacity is an upper bound in each projected coordinate.
  friction_capacity = np.abs(basis).T @ model.dof_frictionloss
  # Only a screening proxy; unobserved acceleration cannot be bounded by it.
  inertia_proxy = np.abs(reduced_mass[:2, :2]) @ (
    np.array(block["speed_spread"]) / block["duration_s"]
  )
  qvel = data.qvel.copy()
  data.qvel[:] = 0
  mujoco.mj_forward(model, data)
  gravity = -basis.T @ data.qfrc_bias
  velocity_bias = -bias - gravity
  residual = motor + gas + passive - bias
  return {
    "accepted": True,
    "motor_Nm": motor[:2].tolist(), "gas_Nm": gas[:2].tolist(),
    "gravity_Nm": gravity[:2].tolist(),
    "passive_Nm": passive[:2].tolist(),
    "velocity_bias_Nm": velocity_bias[:2].tolist(),
    "residual_before_coulomb_Nm": residual[:2].tolist(),
    "friction_capacity_Nm": friction_capacity[:2].tolist(),
    "inertia_screening_proxy_Nm": inertia_proxy.tolist(),
    "power_residual_W": float(residual[:2] @ np.array(block["v"])),
    "limit_clearance_rad": clearance,
    "closure_gap_m": adapter.loop_gap_m(data),
    "Jv_max": float(np.max(np.abs(jac @ qvel))),
    "reduced_mass_kg_m2": reduced_mass.tolist(),
  }


def describe(values):
  array = np.asarray(values)
  if not array.size:
    return None
  return {"median": float(np.median(array)),
          "p10": float(np.percentile(array, 10)),
          "p90": float(np.percentile(array, 90))}


def verify_selected_positions(csv_path, selected_blocks):
  """Cross-check low logged speeds against actual encoder displacement."""
  wanted = {}
  for block in selected_blocks:
    for seq in range(block["sequence"], block["sequence"] + 10):
      wanted[seq] = block["id"]
  grouped = {b["id"]: [] for b in selected_blocks}
  mapping = Mapping()
  for sample in iter_samples(csv_path):
    if sample.sequence in wanted:
      grouped[wanted[sample.sequence]].append(mapping.qpos_from_solver(
        sample.value("solver_q1_rad"), sample.value("solver_q2_rad")
      ))
  results = []
  for block in selected_blocks:
    q = np.array(grouped[block["id"]])
    if q.shape != (10, 2):
      raise ValueError("Missing selected encoder samples")
    dq = wrap_angle(q - q[0])
    span = np.ptp(dq, axis=0)
    measured_change = dq[-1]
    speed_change = np.array(block["v"]) * block["duration_s"]
    results.append({
      "id": block["id"], "static": block["static"],
      "encoder_span_rad": span.tolist(),
      "encoder_change_rad": measured_change.tolist(),
      "speed_times_duration_rad": speed_change.tolist(),
      "position_speed_disagreement_rad": (
        measured_change - speed_change
      ).tolist(),
    })
  return {
    "blocks": results,
    "static_encoder_span_max_rad": max((max(r["encoder_span_rad"])
      for r in results if r["static"]), default=None),
    "moving_position_speed_disagreement_max_rad": max((max(
      abs(x) for x in r["position_speed_disagreement_rad"]
    ) for r in results if not r["static"]), default=None),
  }


def analyze(blocks, pairs, adapter):
  selected = {p[k] for p in pairs for k in ("positive", "negative")}
  static_ids = []
  for run in sorted({b["run"] for b in blocks}):
    subset = [b for b in blocks if b["run"] == run and b["static"]]
    if subset:
      # Deterministic least-moving block; one per sweep avoids frame weighting.
      choice = min(subset, key=lambda b: (max(abs(x) for x in b["v"]),
                                         max(b["speed_spread"]), b["sequence"]))
      static_ids.append(choice["id"])
  selected.update(static_ids)
  computed = {i: evaluate(adapter, blocks[i]) for i in sorted(selected)}
  accepted_pairs = []
  for pair in pairs:
    ai, bi = pair["positive"], pair["negative"]
    a, b = blocks[ai], blocks[bi]
    ea, eb = computed[ai], computed[bi]
    if not (ea["accepted"] and eb["accepted"]):
      continue
    ra = np.array(ea["residual_before_coulomb_Nm"])
    rb = np.array(eb["residual_before_coulomb_Nm"])
    odd, even = (ra - rb) / 2, (ra + rb) / 2
    direction = np.sign(a["v"])
    even_model = sum(np.array(ea[k]) - eb[k] for k in (
      "gas_Nm", "gravity_Nm", "passive_Nm", "velocity_bias_Nm"
    )) / 2
    accepted_pairs.append({
      **pair, "workpoint": a["workpoint"], "phase": a["phase"],
      "odd_Nm": odd.tolist(), "even_Nm": even.tolist(),
      "odd_along_motion_Nm": (odd * direction).tolist(),
      "model_pose_speed_mismatch_odd_Nm": even_model.tolist(),
      "q_distance_rad": float(np.max(np.abs(wrap_angle(
        np.array(a["q"]) - b["q"]
      )))),
      "odd_power_W": float(odd @ (np.array(a["v"])-b["v"]) / 2),
    })
  static = [computed[i] for i in static_ids if computed[i]["accepted"]]
  aggregates = {}
  for phase in (4, 7):
    group = [p for p in accepted_pairs if p["phase"] == phase]
    aggregates[str(phase)] = {
      "pairs": len(group), "workpoints": len({p["workpoint"] for p in group}),
      "positive_odd_power_pairs": sum(p["odd_power_W"] > 0 for p in group),
      "odd_along_motion_each_motor_Nm": [describe([
        p["odd_along_motion_Nm"][i] for p in group
      ]) for i in range(2)],
      "even_each_motor_Nm": [describe([
        p["even_Nm"][i] for p in group
      ]) for i in range(2)],
      "model_mismatch_odd_max_abs_Nm": max((
        max(abs(x) for x in p["model_pose_speed_mismatch_odd_Nm"])
        for p in group
      ), default=None),
    }
  return {
    "selected_blocks": [{**blocks[i], "model": computed[i]}
                        for i in sorted(selected)],
    "pairs": accepted_pairs,
    "summary": {
      "candidate_blocks": len(blocks),
      "steady_moving_blocks": sum(b["moving"] for b in blocks),
      "near_static_blocks": sum(b["static"] for b in blocks),
      "selected_pairs": len(pairs), "accepted_pairs": len(accepted_pairs),
      "accepted_static_blocks": len(static),
      "model_rejections": sum(not r["accepted"] for r in computed.values()),
      "by_phase": aggregates,
      "static_abs_residual_each_motor_Nm": [describe([
        abs(s["residual_before_coulomb_Nm"][i]) for s in static
      ]) for i in range(2)],
      "inertia_proxy_max_Nm": max((max(r["inertia_screening_proxy_Nm"])
        for r in computed.values() if r["accepted"]), default=None),
    },
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
    saved = json.loads(args.selection.read_text(encoding="utf-8"))
    if saved["config"] != CONFIG or saved["source_csv_sha256"] != file_sha256(
      args.csv
    ):
      raise ValueError("Selection configuration or source hash mismatch")
    blocks, pairs = saved["blocks"], saved["pairs"]
  else:
    blocks = candidates(load_samples(args.csv))
    pairs = select_pairs(blocks)
  args.out.mkdir(parents=True, exist_ok=False)
  selection = {"blocks": blocks, "pairs": pairs, "config": CONFIG,
               "source_csv_sha256": file_sha256(args.csv)}
  write_json(args.out / "selection.json", selection)
  if args.select_only:
    print(json.dumps({"blocks": len(blocks), "moving": sum(
      b["moving"] for b in blocks), "static": sum(b["static"] for b in blocks),
      "pairs": len(pairs), "phases": {str(p): sum(
        blocks[x["positive"]]["phase"] == p for x in pairs
      ) for p in (4, 7)}}, ensure_ascii=False))
    return
  adapter = ModelAdapter.create(args.xml, Mapping())
  report = analyze(blocks, pairs, adapter)
  verification = verify_selected_positions(args.csv, report["selected_blocks"])
  write_json(args.out / "position_speed_check.json", verification)
  report["position_speed_check"] = {
    k: v for k, v in verification.items() if k != "blocks"
  }
  report.update({
    "config": CONFIG, "source_csv_sha256": selection["source_csv_sha256"],
    "model": adapter.validation, "optimization_performed": False,
    "repository": repository_state(args.xml.resolve().parents[2]),
    "script_sha256": file_sha256(Path(__file__)),
    "definition": "R=total motor+gas+gravity+velocity bias+passive; "
                  "missing resistance approximately -R only quasi-statically",
    "caveats": [
      "Odd component supports direction dependence, not uniquely Coulomb friction.",
      "Even component includes posture, bias, torque scale, and unmodeled dynamics.",
      "Velocity spread is a screening proxy, NOT an acceleration bound.",
      "No differentiation of measured speed, optimization, or sign changes.",
      "Unrecorded wheel velocity remains zero; telemetry misses 200 Hz updates.",
      "Pairs are descriptive, correlated, and not independent statistical trials.",
    ],
  })
  write_json(args.out / "direction_residual.json", report)
  print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
  main()
