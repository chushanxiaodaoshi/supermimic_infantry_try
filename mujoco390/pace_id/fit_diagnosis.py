"""Read-only force/regularization diagnostics of a completed friction fit.

Only temporary native-friction impedance changes are used as numerical
counterfactuals. No optimizer, CSV parsing, or writes to source XML.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .core import create_output_dir, file_sha256, write_json
from .fit import load_prepared
from .fit_core import PARAMETERS, FrictionReplay, errors, window_metrics
from .initial_balance import dense_constraint_jacobian, tangent_basis
from .replay import repository_state


def probe(engine, qpos, qvel, command):
  """Evaluate forces in a separate MjData, preserving the replay warm start."""
  a, m = engine.adapter, engine.adapter.model
  data = a.new_data()
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  a.apply_logged_torque(data, *command)
  basis, _, _ = tangent_basis(a, data)
  jac = dense_constraint_jacobian(m, data)
  components = {
    "motor": data.qfrc_applied.copy(),
    "gas": data.qfrc_actuator.copy(),
    "passive": data.qfrc_passive.copy(),
    "negative_bias": -data.qfrc_bias.copy(),
  }
  for name, kinds in (("friction", (1, 2)), ("equality", (0,)),
                      ("limit", (3, 4)), ("contact", (5, 6, 7))):
    mask = np.isin(data.efc_type, kinds)
    components[name] = jac[mask].T @ data.efc_force[mask]
  mass = np.zeros((m.nv, m.nv))
  mujoco.mj_fullM(m, mass, data.qM)
  identity_error = float(np.max(np.abs(
    mass @ data.qacc - sum(components.values())
  )))
  friction_rows = []
  for can_id, dof in zip((3, 2), engine.dofs, strict=True):
    rows = np.flatnonzero((data.efc_type == 1) & (data.efc_id == dof))
    if len(rows) != 1:
      raise ValueError(f"Expected one native friction row for CAN{can_id}")
    row = int(rows[0])
    cap = float(data.efc_frictionloss[row])
    force = float(data.efc_force[row])
    aref = float(data.efc_aref[row])
    regularization = float(data.efc_R[row])
    expected = float(np.clip(
      (aref - jac[row] @ data.qacc) / regularization, -cap, cap
    ))
    friction_rows.append({
      "can_id": can_id, "force_Nm": force, "cap_Nm": cap,
      "cap_utilization": abs(force) / cap,
      "regularization_R": regularization,
      "reference_acceleration": aref,
      "native_formula_error_Nm": force - expected,
      "speed_rad_s": float(data.qvel[dof]),
      "acceleration_rad_s2": float(data.qacc[dof]),
    })
  return {
    "active_friction_rows": friction_rows,
    "projected_components_CAN3_CAN2_wheel": {
      name: (basis.T @ value).tolist() for name, value in components.items()
    },
    "active_acceleration_rad_s2": data.qacc[engine.dofs].tolist(),
    "equation_balance_max_error_Nm": identity_error,
    "loop_gap_m": a.loop_gap_m(data),
    "limit_margin_rad": a.passive_limit_diagnostics(data)[
      "minimum_signed_distance_rad"
    ],
  }


def configure(engine, parameters, impedance):
  engine.parameters(parameters)
  # Only d0 changes: keep reference time constant, d_width, friction caps,
  # viscous damping, equality parameters, geometry and inertias unchanged.
  engine.adapter.model.dof_solimp[engine.dofs, 0] = impedance


def trace(engine, w):
  # A jittered sample just after 0.5 s is not part of this replay. Select the
  # last available sample at/before each target, not the nearest future one.
  points = {max(0, int(np.searchsorted(w.time, t + 1e-9, side="right")) - 1)
            for t in (0, 0.1, 0.25, 0.5)}
  probes = []

  def observer(data, i):
    if i in points:
      probes.append({
        "device_relative_s": float(w.time[i]),
        "state": probe(engine, data.qpos, data.qvel, w.command[i]),
      })

  result = engine.replay(w, observer=observer)
  return result, probes


def diagnose(prepared, fit_dir, out):
  manifest = json.loads((prepared / "prepared.json").read_text(encoding="utf-8"))
  fit = json.loads((fit_dir / "fit.json").read_text(encoding="utf-8"))
  windows, _ = load_prepared(
    prepared, Path(manifest["csv"]), Path(manifest["xml"])
  )
  selected = [w for w in windows if (
    w.workpoint in (0, 14, 24) and w.label == "middle"
  ) or w.name in (
    "wp22_phi_sweep_run44_early", "wp29_length_sweep_run59_early"
  )]
  if len(selected) != 8:
    raise ValueError("Expected six training, one worst-validation, one stress window")
  xml = Path(manifest["xml"])
  params = np.array([fit["parameters"][key] for key in PARAMETERS])
  records, arrays = [], {}
  for impedance in (0.9, 0.99, 0.999):
    engine = FrictionReplay(xml, 0.00125)
    configure(engine, params, impedance)
    for i, w in enumerate(selected):
      result, probes = trace(engine, w)
      static = probe(engine, w.qpos, np.zeros_like(w.qvel), w.command[0])
      metric = window_metrics(w, result, 0.5)
      records.append({
        "window": w.name, "impedance_d0": impedance,
        "metric": metric, "probes": probes, "static_counterfactual": static,
      })
      arrays[f"imp{impedance}_w{i}_prediction"] = result["prediction"]
  convergence = []
  for impedance in (0.9, 0.999):
    engines = [FrictionReplay(xml, dt) for dt in (0.00125, 0.000625, 0.0003125)]
    for engine in engines:
      configure(engine, params, impedance)
    for w in selected:
      results = [engine.replay(w) for engine in engines]
      differences = []
      for left, right in ((0, 1), (1, 2)):
        if results[left]["failure"] or results[right]["failure"]:
          differences.append(None)
        else:
          differences.append(np.sqrt(np.mean(errors(
            results[left]["prediction"], results[right]["prediction"]
          )**2, axis=0)).tolist())
      convergence.append({
        "window": w.name, "impedance_d0": impedance,
        "adjacent_dt_rmse_q3_q2_v3_v2_L_phi": differences,
        "failures": [r["failure"] for r in results],
      })
  reference = FrictionReplay(xml, 0.00125)
  configure(reference, params, 0.9)
  m, dofs = reference.adapter.model, reference.dofs
  report = {
    "parameters_unchanged": dict(zip(PARAMETERS, params.tolist(), strict=True)),
    "original_dof_invweight0": m.dof_invweight0[dofs].tolist(),
    "original_friction_solref": m.dof_solref[dofs].tolist(),
    "original_friction_solimp": m.dof_solimp[dofs].tolist(),
    "records": records, "convergence": convergence,
    "provenance": {
      "prepared": str(prepared), "fit": str(fit_dir),
      "fit_sha256": file_sha256(fit_dir / "fit.json"),
      "script_sha256": file_sha256(Path(__file__)),
      "repository": repository_state(Path(__file__).parent),
      "mujoco": mujoco.__version__, "numpy": np.__version__,
    },
    "interpretation_limits": [
      "Exploratory counterfactuals, not a second fit or held-out acceptance.",
      "8 selected windows cannot establish performance over all workpoints.",
      "Frictionloss is a cap, not necessarily the realized sliding torque.",
      "Higher d0 may suppress slip while still locking out real motion.",
      "The 2.975 s WP29 failure has not been retested here.",
      "Force probes run cold forward dynamics in a separate data object.",
    ],
  }
  if file_sha256(xml) != manifest["xml_sha256"]:
    raise RuntimeError("XML unexpectedly changed")
  write_json(out / "diagnosis.json", report)
  np.savez_compressed(out / "traces.npz", **arrays)
  summary = summarize(report)
  (out / "SUMMARY.md").write_text(summary, encoding="utf-8")
  print(summary)


def summarize(report):
  lines = ["# PACE 首轮拟合后的摩擦实现检查", "",
           "只改内存中两主动关节摩擦约束的d0，不改物理参数、闭链约束或原XML。",
           "这不是第二轮拟合，也不是新验证集上的性能证明。", ""]
  for imp in (0.9, 0.99, 0.999):
    rows = [r for r in report["records"] if r["impedance_d0"] == imp]
    valid = [r for r in rows if r["metric"]["failure"] is None]
    if valid:
      rms = np.sqrt(np.mean([
        np.square(r["metric"]["rmse_q3_q2_v3_v2_L_phi"]) for r in valid
      ], axis=0))
      lines.append(
        f"d0={imp}: 8个窗口0.5s RMS(q3/q2/L/phi)="
        + "/".join(f"{v:.5f}" for v in rms[[0, 1, 4, 5]])
        + f"；数值失败={len(rows)-len(valid)}；"
        + f"低幅运动关节={sum(r['metric']['locked_joint_count'] for r in valid)}"
        + f"/{sum(r['metric']['moving_joint_count'] for r in valid)}"
      )
  lines.extend(["", "完整力分解、公式一致性、静止反事实及步长检查见diagnosis.json。",
                "少量窗口改善不意味着已识别真实摩擦；仍需防止锁止式假拟合。"])
  return "\n".join(lines) + "\n"


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--prepared", type=Path, required=True)
  parser.add_argument("--fit", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  diagnose(args.prepared, args.fit, create_output_dir(args.out))


if __name__ == "__main__":
  main()
