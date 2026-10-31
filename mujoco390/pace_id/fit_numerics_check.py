"""Closed-chain timestep gate before expensive friction optimization.

Uses cached training windows only. No parameter search or validation selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from .core import create_output_dir, file_sha256, write_json
from .fit import load_prepared
from .fit_core import FrictionReplay, INITIAL, errors


LIMITS = {"q_rmse_rad": 0.001, "v_rmse_rad_s": 0.02}


def compare(left, right):
  if left["failure"] or right["failure"]:
    return {"passed": False, "rmse_q3_q2_v3_v2_L_phi": None}
  delta = errors(left["prediction"], right["prediction"])
  rms = np.sqrt(np.mean(delta[1:] ** 2, axis=0))
  return {
    "passed": bool(max(rms[:2]) <= LIMITS["q_rmse_rad"]
                   and max(rms[2:4]) <= LIMITS["v_rmse_rad_s"]),
    "rmse_q3_q2_v3_v2_L_phi": rms.tolist(),
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--prepared", type=Path, required=True)
  parser.add_argument("--pilot", type=Path, required=True)
  parser.add_argument("--benchmark", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--timestep", type=float, default=0.0000390625)
  parser.add_argument("--focus-workpoint", type=int)
  parser.add_argument("--strict-solver", action="store_true")
  args = parser.parse_args()
  manifest = json.loads((args.prepared / "prepared.json").read_text())
  benchmark = json.loads(args.benchmark.read_text())
  numerics = benchmark["selection"]
  if (numerics is None or benchmark["mujoco_version"] != mujoco.__version__
      or benchmark.get("noslip_iterations", 0) != 0):
    parser.error("Need passing current-version benchmark without noslip")
  if not 0 < args.timestep <= numerics["timestep_s"]:
    parser.error("Timestep must be positive and no larger than benchmark")
  pilot = json.loads((args.pilot / "fit.json").read_text(encoding="utf-8"))
  windows, _ = load_prepared(
    args.prepared, Path(manifest["csv"]), Path(manifest["xml"])
  )
  windows = [w for w in windows if w.workpoint in (0, 14, 24)
             and w.label == "middle"]
  if len(windows) != 6:
    raise ValueError("Expected six fixed training windows")
  if args.focus_workpoint is not None:
    windows = [w for w in windows if w.workpoint == args.focus_workpoint]
    if not windows:
      parser.error("Focus must belong to the fixed training subset")
  from .fit_core import PARAMETERS
  settings = {
    "initial": INITIAL,
    "pilot": np.array([pilot["parameters"][key] for key in PARAMETERS]),
    "upper_corner": np.array([6., 6., 2.]),
  }
  out = create_output_dir(args.out)
  engines = [FrictionReplay(Path(manifest["xml"]), args.timestep / d, numerics)
             for d in (1, 2, 4)]
  if args.strict_solver:
    for e in engines:
      opt = e.adapter.model.opt
      opt.iterations, opt.ls_iterations = 200, 100
      opt.tolerance, opt.ls_tolerance = 1e-10, 0.001
  records, arrays = [], {}
  started = time.perf_counter()
  for label, parameters in settings.items():
    for e in engines:
      e.parameters(parameters)
    for w in windows:
      results = [e.replay(w, 0.5) for e in engines]
      comparisons = [compare(results[i], results[i+1]) for i in (0, 1)]
      records.append({
        "parameters_label": label, "window": w.name,
        "comparisons": comparisons,
        "diagnostics": [{k: v for k, v in r.items() if k != "prediction"}
                        for r in results],
      })
      for i, result in enumerate(results):
        arrays[f"{label}_{w.name}_dt{i}"] = result["prediction"]
    print(f"Checked {label}: {len(windows)} training windows", flush=True)
  report = {
    "scope": "Numerical gate only; no physical validation or optimization",
    "thresholds": LIMITS, "mujoco_version": mujoco.__version__,
    "timesteps_s": [e.timestep for e in engines], "numerics": numerics,
    "strict_solver": args.strict_solver,
    "focus_workpoint": args.focus_workpoint,
    "solver_options": {k: getattr(engines[0].adapter.model.opt, k)
                       for k in ("solver", "iterations", "ls_iterations",
                                 "tolerance", "ls_tolerance")},
    "parameters": {k: v.tolist() for k, v in settings.items()},
    "passed": all(c["passed"] for r in records for c in r["comparisons"]),
    "pair_passed": [all(r["comparisons"][i]["passed"] for r in records)
                    for i in (0, 1)],
    "records": records, "elapsed_s": time.perf_counter() - started,
    "source_sha256": {str(p): file_sha256(p) for p in (
      args.prepared / "prepared.json", args.pilot / "fit.json", args.benchmark,
    )},
    "code_sha256": {p.name: file_sha256(p)
                    for p in Path(__file__).parent.glob("*.py")},
    "limitations": [
      "Three parameter vectors are a gate, not a proof over the search box.",
      "At most six 0.5 s training windows; not the original WP29 failure.",
      "No comparison with measured loss is used to choose numerics.",
    ],
  }
  write_json(out / "check.json", report)
  np.savez_compressed(out / "trajectories.npz", **arrays)
  print(f"Numerical gate passed: {report['passed']}; "
        f"adjacent pairs: {report['pair_passed']}", flush=True)


if __name__ == "__main__":
  main()
