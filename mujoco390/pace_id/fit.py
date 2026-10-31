"""Offline CMA-ES baseline: two dry-friction values and shared damping.

Run from mujoco390: python -m pace_id.fit --help
No dependency installation, firmware edits, or training XML writes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

from .audit import audit_csv
from .core import create_output_dir, file_sha256, load_samples, write_json
from .diagnose import _initialize, build_short_windows
from .fit_core import (
  HORIZONS, INITIAL, PARAMETERS, Q_SCALE, STRESS_WP, UPPER, VALIDATION_WP,
  V_SCALE, V_WEIGHT, FitWindow, FrictionReplay, aggregate,
  evaluate_parameters, frozen_result, split_name, start_worker,
  trajectory_loss, window_metrics,
)
from .replay import (
  repository_state, validate_mapping, validate_model_coordinate_mapping,
)


ROOT = Path(__file__).resolve().parents[1]


def prepare(csv: Path, xml: Path, out: Path, timestep: float) -> tuple:
  audit = audit_csv(csv)
  write_json(out / "audit.json", audit)
  if not audit["accepted_for_baseline_replay"]:
    raise ValueError("CSV audit rejected input; see audit.json")
  samples = load_samples(csv)
  engine = FrictionReplay(xml, timestep)
  mapping = validate_mapping(samples, engine.adapter.mapping)
  jacobian = validate_model_coordinate_mapping(engine.adapter, samples)
  windows, descriptions, arrays = [], [], {}
  fields = samples.fields
  q3, q2 = engine.adapter.mapping.qpos_from_solver(
    fields["solver_q1_rad"], fields["solver_q2_rad"]
  )
  measured = np.column_stack([
    q3, q2, fields["can3_speed_rad_s"], fields["can2_speed_rad_s"],
    fields["measured_leg_length_m"], fields["measured_leg_phi_rad"],
  ])
  commands = np.column_stack([
    fields["can3_command_nm"], fields["can2_command_nm"],
  ])
  for i, window in enumerate(build_short_windows(samples)):
    data, record = _initialize(engine.adapter, samples, window)
    if window.time_s[-1] < 0.985:
      raise ValueError("Incomplete 1 s window")
    w = FitWindow(
      window.name, window.workpoint, window.phase, window.start_label,
      window.time_s, commands[window.indices], measured[window.indices],
      data.qpos.copy(), data.qvel.copy(),
    )
    windows.append(w)
    desc = {k: v for k, v in asdict(w).items() if not isinstance(v, np.ndarray)}
    desc.update({"split": split_name(w.workpoint), "initialization": record})
    descriptions.append(desc)
    for key in ("time", "command", "measured", "qpos", "qvel"):
      arrays[f"w{i}_{key}"] = getattr(w, key)
    if (i + 1) % 60 == 0:
      print(f"Prepared {i + 1} windows", flush=True)
  np.savez_compressed(out / "windows.npz", **arrays)
  # Save a compiled snapshot, including mesh/inertia data and replay settings.
  mujoco.mj_saveModel(engine.adapter.model, str(out / "baseline.mjb"), None)
  manifest = {
    "schema": "pace-friction-preparation-v1", "windows": descriptions,
    "csv": str(csv.resolve()), "csv_sha256": file_sha256(csv),
    "xml": str(xml.resolve()), "xml_sha256": file_sha256(xml),
    "npz_sha256": file_sha256(out / "windows.npz"),
    "compiled_model_sha256": file_sha256(out / "baseline.mjb"),
    "mapping": mapping, "jacobian": jacobian,
    "model_validation": engine.adapter.validation,
    "mujoco_version": mujoco.__version__, "numpy_version": np.__version__,
    "split_rule": {
      "validation_workpoints": VALIDATION_WP,
      "stress_workpoints": STRESS_WP,
      "all_other_workpoints": "train; both sweep types stay together",
    },
    "assumptions": [
      "Physical-ID mapping unchanged; active q/v from solver/logs.",
      "Passive q/v solved from closure; same cached state for every candidate.",
      "Unrecorded wheels/opposite leg: default positions, zero initial speed.",
      "Wheels stay free; base fixed; policy/PD and contacts disabled.",
      "300 N physical tendon force retained; no added 370.1*L compensation.",
      "Final CAN commands use device-time 100 Hz ZOH, not exact 200 Hz input.",
      "Torque feedback is not an independent torque calibration.",
      "No measured-speed differentiation; virtual states are checks, not loss.",
    ],
  }
  write_json(out / "prepared.json", manifest)
  return windows, manifest


def load_prepared(path: Path, csv: Path, xml: Path) -> tuple:
  manifest = json.loads((path / "prepared.json").read_text(encoding="utf-8"))
  for key, file in (("csv", csv), ("xml", xml), ("npz", path / "windows.npz")):
    if file_sha256(file) != manifest[f"{key}_sha256"]:
      raise ValueError(f"Prepared {key} checksum changed")
  if manifest["mujoco_version"] != mujoco.__version__:
    raise ValueError("Prepared MuJoCo version changed")
  timestep = manifest["model_validation"]["timestep_s"]
  model = FrictionReplay(xml, timestep).adapter.model
  buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
  mujoco.mj_saveModel(model, None, buffer)
  if hashlib.sha256(buffer).hexdigest() != manifest["compiled_model_sha256"]:
    raise ValueError("Compiled model/assets differ from prepared snapshot")
  windows = []
  with np.load(path / "windows.npz", allow_pickle=False) as archive:
    for i, d in enumerate(manifest["windows"]):
      windows.append(FitWindow(
        d["name"], d["workpoint"], d["phase"], d["label"],
        *(archive[f"w{i}_{key}"] for key in (
          "time", "command", "measured", "qpos", "qvel"
        )),
      ))
  return windows, manifest


def evaluate_case(engine, windows, parameters, out, name):
  engine.parameters(parameters)
  records, trajectories, failures = [], {}, []
  losses = {key: [] for key in ("train", "validation", "stress")}
  for i, w in enumerate(windows):
    result = engine.replay(w, 1.0)
    if result["failure"]:
      failures.append({"window": w.name, **result["failure"]})
    for horizon in (*HORIZONS, 1.0):
      records.append(window_metrics(w, result, horizon))
    # A failure beyond the training horizon must not contaminate its score.
    short = dict(result)
    n = int(np.searchsorted(w.time, 0.5 + 1e-9, side="right"))
    short["prediction"] = result["prediction"][:n]
    if short["failure"] and short["failure"]["time_s"] >= 0.5:
      short["failure"] = None
    losses[split_name(w.workpoint)].append(trajectory_loss(w, short))
    trajectories[f"w{i}_prediction"] = result["prediction"]
  write_json(out / f"{name}_windows.json", records)
  np.savez_compressed(out / f"{name}_trajectories.npz", **trajectories)
  return {
    "metrics": aggregate(records), "failures": failures,
    "loss_by_split": {k: float(np.mean(v)) for k, v in losses.items() if v},
  }


def evaluate_frozen(engine, windows):
  records, losses = [], {key: [] for key in ("train", "validation", "stress")}
  engine.parameters(None)
  for w in windows:
    initial = engine.replay(w, 0)["prediction"][0]
    result = frozen_result(w, initial, 1)
    losses[split_name(w.workpoint)].append(trajectory_loss(w, result))
    records.extend(window_metrics(w, result, h) for h in (*HORIZONS, 1.0))
  return {
    "metrics": aggregate(records),
    "loss_by_split": {k: float(np.mean(v)) for k, v in losses.items() if v},
  }


def convergence(xml, windows, parameters, timestep, numerics=None):
  chosen = [w for w in windows if w.workpoint in (0, 14, 24)
            and w.label == "middle"]
  records = []
  for name, params in (("original", None), ("candidate", parameters)):
    engines = [FrictionReplay(
      xml, timestep / divisor, numerics if name == "candidate" else None
    ) for divisor in (1, 2, 4)]
    for e in engines:
      e.parameters(params)
    for w in chosen:
      results = [e.replay(w) for e in engines]
      differences = []
      for left, right in ((0, 1), (1, 2)):
        if results[left]["failure"] or results[right]["failure"]:
          differences.append(None)
        else:
          from .fit_core import errors
          difference = errors(results[left]["prediction"],
                              results[right]["prediction"])
          differences.append(np.sqrt(np.mean(difference**2, axis=0)).tolist())
      records.append({
        "case": name, "window": w.name,
        "timesteps_s": [e.timestep for e in engines],
        "failures": [r["failure"] for r in results],
        "adjacent_dt_rmse_q3_q2_v3_v2_L_phi": differences,
      })
  return records


def run_optimizer(args, training, out):
  import cma
  options = {
    "seed": args.seed, "popsize": args.population, "maxiter": args.generations,
    "bounds": [0, 1], "verbose": -9, "verb_log": 0,
    "tolfun": 1e-5, "tolx": 1e-4,
  }
  optimizer = cma.CMAEvolutionStrategy((INITIAL / UPPER).tolist(), 0.15, options)
  start = time.perf_counter()
  with ProcessPoolExecutor(
    max_workers=args.workers, initializer=start_worker,
    initargs=(str(args.xml.resolve()), args.timestep, training, args.numerics),
  ) as pool, (out / "optimization.jsonl").open("x", encoding="utf-8") as log:
    while not optimizer.stop():
      population = optimizer.ask()
      parameters = [(np.asarray(x) * UPPER).tolist() for x in population]
      results = list(pool.map(evaluate_parameters, parameters))
      scores = [result["loss"] for result in results]
      optimizer.tell(population, scores)
      record = {
        "generation": optimizer.countiter,
        "parameters": parameters, "evaluations": results,
        "best_training_loss": float(optimizer.result.fbest),
        "best_parameters": (np.asarray(optimizer.result.xbest) * UPPER).tolist(),
        "elapsed_s": time.perf_counter() - start,
      }
      log.write(json.dumps(record) + "\n")
      log.flush()
      write_json(out / "checkpoint.json", record)
      if optimizer.countiter == 1 or optimizer.countiter % 5 == 0:
        print(
          f"Generation {optimizer.countiter}: train loss="
          f"{optimizer.result.fbest:.5g}, elapsed={record['elapsed_s']:.1f}s",
          flush=True,
        )
  return np.asarray(optimizer.result.xbest) * UPPER, {
    "best_training_loss": float(optimizer.result.fbest),
    "generations": optimizer.countiter, "evaluations": optimizer.countevals,
    "elapsed_s": time.perf_counter() - start,
    "stop_reason": {k: str(v) for k, v in optimizer.stop().items()},
  }


def summary(report: dict) -> str:
  lines = ["# PACE 三参数摩擦拟合", "",
           f"模式：{report['mode']}；仅为等效候选，未修改训练模型或固件。", "",
           "参数 CAN3/CAN2 干摩擦 Nm、共享阻尼 Nms/rad：",
           str(report["parameters"]), "",
           "验证按完整工作点划分；工作点29单列压力测试。",
           "以下RMSE仅统计成功窗口；失败另列。", ""]
  for case in ("original", "numerics_only", "candidate", "frozen"):
    if report.get(case) is None:
      continue
    lines.append(f"{case}：")
    for key, value in report[case]["metrics"].items():
      if key.endswith("_0.5s") or key.endswith("_1s"):
        rms = value["rmse_q3_q2_v3_v2_L_phi"]
        values = "无成功窗口" if rms is None else ", ".join(f"{x:.5g}" for x in rms)
        lines.append(
          f"- {key}: q3/q2/v3/v2/L/phi RMSE={values}; "
          f"失败 {value['failed']}/{value['windows']}; "
          f"疑似锁止 {value['locked_joint_count']}/{value['moving_joint_count']}"
        )
    lines.append(f"归一化损失：{report[case]['loss_by_split']}")
    lines.append("")
  lines.extend([
    f"参数距边界比例：{report['fraction_of_upper_bound']}",
    f"优化停止原因：{report['optimizer']['stop_reason']}",
    "峰峰值比仅检查是否锁止，不是频响或信噪比。",
    "不以训练损失下降宣称成功；需结合留出误差、锁止、步长敏感性。",
    "100 Hz指令重建、轮速未知、气弹簧定律及摩擦形式仍有限制。",
  ])
  return "\n".join(lines) + "\n"


def assessment(report: dict, records: list[dict]) -> dict:
  """Decision checks only: never feed validation results back to optimizer."""
  candidate = report["candidate"]["loss_by_split"]
  frozen = report["frozen"]["loss_by_split"]
  boundary = [name for name, fraction in zip(
    PARAMETERS, report["fraction_of_upper_bound"], strict=True
  ) if fraction >= 0.95 or fraction <= 0.01]
  beats_frozen = (
    candidate["validation"] < frozen["validation"]
    if "validation" in candidate else None
  )
  phase_metrics = {}
  for split in ("train", "validation", "stress"):
    for phase in (4, 7):
      chosen = [r for r in records if r["split"] == split
                and r["phase"] == phase and r["horizon_s"] == 0.5]
      valid = [r for r in chosen if r["failure"] is None]
      if not chosen:
        continue
      phase_metrics[f"{split}_phase{phase}"] = {
        "windows": len(chosen), "failures": len(chosen) - len(valid),
        "rmse_q3_q2_v3_v2_L_phi": np.sqrt(np.mean([
          np.square(r["rmse_q3_q2_v3_v2_L_phi"]) for r in valid
        ], axis=0)).tolist() if valid else None,
      }
  full = [r for r in records if r["horizon_s"] == 1.0
          and r["failure"] is None]
  margins = [r["passive_limit_min_rad"] for r in full
             if r.get("passive_limit_min_rad") is not None]
  return {
    "near_search_boundary": boundary,
    "beats_frozen_on_validation": beats_frozen,
    "accepted_for_training_model": False,
    "reason": (
      "Rejected baseline: validation does not beat the frozen-state control."
      if beats_frozen is False else
      "Manual review required; loss alone cannot establish identification."
    ),
    "candidate_phase_metrics_0p5s": phase_metrics,
    "candidate_full_1s_max_loop_gap_m": max(
      (r["loop_gap_max_m"] for r in full), default=None
    ),
    "candidate_full_1s_min_passive_limit_margin_rad": min(margins, default=None),
    "limitations": [
      "Boundary hits do not identify the unconstrained physical optimum.",
      "One seed and a fixed generation budget do not prove global convergence.",
      "WP29 1 s stress checks do not retest the original 2.975 s failure.",
      "No claim that three parameters can never fit under any bounds/model.",
      "Holdout excludes optimizer fitting but was seen in prior diagnostics.",
    ],
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--csv", type=Path, required=True)
  parser.add_argument("--xml", type=Path, default=ROOT / "mjcf/scene.xml")
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--prepared", type=Path)
  parser.add_argument("--prepare-only", action="store_true")
  parser.add_argument("--pilot", action="store_true")
  parser.add_argument("--generations", type=int, default=30)
  parser.add_argument("--population", type=int, default=8)
  parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
  parser.add_argument("--seed", type=int, default=20261015)
  parser.add_argument("--timestep", type=float, default=0.0025)
  parser.add_argument("--friction-benchmark", type=Path)
  parser.add_argument("--numerics-check", type=Path)
  args = parser.parse_args()
  args.numerics = None
  if args.friction_benchmark:
    bench = json.loads(args.friction_benchmark.read_text(encoding="utf-8"))
    args.numerics = bench["selection"]
    if args.numerics is None or bench["mujoco_version"] != mujoco.__version__:
      parser.error("No passing benchmark settings for the current MuJoCo")
    if bench.get("noslip_iterations", 0) != 0:
      parser.error("Replay does not enable noslip; incompatible benchmark")
    if args.timestep > args.numerics["timestep_s"] + 1e-12:
      parser.error("Requested timestep exceeds benchmark-validated step")
  if args.numerics_check:
    if args.numerics is None or args.prepared is None:
      parser.error("Numerics check requires benchmark and prepared windows")
    check = json.loads(args.numerics_check.read_text(encoding="utf-8"))
    if (not check["passed"] or check.get("focus_workpoint") is not None
        or check["mujoco_version"] != mujoco.__version__
        or check["numerics"] != args.numerics):
      parser.error("Need a passing full training-subset check of this profile")
    expected = check["source_sha256"]
    for path in (args.prepared / "prepared.json", args.friction_benchmark):
      if file_sha256(path) != expected.get(str(path)):
        parser.error("Numerical check does not match prepared data/benchmark")
    if args.timestep > check["timesteps_s"][0] + 1e-12:
      parser.error("Requested timestep exceeds closed-chain checked step")
    if check.get("strict_solver"):
      args.numerics["solver_options"] = {
        k: check["solver_options"][k] for k in (
          "iterations", "ls_iterations", "tolerance", "ls_tolerance"
        )
      }
  elif args.numerics is not None and not (args.pilot or args.prepare_only):
    parser.error("Changed-friction full fit requires --numerics-check")
  if min(args.generations, args.population, args.workers, args.timestep) <= 0:
    parser.error("Counts and timestep must be positive")
  if not args.prepare_only:
    try:
      import cma
    except ImportError:
      parser.error("Missing cma. User must install it; --prepare-only needs no cma.")
  out = create_output_dir(args.out)
  if args.prepared:
    windows, manifest = load_prepared(args.prepared, args.csv, args.xml)
  else:
    windows, manifest = prepare(args.csv, args.xml, out, args.timestep)
  write_json(out / "config.json", {
    "arguments": {k: str(v) if isinstance(v, Path) else v
                  for k, v in vars(args).items()},
    "python": sys.executable, "mujoco": mujoco.__version__,
    "numpy": np.__version__,
    "cma": None if args.prepare_only else importlib.metadata.version("cma"),
    "parameters": PARAMETERS, "bounds": [np.zeros(3).tolist(), UPPER.tolist()],
    "initial": INITIAL.tolist(), "horizons_s": HORIZONS,
    "loss": {"kind": "Huber", "q_scale_rad": Q_SCALE,
             "v_scale_rad_s": V_SCALE, "v_weight": V_WEIGHT,
             "aggregation": "equal horizons then equal windows; t=0 excluded"},
    "source": {k: manifest[k] for k in ("csv_sha256", "xml_sha256")},
    "repository": repository_state(ROOT),
    "code_sha256": {p.name: file_sha256(p) for p in Path(__file__).parent.glob("*.py")},
    "prepared_directory": str((args.prepared or out).resolve()),
    "friction_benchmark_sha256": (
      file_sha256(args.friction_benchmark) if args.friction_benchmark else None
    ),
    "numerics_check_sha256": (
      file_sha256(args.numerics_check) if args.numerics_check else None
    ),
  })
  if args.prepare_only:
    print(f"Prepared {len(windows)} windows: {out}", flush=True)
    return
  training = [w for w in windows if split_name(w.workpoint) == "train"]
  if args.pilot:
    training = [w for w in training if w.workpoint in (0, 14, 24)
                and w.label == "middle"]
    windows = training  # Do not inspect held-out results during pilot tuning.
  print(f"CMA-ES: {len(training)} train windows, {args.workers} workers", flush=True)
  engine = FrictionReplay(args.xml, args.timestep)
  baseline = evaluate_case(engine, windows, None, out, "original")
  engine = FrictionReplay(args.xml, args.timestep, args.numerics)
  numerics_only = None
  if args.numerics is not None:
    numerics_only = evaluate_case(engine, windows, None, out, "numerics_only")
  parameters, optimization = run_optimizer(args, training, out)
  candidate = evaluate_case(engine, windows, parameters, out, "candidate")
  frozen = evaluate_frozen(engine, windows)
  report = {
    "mode": "pilot" if args.pilot else "fit",
    "friction_numerics": args.numerics,
    "numerics_only": numerics_only,
    "parameters": dict(zip(PARAMETERS, parameters.tolist(), strict=True)),
    "fraction_of_upper_bound": (parameters / UPPER).tolist(),
    "optimizer": optimization, "original": baseline, "candidate": candidate,
    "frozen": frozen,
    "convergence": convergence(
      args.xml, windows, parameters, args.timestep, args.numerics
    ),
    "assumptions": manifest["assumptions"],
  }
  if file_sha256(args.xml) != manifest["xml_sha256"]:
    raise RuntimeError("Source XML changed during fitting")
  write_json(out / "fit.json", report)
  records = json.loads((out / "candidate_windows.json").read_text(encoding="utf-8"))
  write_json(out / "assessment.json", assessment(report, records))
  (out / "SUMMARY.md").write_text(summary(report), encoding="utf-8")
  print(f"Saved summary: {out / 'SUMMARY.md'}", flush=True)


if __name__ == "__main__":
  main()
