"""Three-parameter closed-chain replay; no optimizer or file mutation here."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .core import Mapping, wrap_angle
from .diagnose import _failure_kind
from .model import ModelAdapter


HORIZONS = (0.1, 0.25, 0.5)
VALIDATION_WP = (1, 8, 15, 22, 25, 28)
STRESS_WP = (29,)
PARAMETERS = ("can3_friction_Nm", "can2_friction_Nm", "shared_damping_Nms_rad")
UPPER = np.array([6.0, 6.0, 2.0])
INITIAL = np.array([1.5, 1.5, 0.1])
FAILURE_LOSS = 1.0e6
Q_SCALE = 0.05
V_SCALE = 0.5
V_WEIGHT = 0.1


def split_name(workpoint: int) -> str:
  if workpoint in STRESS_WP:
    return "stress"
  return "validation" if workpoint in VALIDATION_WP else "train"


@dataclass
class FitWindow:
  name: str
  workpoint: int
  phase: int
  label: str
  time: np.ndarray
  command: np.ndarray
  measured: np.ndarray  # q3, q2, v3, v2, L, phi
  qpos: np.ndarray
  qvel: np.ndarray


class FrictionReplay:
  """Only the two physical-ID active DOFs are fitted in memory.

  frictionloss uses MuJoCo's native dry-friction constraints, including
  stiction. It is not a separate hand-coded sign(velocity) force. Passive
  joint friction remains unchanged; these active-joint values are equivalent
  model parameters, not independently measured motor friction.
  """

  def __init__(self, xml: Path, timestep: float = 0.0025, numerics=None):
    self.adapter = ModelAdapter.create(xml, Mapping(), timestep_s=timestep)
    a = self.adapter
    self.dofs = np.array([a.can3_dof, a.can2_dof])
    self.qids = np.array([a.can3_qpos, a.can2_qpos])
    self.original_friction = a.model.dof_frictionloss.copy()
    self.original_damping = a.model.dof_damping.copy()
    self.timestep = timestep
    limited = [j for j in a.passive_joint_ids if a.model.jnt_limited[j]]
    self.limit_qids = a.model.jnt_qposadr[limited]
    self.limit_ranges = a.model.jnt_range[limited]
    if numerics is not None:
      if timestep > numerics["timestep_s"] + 1e-12:
        raise ValueError("Timestep exceeds the validated friction benchmark")
      a.model.dof_solimp[self.dofs, 0] = numerics["d0"]
      a.model.dof_solref[self.dofs, 0] = numerics["timeconst_s"]
      for key, value in numerics.get("solver_options", {}).items():
        if key not in ("iterations", "ls_iterations", "tolerance", "ls_tolerance"):
          raise ValueError(f"Unsupported numerical override: {key}")
        setattr(a.model.opt, key, value)

  def parameters(self, values: np.ndarray | None) -> None:
    m = self.adapter.model
    m.dof_frictionloss[:] = self.original_friction
    m.dof_damping[:] = self.original_damping
    if values is not None:
      values = np.asarray(values, dtype=float)
      if (values.shape != (3,) or not np.isfinite(values).all()
          or np.any(values < 0) or np.any(values > UPPER)):
        raise ValueError("Invalid bounded friction parameter vector")
      # Replace, do not add to the XML's active-joint friction/damping.
      m.dof_frictionloss[self.dofs] = values[:2]
      m.dof_damping[self.dofs] = values[2]

  def observe(self, data: mujoco.MjData) -> np.ndarray:
    a = self.adapter
    vector = data.xmat[a.base_id].reshape(3, 3).T @ (
      data.site_xpos[a.wheel_site_id] - data.site_xpos[a.hip_site_id]
    )
    return np.r_[
      data.qpos[self.qids], data.qvel[self.dofs], np.linalg.norm(vector),
      np.arctan2(vector[0], -vector[2]),
    ]

  def replay(self, w: FitWindow, duration: float = 0.5, observer=None) -> dict:
    a, m = self.adapter, self.adapter.model
    d = a.new_data()
    d.qpos[:] = w.qpos
    d.qvel[:] = w.qvel
    d.qacc_warmstart[:] = 0
    d.time = 0
    mujoco.mj_forward(m, d)
    count = int(np.searchsorted(w.time, duration + 1e-9, side="right"))
    prediction = np.empty((count, 6))
    peak_speed, peak_acceleration, gap = 0.0, 0.0, 0.0
    limit_margin = float("inf")
    failure = None
    completed = 0
    for i in range(count):
      # Advance exactly to each device timestamp. The command at i is not
      # applied before reaching i; command[i-1] owns [t[i-1], t[i]).
      if i:
        a.apply_logged_torque(d, *w.command[i - 1])
      while d.time < w.time[i] - 1e-12:
        dt = min(self.timestep, float(w.time[i] - d.time))
        m.opt.timestep = dt
        previous_time = float(d.time)
        mujoco.mj_step(m, d)
        kind, details = _failure_kind(d, previous_time, dt)
        peak_speed = max(peak_speed, details["post_step_speed_peak_rad_s"])
        peak_acceleration = max(
          peak_acceleration, details["post_step_acceleration_peak_rad_s2"]
        )
        if kind:
          failure = {"kind": kind, "time_s": previous_time, **details}
          break
      if failure:
        break
      # mj_step leaves derived positions at the pre-integration state.
      # Update kinematics without a second forward dynamics solve.
      mujoco.mj_kinematics(m, d)
      prediction[i] = self.observe(d)
      gap = max(gap, a.loop_gap_m(d))
      if self.limit_qids.size:
        positions = d.qpos[self.limit_qids]
        margin = np.min(np.minimum(positions - self.limit_ranges[:, 0],
                                   self.limit_ranges[:, 1] - positions))
        limit_margin = min(limit_margin, float(margin))
      completed = i + 1
      if observer is not None:
        observer(d, i)  # Read-only observer; diagnostics must clone before forward.
    m.opt.timestep = self.timestep
    return {
      "prediction": prediction[:completed], "failure": failure,
      "peak_speed": peak_speed, "peak_acceleration": peak_acceleration,
      "loop_gap_max_m": gap,
      "passive_limit_min_rad": limit_margin if np.isfinite(limit_margin) else None,
    }


def errors(prediction: np.ndarray, measured: np.ndarray) -> np.ndarray:
  result = prediction - measured
  result[:, [0, 1, 5]] = wrap_angle(result[:, [0, 1, 5]])
  return result


def huber(value: np.ndarray) -> np.ndarray:
  absolute = np.abs(value)
  return np.where(absolute <= 1.0, 0.5 * value**2, absolute - 0.5)


def trajectory_loss(w: FitWindow, result: dict) -> float:
  if result["failure"] is not None:
    return FAILURE_LOSS
  error = errors(result["prediction"], w.measured[:len(result["prediction"])])
  if not np.isfinite(error).all():
    return FAILURE_LOSS
  if len(error) < 2 or w.time[len(error) - 1] < HORIZONS[-1] - 0.015:
    raise ValueError("Incomplete trajectory without a recorded failure")
  values = []
  for horizon in HORIZONS:
    mask = (w.time[:len(error)] > 0) & (w.time[:len(error)] <= horizon + 1e-9)
    if not np.any(mask):
      raise ValueError("Missing requested fitting horizon")
    qloss = np.mean(huber(error[mask, :2] / Q_SCALE))
    vloss = np.mean(huber(error[mask, 2:4] / V_SCALE))
    values.append(float(qloss + V_WEIGHT * vloss))
  return float(np.mean(values))


def frozen_result(w: FitWindow, initial: np.ndarray, duration: float) -> dict:
  count = int(np.searchsorted(w.time, duration + 1e-9, side="right"))
  prediction = np.tile(initial, (count, 1))
  prediction[:, 2:4] = 0
  return {"prediction": prediction, "failure": None, "loop_gap_max_m": 0}


def window_metrics(w: FitWindow, result: dict, horizon: float) -> dict:
  base = {
    "window": w.name, "workpoint": w.workpoint, "phase": w.phase,
    "split": split_name(w.workpoint), "horizon_s": horizon,
  }
  if result["failure"] and result["failure"]["time_s"] < horizon:
    return {**base, "failure": result["failure"]}
  mask = w.time[:len(result["prediction"])] <= horizon + 1e-9
  predicted = result["prediction"][mask]
  measured = w.measured[:len(predicted)]
  if len(predicted) < 2 or w.time[len(predicted) - 1] < horizon - 0.015:
    return {**base, "failure": {"kind": "incomplete_horizon"}}
  err = errors(predicted, measured)
  real_span = np.ptp(np.unwrap(measured[:, :2], axis=0), axis=0)
  sim_span = np.ptp(np.unwrap(predicted[:, :2], axis=0), axis=0)
  moving = real_span >= 0.01
  return {
    **base, "failure": None,
    "rmse_q3_q2_v3_v2_L_phi": np.sqrt(np.mean(err[1:]**2, axis=0)).tolist(),
    "initial_errors": err[0].tolist(),
    "real_q_span_rad": real_span.tolist(),
    "sim_q_span_rad": sim_span.tolist(),
    "span_ratio": [
      float(sim_span[i] / real_span[i]) if moving[i] else None
      for i in range(2)
    ],
    "moving_joint_count": int(np.sum(moving)),
    "locked_joint_count": int(np.sum(moving & (sim_span < 0.2 * real_span))),
    "loop_gap_max_m": result["loop_gap_max_m"],
    "passive_limit_min_rad": result.get("passive_limit_min_rad"),
    "diagnostic_extrema_scope": "entire replay, not this horizon prefix",
  }


def aggregate(records: list[dict]) -> dict:
  groups = {}
  for split in ("train", "validation", "stress"):
    for horizon in (0.1, 0.25, 0.5, 1.0):
      selected = [r for r in records if r["split"] == split
                  and r["horizon_s"] == horizon]
      if not selected:
        continue
      success = [r for r in selected if r["failure"] is None]
      rms = None
      if success:
        rms = np.sqrt(np.mean([
          np.square(r["rmse_q3_q2_v3_v2_L_phi"]) for r in success
        ], axis=0)).tolist()
      groups[f"{split}_{horizon:g}s"] = {
        "windows": len(selected), "failed": len(selected) - len(success),
        "rmse_q3_q2_v3_v2_L_phi": rms,
        "moving_joint_count": sum(r["moving_joint_count"] for r in success),
        "locked_joint_count": sum(r["locked_joint_count"] for r in success),
      }
  return groups


_WORKER = None
_WINDOWS = None


def start_worker(xml: str, timestep: float, windows: list[FitWindow],
                 numerics=None) -> None:
  global _WORKER, _WINDOWS
  _WORKER = FrictionReplay(Path(xml), timestep, numerics)
  _WINDOWS = windows


def evaluate_parameters(values: list[float]) -> dict:
  _WORKER.parameters(np.asarray(values))
  losses, failures = [], []
  for w in _WINDOWS:
    result = _WORKER.replay(w)
    losses.append(trajectory_loss(w, result))
    if result["failure"]:
      failures.append({"window": w.name, **result["failure"]})
  return {"loss": float(np.mean(losses)), "failures": failures}
