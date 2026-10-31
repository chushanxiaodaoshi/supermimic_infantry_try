"""Select native-friction numerics against analytic one-hinge dynamics.

This is not hardware identification: all loads, inertia and friction values
are synthetic. Selection never uses a hardware replay loss.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import mujoco
import numpy as np

from .core import create_output_dir, write_json


IMPEDANCES = (0.9, 0.99, 0.999, 0.9999)
TIMECONSTS = (0.02, 0.005, 0.0025)
TIMESTEPS = (0.00125, 0.000625)
NOSLIP_ITERATIONS = 0
LIMITS = {"q_error_rad": 0.001, "v_error_rad_s": 0.02,
          "static_speed_rad_s": 0.001, "dt_q_difference_rad": 0.001}


def exact_step(q, v, torque, inertia, friction, damping, duration):
  """Exact piecewise Coulomb + viscous solution, including stop/reversal."""
  if duration < 1e-15:
    return q, v
  if abs(v) < 1e-12:
    if abs(torque) <= friction:
      return q, 0.0
    direction = np.sign(torque)
    v = 0.0
  else:
    direction = np.sign(v)
  force = torque - direction * friction
  crossing = float("inf")
  if damping:
    equilibrium = force / damping
    if v * equilibrium < 0:
      crossing = -inertia / damping * np.log(-equilibrium / (v-equilibrium))
    step = min(duration, crossing)
    decay = np.exp(-damping * step / inertia)
    qnext = q + equilibrium * step + (
      v-equilibrium
    ) * inertia / damping * (1-decay)
    vnext = equilibrium + (v-equilibrium) * decay
  else:
    acceleration = force / inertia
    if v * acceleration < 0:
      crossing = -v / acceleration
    step = min(duration, crossing)
    qnext = q + v * step + 0.5 * acceleration * step**2
    vnext = v + acceleration * step
  if crossing <= duration:
    return exact_step(qnext, 0., torque, inertia, friction, damping,
                      duration - crossing)
  return float(qnext), float(vnext)


def hinge(inertia, friction, damping, d0, timeconst, dt):
  # No contacts, gravity, policy, springs, or geometry-derived inertia.
  xml = f'''<mujoco><option gravity="0 0 0" timestep="{dt}"
    integrator="implicitfast" iterations="80"/>
    <worldbody><body><inertial pos="0 0 0" mass="1"
    diaginertia="{inertia} {inertia} {inertia}"/>
    <joint type="hinge" axis="0 0 1" frictionloss="{friction}"
    damping="{damping}" solreffriction="{timeconst} 1"
    solimpfriction="{d0} 0.95 0.001 0.5 2"/>
    </body></worldbody></mujoco>'''
  model = mujoco.MjModel.from_xml_string(xml)
  model.opt.noslip_iterations = NOSLIP_ITERATIONS
  return model


def run_case(inertia, friction, damping, direction, case, d0, timeconst, dt):
  if case == "static":
    v0, torque = 0., direction * friction * 0.8
  elif case == "slide":
    v0 = 0.
    torque = direction * (friction + inertia * 10 + damping * 0.5)
  elif case == "coast":
    v0, torque = direction * 0.5, 0.
  else:
    v0 = direction * 0.5
    torque = -direction * (friction + inertia * 10 + damping * 0.5)
  m = hinge(inertia, friction, damping, d0, timeconst, dt)
  data = mujoco.MjData(m)
  data.qvel[0] = v0
  data.qfrc_applied[0] = torque
  q, v, predicted, expected = 0., v0, [], []
  max_static_speed, max_power = 0., 0.
  for i in range(1, round(0.1 / dt) + 1):
    mujoco.mj_step(m, data)
    q, v = exact_step(q, v, torque, inertia, friction, damping, dt)
    if case == "static":
      max_static_speed = max(max_static_speed, abs(float(data.qvel[0])))
    if i % round(0.01 / dt) == 0:
      # Observe the present force with a fresh forward call (no RK states).
      mujoco.mj_forward(m, data)
      max_power = max(max_power, float(data.qfrc_constraint[0] * data.qvel[0]))
      predicted.append([float(data.qpos[0]), float(data.qvel[0])])
      expected.append([q, v])
  prediction, reference = np.array(predicted), np.array(expected)
  error = np.max(np.abs(prediction-reference), axis=0)
  failures = int(sum(w.number for w in data.warning))
  finite = bool(np.isfinite(prediction).all())
  return {
    "case": case, "inertia": inertia, "friction": friction,
    "damping": damping, "direction": direction, "timestep_s": dt,
    "q_error_rad": float(error[0]), "v_error_rad_s": float(error[1]),
    "static_speed_rad_s": max_static_speed, "positive_friction_power_W": max_power,
    "warnings": failures, "finite": finite,
    "prediction": prediction.tolist(), "reference": reference.tolist(),
  }


def benchmark():
  records = []
  # Match the low/high inertia scales implicated by the actual model's
  # reference inverse weights, without using any recorded trajectory.
  cases = list(itertools.product(
    (0.0008, 0.04), (1., 6.), (0., 2.), (-1., 1.),
    ("static", "slide", "coast", "reverse"),
  ))
  for timeconst, d0 in itertools.product(TIMECONSTS, IMPEDANCES):
    rows = [run_case(*case, d0, timeconst, dt)
            for case in cases for dt in TIMESTEPS]
    dt_difference = max(
      float(np.max(np.abs(
        np.array(rows[i]["prediction"])[:, 0]
        - np.array(rows[i+1]["prediction"])[:, 0]
      ))) for i in range(0, len(rows), 2)
    )
    maxima = {key: max(r[key] for r in rows) for key in (
      "q_error_rad", "v_error_rad_s", "static_speed_rad_s"
    )}
    maxima["dt_q_difference_rad"] = dt_difference
    passed = all(maxima[k] <= limit for k, limit in LIMITS.items())
    passed &= all(r["finite"] and not r["warnings"] for r in rows)
    passed &= max(r["positive_friction_power_W"] for r in rows) < 1e-8
    records.append({
      "d0": d0, "timeconst_s": timeconst, "passed": bool(passed),
      "maxima": maxima, "cases": rows,
    })
  # Preference fixed before results: retain the longest reference time and
  # the softest impedance satisfying every analytic test at both steps.
  passing = sorted([r for r in records if r["passed"]],
                   key=lambda r: (-r["timeconst_s"], r["d0"]))
  chosen = passing[0] if passing else None
  return {
    "schema": "pace-friction-numerical-benchmark-v1",
    "mujoco_version": mujoco.__version__, "thresholds": LIMITS,
    "noslip_iterations": NOSLIP_ITERATIONS,
    "selection_rule": "largest listed timeconst then smallest listed d0; all tests",
    "selection": None if chosen is None else {
      "d0": chosen["d0"], "timeconst_s": chosen["timeconst_s"],
      "timestep_s": TIMESTEPS[0],
    },
    "records": records,
    "sources": [
      "https://mujoco.readthedocs.io/en/stable/modeling.html#friction",
      "https://github.com/google-deepmind/mujoco/discussions/3413",
      "https://github.com/google-deepmind/mujoco/issues/2423",
    ],
    "scope": "Synthetic numerical check, not hardware-model validation.",
  }


def main():
  global IMPEDANCES, TIMECONSTS, TIMESTEPS, NOSLIP_ITERATIONS
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--impedances", type=float, nargs="+", default=IMPEDANCES)
  parser.add_argument("--timeconsts", type=float, nargs="+", default=TIMECONSTS)
  parser.add_argument("--timesteps", type=float, nargs=2, default=TIMESTEPS)
  parser.add_argument("--noslip-iterations", type=int, default=0)
  args = parser.parse_args()
  IMPEDANCES, TIMECONSTS, TIMESTEPS = (
    tuple(args.impedances), tuple(args.timeconsts), tuple(args.timesteps)
  )
  NOSLIP_ITERATIONS = args.noslip_iterations
  if any(x <= 0 for x in (*TIMECONSTS, *TIMESTEPS)):
    parser.error("Time values must be positive")
  if TIMESTEPS[1] >= TIMESTEPS[0] or NOSLIP_ITERATIONS < 0:
    parser.error("Require coarse then finer timestep, nonnegative noslip")
  if any(not 0 < x <= mujoco.mjMAXIMP for x in IMPEDANCES):
    parser.error("Impedance outside the simulator's unclamped range")
  if any(abs(0.01 / dt - round(0.01 / dt)) > 1e-9 for dt in TIMESTEPS):
    parser.error("Timesteps must divide the 10 ms observation period")
  out = create_output_dir(args.out)
  result = benchmark()
  write_json(out / "benchmark.json", result)
  print("Synthetic settings:", result["selection"], flush=True)
  print("Passed profiles:", sum(r["passed"] for r in result["records"]),
        "/", len(result["records"]), flush=True)


if __name__ == "__main__":
  main()
