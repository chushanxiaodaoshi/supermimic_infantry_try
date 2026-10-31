"""Fitting invariants; callable directly when pytest is unavailable."""

from pathlib import Path

import mujoco
import numpy as np

from pace_id.fit_core import (
  FAILURE_LOSS, FitWindow, FrictionReplay, errors, split_name,
  trajectory_loss, window_metrics,
)


def example():
  engine = FrictionReplay(Path(__file__).resolve().parents[1] / "mjcf/scene.xml")
  a = engine.adapter
  d = a.new_data()
  a.initialize(d, -0.6882019, -1.4049799, 0.2, -0.1)
  time = np.arange(51) * 0.01
  w = FitWindow(
    "test", 0, 4, "middle", time, np.tile([-10., 10.], (51, 1)),
    np.tile(engine.observe(d), (51, 1)), d.qpos.copy(), d.qvel.copy(),
  )
  return engine, w


def test_fit_changes_only_active_friction_and_damping():
  engine, w = example()
  m = engine.adapter.model
  mass, armature = m.body_mass.copy(), m.dof_armature.copy()
  equality = m.eq_solref.copy()
  engine.parameters(np.array([1., 2., 0.3]))
  np.testing.assert_array_equal(m.dof_frictionloss[engine.dofs], [1., 2.])
  np.testing.assert_array_equal(m.dof_damping[engine.dofs], [0.3, 0.3])
  untouched = np.setdiff1d(np.arange(m.nv), engine.dofs)
  np.testing.assert_array_equal(
    m.dof_frictionloss[untouched], engine.original_friction[untouched]
  )
  np.testing.assert_array_equal(m.body_mass, mass)
  np.testing.assert_array_equal(m.dof_armature, armature)
  np.testing.assert_array_equal(m.eq_solref, equality)
  engine.parameters(None)
  np.testing.assert_array_equal(m.dof_frictionloss, engine.original_friction)
  np.testing.assert_array_equal(m.dof_damping, engine.original_damping)


def test_fitting_keeps_total_command_and_single_gas_compensation():
  engine, w = example()
  a = engine.adapter
  engine.parameters(np.array([1., 2., 0.3]))
  d = a.new_data()
  a.apply_logged_torque(d, -12.3, 11.4)
  np.testing.assert_array_equal(d.qfrc_applied[engine.dofs], [-12.3, 11.4])
  np.testing.assert_array_equal(d.ctrl[list(a.gas_actuator_ids)], [300., 300.])
  assert not np.any(a.model.actuator_gainprm[list(a.policy_actuator_ids)])


def test_replay_is_repeatable_and_command_has_no_lookahead():
  engine, w = example()
  engine.parameters(np.array([1., 2., 0.3]))
  original_qpos, original_qvel = w.qpos.copy(), w.qvel.copy()
  first = engine.replay(w, 0.03)
  second = engine.replay(w, 0.03)
  assert first["failure"] is None
  np.testing.assert_array_equal(first["prediction"], second["prediction"])
  w.command[2:] = [30, -30]
  changed = engine.replay(w, 0.03)
  np.testing.assert_array_equal(
    first["prediction"][:3], changed["prediction"][:3]
  )
  assert not np.allclose(first["prediction"][3], changed["prediction"][3])
  np.testing.assert_array_equal(w.qpos, original_qpos)
  np.testing.assert_array_equal(w.qvel, original_qvel)


def test_angles_wrap_and_failures_do_not_score_as_success():
  measured = np.zeros((2, 6))
  predicted = measured.copy()
  predicted[:, [0, 1, 5]] = 2 * np.pi + 0.01
  np.testing.assert_allclose(errors(predicted, measured)[:, [0, 1, 5]], 0.01)
  engine, w = example()
  result = {"prediction": w.measured[:1], "failure": {"time_s": 0.001}}
  assert trajectory_loss(w, result) == FAILURE_LOSS
  assert window_metrics(w, result, 0.5)["failure"] is not None


def test_whole_workpoints_are_reserved_and_locking_is_reported():
  assert split_name(1) == split_name(28) == "validation"
  assert split_name(29) == "stress"
  assert split_name(0) == "train"
  engine, w = example()
  prediction = w.measured.copy()
  w.measured[:, 0] += np.linspace(0, 0.1, len(w.time))
  result = {"prediction": prediction, "failure": None, "loop_gap_max_m": 0}
  metric = window_metrics(w, result, 0.5)
  assert metric["moving_joint_count"] == metric["locked_joint_count"] == 1


def test_cma_ask_tell_recovers_known_quadratic():
  import cma
  target = np.array([0.2, 0.7, 0.4])
  opt = cma.CMAEvolutionStrategy([0.5] * 3, 0.2, {
    "bounds": [0, 1], "seed": 123, "maxiter": 100,
    "verbose": -9, "verb_log": 0,
  })
  while not opt.stop():
    xs = opt.ask()
    opt.tell(xs, [float(np.sum((x-target)**2)) for x in xs])
  assert opt.result.fbest < 1e-8


def test_native_friction_changes_dynamics_and_observes_current_pose():
  engine, w = example()
  engine.parameters(None)
  baseline = engine.replay(w, 0.02)
  engine.parameters(np.array([3., 3., 0.1]))
  changed = engine.replay(w, 0.02)
  assert not np.allclose(baseline["prediction"][1:], changed["prediction"][1:])
  # State observed immediately after initialization agrees with full FK.
  a = engine.adapter
  d = a.new_data()
  d.qpos[:] = w.qpos
  d.qvel[:] = w.qvel
  mujoco.mj_forward(a.model, d)
  np.testing.assert_allclose(engine.observe(d)[4:], a.virtual_state(d)[:2])


def test_numerics_override_preserves_physics_mapping_and_equalities():
  xml = Path(__file__).resolve().parents[1] / "mjcf/scene.xml"
  config = {"d0": 0.9999, "timeconst_s": 0.00025,
            "timestep_s": 0.000078125,
            "solver_options": {"iterations": 200, "ls_iterations": 100,
                               "tolerance": 1e-10, "ls_tolerance": 0.001}}
  original = FrictionReplay(xml, config["timestep_s"])
  changed = FrictionReplay(xml, config["timestep_s"], config)
  before, after = original.adapter.model, changed.adapter.model
  for name in ("body_mass", "body_inertia", "dof_armature", "dof_frictionloss",
               "dof_damping", "eq_solimp", "eq_solref", "actuator_gainprm",
               "actuator_biasprm", "tendon_stiffness", "tendon_damping"):
    np.testing.assert_array_equal(getattr(before, name), getattr(after, name))
  expected = before.dof_solimp.copy()
  expected[changed.dofs, 0] = config["d0"]
  np.testing.assert_array_equal(expected, after.dof_solimp)
  expected = before.dof_solref.copy()
  expected[changed.dofs, 0] = config["timeconst_s"]
  np.testing.assert_array_equal(expected, after.dof_solref)
  np.testing.assert_array_equal(original.dofs, changed.dofs)
  for key, value in config["solver_options"].items():
    assert getattr(after.opt, key) == value
  try:
    FrictionReplay(xml, config["timestep_s"] * 2, config)
  except ValueError:
    pass
  else:
    raise AssertionError("Coarser-than-benchmark timestep was accepted")


def test_numerical_gate_uses_wrapped_angles_and_rejects_failures():
  from pace_id.fit_numerics_check import compare
  a = {"failure": None, "prediction": np.zeros((5, 6))}
  b = {"failure": None, "prediction": np.zeros((5, 6))}
  b["prediction"][:, [0, 1, 5]] = 2 * np.pi
  assert compare(a, b)["passed"]
  b["prediction"][:, 2] = 0.03
  assert not compare(a, b)["passed"]
  b["failure"] = {"kind": "test"}
  assert not compare(a, b)["passed"]
