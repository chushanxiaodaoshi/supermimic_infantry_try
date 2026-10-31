"""Check native-friction probes and isolated numerical counterfactuals."""

from pathlib import Path

import numpy as np

from pace_id.fit_core import FitWindow, FrictionReplay
from pace_id.fit_diagnosis import configure, probe, trace


def example():
  e = FrictionReplay(Path(__file__).resolve().parents[1] / "mjcf/scene.xml")
  data = e.adapter.new_data()
  e.adapter.initialize(data, -0.6882019, -1.4049799, 0.2, -0.1)
  w = FitWindow(
    "test", 0, 4, "middle", np.arange(51) * 0.01,
    np.tile([-10., 10.], (51, 1)), np.tile(e.observe(data), (51, 1)),
    data.qpos.copy(), data.qvel.copy(),
  )
  return e, w


def test_friction_probe_matches_native_formula_and_force_balance():
  e, w = example()
  configure(e, np.array([5., 5., 1.]), 0.9)
  result = probe(e, w.qpos, w.qvel, w.command[0])
  assert result["equation_balance_max_error_Nm"] < 1e-8
  for row in result["active_friction_rows"]:
    assert abs(row["native_formula_error_Nm"]) < 1e-7
    assert row["cap_utilization"] <= 1.0 + 1e-7


def test_force_probes_do_not_change_replay():
  e, w = example()
  configure(e, np.array([5., 5., 1.]), 0.9)
  reference = e.replay(w)
  observed, probes = trace(e, w)
  assert reference["failure"] is observed["failure"] is None
  np.testing.assert_array_equal(reference["prediction"], observed["prediction"])
  assert len(probes) == 4


def test_probes_cover_final_available_sample_with_timestamp_jitter():
  e, w = example()
  configure(e, np.array([5., 5., 1.]), 0.9)
  w.time[1:] += 0.0001
  result, probes = trace(e, w)
  assert result["failure"] is None
  assert len(probes) == 4
  assert probes[-1]["device_relative_s"] == w.time[49]


def test_impedance_counterfactual_changes_no_physical_or_equality_parameters():
  e, w = example()
  parameters = np.array([5., 5., 1.])
  configure(e, parameters, 0.9)
  m = e.adapter.model
  names = ("body_mass", "body_inertia", "dof_armature", "dof_frictionloss",
           "dof_damping", "dof_solref", "eq_solref", "eq_solimp",
           "actuator_gainprm", "actuator_biasprm")
  before = {name: getattr(m, name).copy() for name in names}
  impedance = m.dof_solimp.copy()
  configure(e, parameters, 0.999)
  for name, expected in before.items():
    np.testing.assert_array_equal(getattr(m, name), expected)
  impedance[e.dofs, 0] = 0.999
  np.testing.assert_array_equal(m.dof_solimp, impedance)
