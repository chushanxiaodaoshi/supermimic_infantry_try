"""Check impulse balance and closed-chain kinetic-energy identities."""

from pathlib import Path

import mujoco
import numpy as np

from pace_id.core import Mapping
from pace_id.initial_balance import tangent_basis
from pace_id.integral_residual import integrate, set_pose, terms
from pace_id.model import ModelAdapter


def adapter():
  return ModelAdapter.create(
    Path(__file__).resolve().parents[1] / "mjcf/scene.xml", Mapping()
  )


def test_reduced_bias_obeys_kinetic_energy_identity():
  a = adapter()
  data, probe = a.new_data(), a.new_data()
  set_pose(a, data, [-0.8, -1.2], [0.4, -0.3], first=True)
  state = terms(a, data, probe)
  basis, _, free = tangent_basis(a, data)
  u, bias = data.qvel[free].copy(), data.qfrc_bias.copy()
  data.qvel[:] = 0
  mujoco.mj_forward(a.model, data)
  coriolis = basis.T @ (bias-data.qfrc_bias)
  power = u @ (state["curvature_force"] + coriolis)
  energy_rate = 0.5 * u @ state["mass_rate_force"]
  np.testing.assert_allclose(power, energy_rate, atol=1e-7, rtol=1e-5)


def test_impulse_balance_handles_variable_reduced_mass():
  a = adapter()
  data, probe = a.new_data(), a.new_data()
  time = np.linspace(0, 0.1, 21)
  initial = np.array([-0.8, -1.2])
  velocity = np.array([0.4, -0.3])
  acceleration = np.array([0.8, -0.6])
  states = []
  for i, t in enumerate(time):
    set_pose(a, data, initial+velocity*t+0.5*acceleration*t*t,
             velocity+acceleration*t, first=i == 0)
    states.append(terms(a, data, probe))
  result = integrate(time, np.zeros((len(time), 2)), states)
  expected = np.trapezoid([
    s["reduced_mass"] @ np.r_[acceleration, 0] + s["curvature_force"]
    for s in states
  ], time, axis=0)/time[-1]
  np.testing.assert_allclose(
    result["mean_inertial_torque_Nm"], expected, atol=2e-7, rtol=2e-5
  )
