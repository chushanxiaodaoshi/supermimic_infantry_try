"""Closed-chain force projection independently checked by virtual work."""

from pathlib import Path

import mujoco
import numpy as np

from pace_id.core import Mapping
from pace_id.initial_balance import dense_constraint_jacobian, tangent_basis
from pace_id.model import ModelAdapter


def test_tangent_projection_preserves_virtual_work():
  adapter = ModelAdapter.create(
    Path(__file__).resolve().parents[1] / "mjcf/scene.xml", Mapping()
  )
  data = adapter.new_data()
  adapter.initialize(data, -0.7, -1.3, 0.03, -0.04)
  adapter.apply_logged_torque(data, -9.0, 10.0)
  basis, jac, _ = tangent_basis(adapter, data)
  np.testing.assert_allclose(jac @ basis, 0, atol=1e-10)
  # Friction/equality forces must sum to MuJoCo's constraint force vector.
  efc = dense_constraint_jacobian(adapter.model, data)
  np.testing.assert_allclose(
    efc.T @ data.efc_force, data.qfrc_constraint, atol=1e-10
  )
  tendon = int(adapter.model.actuator_trnid[adapter.gas_actuator_ids[0], 0])
  delta = np.array([0.3, -0.2, 0.1])
  lengths = []
  for sign in (1, -1):
    trial = adapter.new_data()
    trial.qpos[:] = data.qpos
    mujoco.mj_integratePos(
      adapter.model, trial.qpos, basis @ delta, sign * 1e-6
    )
    mujoco.mj_fwdPosition(adapter.model, trial)
    lengths.append(trial.ten_length[tendon])
  power = delta @ basis.T @ data.qfrc_actuator
  fd_power = adapter.gas_force_n * (lengths[0] - lengths[1]) / 2e-6
  np.testing.assert_allclose(power, fd_power, rtol=1e-6, atol=1e-7)
