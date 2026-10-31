"""Independent checks of the analytic dry-friction benchmark reference."""

import numpy as np

from pace_id.friction_bench import exact_step, hinge, run_case


def test_analytic_static_and_coasting_stop():
  assert exact_step(0., 0., 0.8, 1., 1., 0., 1.) == (0., 0.)
  # v=1 under unit deceleration stops after 1 s at q=0.5.
  np.testing.assert_allclose(exact_step(0., 1., 0., 1., 1., 0., 2.), [0.5, 0])


def test_analytic_reverse_crossing_and_viscous_solution():
  # First stop at t=1/3 under -3 acceleration, then -1 for 2/3 s.
  np.testing.assert_allclose(
    exact_step(0., 1., -2., 1., 1., 0., 1.), [-1/18, -2/3]
  )
  # Positive motion with tau=3, F=1, b=2: v=1-exp(-2t).
  np.testing.assert_allclose(
    exact_step(0., 0., 3., 1., 1., 2., 1.),
    [1-(1-np.exp(-2))/2, 1-np.exp(-2)],
  )


def test_analytic_reference_is_subdivision_invariant():
  for damping in (0., 2.):
    expected = exact_step(0., 0.5, -6.008, 0.0008, 6., damping, 0.1)
    q, v = 0., 0.5
    for _ in range(1000):
      q, v = exact_step(q, v, -6.008, 0.0008, 6., damping, 0.0001)
    np.testing.assert_allclose([q, v], expected, atol=1e-10)


def test_benchmark_default_friction_reproduces_slip_and_clamping():
  result = run_case(0.0008, 1., 0., 1., "static", 0.9, 0.02, 0.00125)
  assert result["static_speed_rad_s"] > 0.1
  assert result["finite"] and not result["warnings"]
  model = hinge(0.0008, 1., 0., 0.99999, 0.02, 0.00125)
  # mjModel stores requested solimp; the runtime solver clamps its impedance.
  assert model.dof_solimp[0, 0] == 0.99999
