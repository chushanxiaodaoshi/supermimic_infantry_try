# First friction fit

`python -m pace_id.fit` performs a bounded CMA-ES baseline, not a complete
physical calibration. Dependencies: the existing NumPy/MuJoCo environment
plus `cma`. The tool never installs packages. Run from `mujoco390`.

Only native `dof_frictionloss` for CAN3 and CAN2, and a shared active-joint
`dof_damping`, are replaced in memory. Bounds are [0, 6] Nm and [0, 2]
Nms/rad. Passive-joint friction, geometry, inertia, force scale, spring,
delay, equality solver settings, and the physical-ID mapping stay fixed.
These are equivalent parameters; the dry-friction term includes native
stiction and does not separately identify static versus kinetic friction.

Preparation audits the CSV, solves and caches closed-chain initial states,
and records early/middle/late 1 s windows. Device timestamps define exact
observation and zero-order-held torque boundaries. No 200 Hz reconstruction
or duplicate spring compensation is performed. Wheels are free but their
unrecorded initial velocities are assumed zero, as in the earlier diagnostics.

Validation workpoints are fixed before fitting: 1, 8, 15, 22, 25, 28.
Both sweeps of workpoint 29 are reserved for stress evaluation, not fitting.
All other workpoints train one global parameter set. Windows are not randomly
split. The holdout is for parameter fitting; previous exploratory diagnostics
have already examined this dataset, so it is not a pristine external test.

Training averages Huber errors over 0.1/0.25/0.5 s prefixes and then over
windows. Angle errors wrap at pi, scale 0.05 rad; velocity errors scale
0.5 rad/s and weight 0.1. Initial samples are excluded. L/phi are independent
reporting checks, not extra loss terms on the same coordinates. Numerical
failures receive a fixed 1e6 penalty and are listed separately. Successful
RMSE alone must not hide failures. Frozen-position and motion-span controls
flag the possible degenerate solution of making the leg stick. Span ratios
are not frequency responses or SNR estimates.

Workflow (each output directory must be new):

```powershell
python -m pace_id.fit --csv "D:/path/pace.csv" --out "D:/path/fit_prepare01" --prepare-only
python -m pace_id.fit --csv "D:/path/pace.csv" --prepared "D:/path/fit_prepare01" --out "D:/path/fit_pilot01" --pilot --generations 2
python -m pace_id.fit --csv "D:/path/pace.csv" --prepared "D:/path/fit_prepare01" --out "D:/path/fit_run01" --generations 30 --workers 4 --timestep 0.00125
```

Pilot uses only six training windows, not the validation set. Final evaluation
reports 1 s generalization, original parameters, candidate parameters, frozen
positions, and original/half/quarter timestep checks on representative training
windows. Results are JSON/NPZ plus `SUMMARY.md`; candidates never overwrite XML.
Optimizer checkpoints contain all evaluated parameters, failures and scores.
A budget stop is not proof of convergence or unique identifiability.
`assessment.json` compares with the frozen-state control and flags search
boundaries. Loop-gap/limit extrema in window records refer to the entire
replay (1 s in final evaluation), not each shorter reporting prefix. The
workpoint-29 short stress windows do not replace a replay of its original
2.975 s failure trajectory. No candidate is automatically deployed.

Algorithm reference: https://cma-es.github.io/apidocs-pycma/

## Native-friction implementation diagnosis

`python -m pace_id.fit_diagnosis --prepared PREPARED --fit FIT --out NEW_OUT`
reuses the cached windows and fitted candidate for eight representative cases.
It probes actual native friction forces, caps, regularization, projected force
balance and static counterfactuals. Only in-memory active-friction `solimp[0]`
changes (0.9/0.99/0.999); reference decay, physical parameters and equalities
stay fixed. This is a numerical sensitivity check, not a second optimization.
It also checks timestep sensitivity and possible loss of real motion.

Native frictionloss is a force cap, not guaranteed sliding torque. Its soft
constraint regularization can admit appreciable slip below that cap. A fitted
cap at a search bound is therefore not sufficient evidence of large hardware
friction. See the official MuJoCo documentation:
https://mujoco.readthedocs.io/en/stable/modeling.html#friction

## Synthetic numerical gate and second pilot

`friction_bench.py` compares native friction against the analytic solution of
a synthetic one-hinge Coulomb-plus-viscous system: static sub-threshold load,
sliding, coast-to-stop and reversal, both directions, two inertia scales,
two friction caps and two damping values. These are numerical tests, not
hardware identification. Settings are selected without measured replay loss.
The fixed tolerances are 0.001 rad position error, 0.02 rad/s speed error,
0.001 rad/s static slip and 0.001 rad coarse/fine position difference.
Warnings/nonfinite results and sampled positive friction power also reject
a profile. Power is sampled at 10 ms, not proven nonpositive at every substep.
The test lasts 0.1 s and cannot certify an entire closed-chain search domain.

Native joint-friction regularization is a force cap rather than an exact
Coulomb implementation. Hardening only impedance was insufficient for the
tested stop/reversal cases. Tested noslip settings also failed the same gate;
this is not proof that noslip always fails, nor confirmation of a particular
engine bug. Related upstream report (different engine version):
https://github.com/google-deepmind/mujoco/issues/2423

One passing profile on MuJoCo 3.9.0 uses active friction d0=0.9999,
timeconst=0.00025 s, steps 0.000078125/0.0000390625 s. These are numerical
settings, not fitted hardware parameters. `--friction-benchmark` imports
the profile into the fitting process, changing only active `dof_solimp[:,0]`
and `dof_solref[:,0]` in memory. Equalities and all physical parameters remain
fixed except the same three explicitly optimized friction/damping values.
Original, numerics-only, candidate and frozen-state results are separated.
Benchmark hash and applied settings are recorded with the fit configuration.

```powershell
python -m pace_id.friction_bench --out NEW_BENCH --impedances 0.9999 --timeconsts 0.00025 --timesteps 0.000078125 0.0000390625
python -m pace_id.fit_numerics_check --prepared PREPARED --pilot PILOT --benchmark BENCH/benchmark.json --out NEW_CHECK
```

The second pilot uses six fixed training windows and two CMA-ES generations;
its parameters are not accepted estimates. Its original step still produced
0.030 rad coarse/half-step discrepancy on one window despite the synthetic
pass. The closed-chain gate therefore checks three fixed parameter vectors
(initial, pilot, upper corner), six training windows each, and three steps.
It reports per-pair angle/speed convergence and failures separately. The
thresholds are 0.001 rad and 0.02 rad/s trajectory RMS difference. It does not
optimize parameters, consult held-out loss, or validate the old WP29 failure.

The first full closed-chain gate failed even at smaller steps: an upper-corner
case showed 42.4 mm loop opening without warnings. Tightening only Newton
termination/line-search settings (iterations=200, ls_iterations=100,
tolerance=1e-10, ls_tolerance=0.001) resolved this in the tested subset.
With that profile, all 18 window/parameter combinations passed both timestep
comparisons (78.125/39.0625/19.53125 microseconds): maximum angle RMS
difference 0.000167 rad, speed RMS difference 0.00821 rad/s, and maximum
sampled loop gap 0.0752 mm. This supports a numerical-solver sensitivity
diagnosis in these cases, not a unique cause or universal stability proof.

Full fitting with modified friction numerics requires `--numerics-check`
pointing to the passing, unfocused check JSON. It verifies the data manifest,
benchmark hash, engine version and timestep, and propagates the checked
solver settings to workers and candidate evaluations. The original baseline
keeps the original solver/friction settings; numerics-only uses the new
numerics with unchanged physical parameters. No equality parameters change.

```powershell
python -m pace_id.fit_numerics_check --prepared PREPARED --pilot PILOT --benchmark BENCH/benchmark.json --out NEW_CHECK --timestep 0.000078125 --strict-solver
python -m pace_id.fit --csv CSV --prepared PREPARED --out NEW_FIT --generations 30 --workers 4 --timestep 0.000078125 --friction-benchmark BENCH/benchmark.json --numerics-check CHECK/check.json
```
