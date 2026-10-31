# PACE offline audit and baseline replay

This package provides PACE audit, replay, diagnostics, and a restricted
[first friction fit](FITTING.md). The original commands remain non-optimizing.

The first hardware-to-simulation baseline is recorded in
[`BASELINE_RUN04.md`](BASELINE_RUN04.md).
The identification-stage status through 2026-10-15 is recorded in
[`STAGE_RECORD_2026-10-15.md`](STAGE_RECORD_2026-10-15.md).

`audit` streams the CSV and checks device sequence/time continuity, resets,
non-finite values, protection and saturation flags, sweep completeness, actual
motion, and the firmware solver-coordinate mapping. A discontinuity always
starts a new segment; samples on opposite sides are never joined.

`replay` first requires a passing audit. It fixes the MuJoCo base, disables
robot contacts and all six policy/PD actuators, leaves both wheel joints free,
and applies the logged CAN3/CAN2 total commands directly as generalized motor
torques. The model's two gas-spring tendon actuators remain at their original
constant 300 N. It does not add the firmware's `370.1 * L` compensation again.

The mapping source is the logged solver coordinates. Policy angles are used
only as a hard cross-check and are not the source of MuJoCo qpos.

The replay runs each complete sweep once and then repeats it as independently
initialized 5 s windows. Outputs are new JSON, text, CSV, and selected PNG
files in a newly created directory. Existing logs and models are never
overwritten. The `replay` command does not invoke a parameter optimizer.

`diagnose` selects reproducible early/middle/late starts in every sweep and
reports 0.25/0.5/1 s error growth from one shared initialization per start. It
also compares the original, half, and quarter simulation timesteps on three
ordinary windows and the workpoint-29 failure segment. MuJoCo warnings,
non-finite state, automatic time resets, extreme velocity/acceleration,
closure SVD, loop gap, and passive-limit distance are recorded. Failed
trajectories are kept separate from successful metrics.

The command snapshots are logged at 100 Hz and replayed with zero-order hold.
This cannot reconstruct the intervening 200 Hz controller updates or exact CAN
scheduling. Logged motor feedback may be current-derived, so it is not treated
as an independent torque sensor.

`python -m pace_id.initial_balance` reuses ten saved initial states from
`diagnose`, streams their first 100 ms from the source CSV, and projects force
components through the closed-chain tangent basis onto CAN3/CAN2 and the free
wheel. It checks gas-spring virtual work, separates gravity, velocity bias,
passive forces, friction and constraints, and compares first-step acceleration
with an ideal-closure calculation. It never fits parameters. From this
`mujoco390` directory, run the following single-line command with new output:

```powershell
python -m pace_id.initial_balance --diagnosis "D:/path/diagnose_run01" --csv "D:/path/pace.csv" --xml "mjcf/scene.xml" --out "D:/path/balance_run01"
```

`python -m pace_id.direction_residual` compares matched opposite-motion blocks
within the same sweep, plus one near-static block per sweep. It reports
direction-dependent and direction-independent projected residuals without
estimating friction parameters. Strict matching can leave entire excitation
directions uncovered; these absences are part of the result. Velocity spread
is a screening proxy, not a bound on acceleration. Selection can be cached
with `--select-only` and reused with `--selection path/to/selection.json`.

```powershell
python -m pace_id.direction_residual --csv "D:/path/pace.csv" --xml "mjcf/scene.xml" --out "D:/path/direction_run01"
```

`python -m pace_id.integral_residual` handles nonuniform length-sweep motion
using projected impulse balance over matched 0.2 s windows. It includes
endpoint momentum, varying reduced mass and closure curvature without
differentiating measured speed. It assumes zero unrecorded wheel velocity;
residuals are diagnostic averages, not identified friction coefficients.
`--select-only` and `--selection` support cached selection without replay.

```powershell
python -m pace_id.integral_residual --csv "D:/path/pace.csv" --xml "mjcf/scene.xml" --out "D:/path/integral_run01"
```

```powershell
python -m pace_id audit --csv "D:/path/pace.csv" --out "D:/path/audit_run01"
python -m pace_id replay --csv "D:/path/pace.csv" --out "D:/path/replay_run01"
python -m pace_id diagnose --csv "D:/path/pace.csv" --baseline "D:/path/replay_run01" --out "D:/path/diagnose_run01"
```
