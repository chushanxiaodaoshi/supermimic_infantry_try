"""Coarse-fit common MuJoCo joint friction and damping to PACE logs."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import replay_pace as replay


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--csv-pattern", required=True)
  parser.add_argument("--xml", type=Path, default=ROOT / "mjcf/scene.xml")
  parser.add_argument(
    "--frictionloss",
    type=float,
    nargs="+",
    default=(0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4),
  )
  parser.add_argument(
    "--damping",
    type=float,
    nargs="+",
    default=(0.05, 0.1, 0.2, 0.3, 0.5, 0.8),
  )
  parser.add_argument("--torque-scale", type=float, nargs="+", default=(1.0,))
  parser.add_argument("--delay-ms", type=float, nargs="+", default=(0.0,))
  parser.add_argument("--armature", type=float, nargs="+", default=(0.0,))
  parser.add_argument("--velocity-weight", type=float, default=0.1)
  return parser.parse_args()


def replay_args(
  damping: float,
  frictionloss: float,
  torque_scale: float,
  delay_ms: float,
  armature: float,
) -> SimpleNamespace:
  return SimpleNamespace(
    timestep=0.0025,
    gas_force=300.0,
    damping=damping,
    frictionloss=frictionloss,
    armature=armature,
    torque_scale=torque_scale,
    delay_ms=delay_ms,
    front_sign=1.0,
    rear_sign=1.0,
    front_torque_sign=1.0,
    rear_torque_sign=1.0,
    front_offset=0.94,
    rear_offset=0.24,
    pose_ramp_s=2.0,
    pose_settle_s=1.0,
    pose_kp=100.0,
    pose_kd=6.0,
    pose_torque_limit=28.0,
  )


def main() -> None:
  cli = parse_args()
  pattern = Path(cli.csv_pattern)
  paths = sorted(pattern.parent.glob(pattern.name))
  if not paths:
    raise FileNotFoundError(f"No CSV files match: {cli.csv_pattern}")
  logs = [(path, replay.read_log(path)) for path in paths]
  print(f"PACE logs: {len(logs)}")
  for path, _ in logs:
    print(f"  {path}")

  results = []
  total = (
    len(cli.frictionloss)
    * len(cli.damping)
    * len(cli.torque_scale)
    * len(cli.delay_ms)
    * len(cli.armature)
  )
  completed = 0
  for frictionloss in cli.frictionloss:
    for damping in cli.damping:
      for torque_scale in cli.torque_scale:
        for delay_ms in cli.delay_ms:
          for armature in cli.armature:
            args = replay_args(
              damping, frictionloss, torque_scale, delay_ms, armature
            )
            model = replay.load_fixed_base_model(cli.xml.resolve(), args.timestep)
            replay.configure_model(model, args)
            position_errors = []
            velocity_errors = []
            for _, log in logs:
              for motor_id in (1, 2):
                phase = replay.replay_phase(
                  model, log, replay.phase_indices(log.active_motor, motor_id), args
                )
                position_errors.append(phase["position_rmse"])
                velocity_errors.append(phase["velocity_rmse"])
            position_rmse = sum(position_errors) / len(position_errors)
            velocity_rmse = sum(velocity_errors) / len(velocity_errors)
            score = position_rmse + cli.velocity_weight * velocity_rmse
            results.append(
              (
                score, position_rmse, velocity_rmse, frictionloss, damping,
                torque_scale, delay_ms, armature,
              )
            )
            completed += 1
            print(
              f"[{completed:02d}/{total:02d}] friction={frictionloss:.3f}, "
              f"damping={damping:.3f}, torque_scale={torque_scale:.3f}, "
              f"delay_ms={delay_ms:.1f}, armature={armature:.5f}: "
              f"pos={position_rmse:.4f}, vel={velocity_rmse:.4f}, "
              f"score={score:.4f}"
            )

  print("\nBest candidates:")
  for (
    score, position, velocity, frictionloss, damping, torque_scale, delay_ms,
    armature,
  ) in sorted(results)[:10]:
    print(
      f"frictionloss={frictionloss:.3f}, damping={damping:.3f}, "
      f"torque_scale={torque_scale:.3f}, delay_ms={delay_ms:.1f}, "
      f"armature={armature:.5f}, "
      f"position={position:.4f}, velocity={velocity:.4f}, score={score:.4f}"
    )


if __name__ == "__main__":
  main()
