"""Replay PACE torque logs in a fixed-base MuJoCo model.

This first-stage tool verifies joint mapping, signs, and baseline trajectory
error before any parameter optimizer is introduced.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
import tempfile

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
JOINT_NAMES = {
  "front": "robot/Left_front_joint",
  "rear": "robot/Left_rear_joint",
}
ACTUATOR_NAMES = (
  "robot/Left_front_joint",
  "robot/Left_rear_joint",
  "robot/Right_front_joint",
  "robot/Right_rear_joint",
  "robot/Left_Wheel_joint",
  "robot/Right_Wheel_joint",
)


@dataclass
class PaceLog:
  time: np.ndarray
  active_motor: np.ndarray
  front_command: np.ndarray
  front_angle: np.ndarray
  front_speed: np.ndarray
  rear_command: np.ndarray
  rear_angle: np.ndarray
  rear_speed: np.ndarray


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--csv", type=Path, required=True)
  parser.add_argument("--xml", type=Path, default=ROOT / "mjcf/scene.xml")
  parser.add_argument("--output", type=Path, default=Path("pace_replay.csv"))
  parser.add_argument("--timestep", type=float, default=0.0025)
  parser.add_argument("--gas-force", type=float, default=300.0)
  parser.add_argument("--damping", type=float, default=0.03)
  parser.add_argument("--frictionloss", type=float, default=0.015)
  parser.add_argument("--armature", type=float, default=0.0)
  parser.add_argument("--torque-scale", type=float, default=1.0)
  parser.add_argument("--delay-ms", type=float, default=0.0)
  parser.add_argument("--front-sign", type=float, choices=(-1.0, 1.0), default=1.0)
  parser.add_argument("--rear-sign", type=float, choices=(-1.0, 1.0), default=1.0)
  parser.add_argument(
    "--front-torque-sign", type=float, choices=(-1.0, 1.0), default=1.0
  )
  parser.add_argument(
    "--rear-torque-sign", type=float, choices=(-1.0, 1.0), default=1.0
  )
  parser.add_argument("--front-offset", type=float, default=0.94)
  parser.add_argument("--rear-offset", type=float, default=0.24)
  parser.add_argument("--pose-ramp-s", type=float, default=2.0)
  parser.add_argument("--pose-settle-s", type=float, default=1.0)
  parser.add_argument("--pose-kp", type=float, default=100.0)
  parser.add_argument("--pose-kd", type=float, default=6.0)
  parser.add_argument("--pose-torque-limit", type=float, default=28.0)
  return parser.parse_args()


def read_log(path: Path) -> PaceLog:
  with path.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
  if not rows:
    raise ValueError(f"PACE CSV has no samples: {path}")

  def values(name: str) -> np.ndarray:
    return np.asarray([float(row[name]) for row in rows], dtype=np.float64)

  return PaceLog(
    time=values("device_time_us") * 1.0e-6,
    active_motor=values("active_motor").astype(np.int32),
    front_command=values("left_front_command_nm"),
    front_angle=values("left_front_angle_rad"),
    front_speed=values("left_front_speed_rad_s"),
    rear_command=values("left_back_command_nm"),
    rear_angle=values("left_back_angle_rad"),
    rear_speed=values("left_back_speed_rad_s"),
  )


def load_fixed_base_model(xml_path: Path, timestep: float) -> mujoco.MjModel:
  text = xml_path.read_text(encoding="utf-8")
  text, count = re.subn(
    r'<joint\s+name="robot/base_free_joint"\s+type="free"\s*/>', "", text
  )
  if count != 1:
    raise ValueError("Expected exactly one robot/base_free_joint in XML")
  text = re.sub(r"\s*<keyframe>.*?</keyframe>", "", text, flags=re.DOTALL)
  temporary_path: Path | None = None
  try:
    with tempfile.NamedTemporaryFile(
      mode="w", suffix=".xml", dir=xml_path.parent, delete=False, encoding="utf-8"
    ) as temporary:
      temporary.write(text)
      temporary_path = Path(temporary.name)
    model = mujoco.MjModel.from_xml_path(str(temporary_path))
  finally:
    if temporary_path is not None:
      temporary_path.unlink(missing_ok=True)
  model.opt.timestep = timestep
  return model


def object_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
  result = mujoco.mj_name2id(model, kind, name)
  if result < 0:
    raise KeyError(f"MuJoCo object not found: {name}")
  return result


def configure_model(model: mujoco.MjModel, args: argparse.Namespace) -> None:
  # The real identification run is performed with the base fixed and both
  # wheels suspended.  Removing the free joint alone leaves the XML base at
  # its normal low height, where the wheels still collide with the terrain.
  # Disable robot/terrain contacts so the leg follows the same free-space
  # boundary condition as the suspended hardware.
  robot_geoms = model.geom_bodyid != 0
  model.geom_contype[robot_geoms] = 0
  model.geom_conaffinity[robot_geoms] = 0
  for name in JOINT_NAMES.values():
    joint_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    dof_id = model.jnt_dofadr[joint_id]
    model.dof_damping[dof_id] = args.damping
    model.dof_frictionloss[dof_id] = args.frictionloss
    model.dof_armature[dof_id] = args.armature
  for name in ACTUATOR_NAMES:
    actuator_id = object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    model.actuator_gainprm[actuator_id] = 0.0
    model.actuator_biasprm[actuator_id] = 0.0


def phase_indices(active_motor: np.ndarray, motor_id: int) -> np.ndarray:
  indices = np.flatnonzero(active_motor == motor_id)
  if indices.size == 0:
    raise ValueError(f"CSV has no samples for active_motor={motor_id}")
  return indices


def initialize_closed_chain_pose(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  front_target: float,
  rear_target: float,
  args: argparse.Namespace,
) -> tuple[float, float, float]:
  """Move from the valid XML pose to the measured pose without breaking loops."""
  front_joint = object_id(
    model, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAMES["front"]
  )
  rear_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAMES["rear"])
  front_qpos = model.jnt_qposadr[front_joint]
  rear_qpos = model.jnt_qposadr[rear_joint]
  front_dof = model.jnt_dofadr[front_joint]
  rear_dof = model.jnt_dofadr[rear_joint]
  targets = np.asarray((front_target, rear_target))
  qpos_ids = np.asarray((front_qpos, rear_qpos))
  dof_ids = np.asarray((front_dof, rear_dof))
  starts = data.qpos[qpos_ids].copy()

  duration = args.pose_ramp_s + args.pose_settle_s
  while data.time < duration:
    if args.pose_ramp_s > 0.0:
      progress = min(data.time / args.pose_ramp_s, 1.0)
    else:
      progress = 1.0
    # Smooth ramp avoids injecting an impulse into the closed-chain constraints.
    progress = progress * progress * (3.0 - 2.0 * progress)
    references = starts + progress * (targets - starts)
    torque = (
      args.pose_kp * (references - data.qpos[qpos_ids])
      - args.pose_kd * data.qvel[dof_ids]
    )
    data.qfrc_applied[:] = 0.0
    data.qfrc_applied[dof_ids] = np.clip(
      torque, -args.pose_torque_limit, args.pose_torque_limit
    )
    mujoco.mj_step(model, data)

  data.qfrc_applied[:] = 0.0
  data.qvel[:] = 0.0
  data.time = 0.0
  mujoco.mj_forward(model, data)

  loop_errors = []
  for front_site, rear_site in (
    ("robot/Left_front_site1", "robot/Left_rear_site1"),
    ("robot/Left_front_site2", "robot/Left_rear_site2"),
  ):
    front_site_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, front_site)
    rear_site_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, rear_site)
    loop_errors.append(
      np.linalg.norm(data.site_xpos[front_site_id] - data.site_xpos[rear_site_id])
    )
  front_error, rear_error = data.qpos[qpos_ids] - targets
  return float(front_error), float(rear_error), float(max(loop_errors))


def replay_phase(
  model: mujoco.MjModel,
  log: PaceLog,
  indices: np.ndarray,
  args: argparse.Namespace,
) -> dict[str, np.ndarray | float | str]:
  data = mujoco.MjData(model)
  front_joint = object_id(
    model, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAMES["front"]
  )
  rear_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAMES["rear"])
  front_qpos = model.jnt_qposadr[front_joint]
  rear_qpos = model.jnt_qposadr[rear_joint]
  front_dof = model.jnt_dofadr[front_joint]
  rear_dof = model.jnt_dofadr[rear_joint]

  first = int(indices[0])
  real_front = args.front_sign * log.front_angle[indices] + args.front_offset
  real_rear = args.rear_sign * log.rear_angle[indices] + args.rear_offset

  for name in ("robot/Left_gas_spring_tendon", "robot/Right_gas_spring_tendon"):
    actuator_id = object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    data.ctrl[actuator_id] = args.gas_force
  mujoco.mj_forward(model, data)
  init_front_error, init_rear_error, init_loop_error = initialize_closed_chain_pose(
    model, data, real_front[0], real_rear[0], args
  )

  sample_time = log.time[indices] - log.time[first]
  command_front = (
    args.front_sign * args.front_torque_sign * log.front_command[indices]
  )
  command_rear = args.rear_sign * args.rear_torque_sign * log.rear_command[indices]
  delay_s = args.delay_ms * 1.0e-3
  sim_front = np.empty(indices.size)
  sim_rear = np.empty(indices.size)
  sim_front_speed = np.empty(indices.size)
  sim_rear_speed = np.empty(indices.size)

  command_index = 0
  for output_index, target_time in enumerate(sample_time):
    while data.time + 0.5 * model.opt.timestep < target_time:
      delayed_time = max(0.0, data.time - delay_s)
      while (
        command_index + 1 < sample_time.size
        and sample_time[command_index + 1] <= delayed_time
      ):
        command_index += 1
      data.qfrc_applied[:] = 0.0
      data.qfrc_applied[front_dof] = (
        args.torque_scale * command_front[command_index]
      )
      data.qfrc_applied[rear_dof] = args.torque_scale * command_rear[command_index]
      mujoco.mj_step(model, data)
    sim_front[output_index] = data.qpos[front_qpos]
    sim_rear[output_index] = data.qpos[rear_qpos]
    sim_front_speed[output_index] = data.qvel[front_dof]
    sim_rear_speed[output_index] = data.qvel[rear_dof]

  real_front_relative = real_front - real_front[0]
  real_rear_relative = real_rear - real_rear[0]
  sim_front_relative = sim_front - sim_front[0]
  sim_rear_relative = sim_rear - sim_rear[0]
  position_error = np.concatenate(
    (sim_front_relative - real_front_relative, sim_rear_relative - real_rear_relative)
  )
  velocity_error = np.concatenate(
    (
      sim_front_speed - args.front_sign * log.front_speed[indices],
      sim_rear_speed - args.rear_sign * log.rear_speed[indices],
    )
  )
  motor_name = "left_front" if log.active_motor[first] == 1 else "left_rear"
  return {
    "motor": motor_name,
    "time": sample_time,
    "real_front": real_front_relative,
    "sim_front": sim_front_relative,
    "real_rear": real_rear_relative,
    "sim_rear": sim_rear_relative,
    "real_front_speed": args.front_sign * log.front_speed[indices],
    "sim_front_speed": sim_front_speed,
    "real_rear_speed": args.rear_sign * log.rear_speed[indices],
    "sim_rear_speed": sim_rear_speed,
    "front_command": command_front,
    "rear_command": command_rear,
    "init_front_error": init_front_error,
    "init_rear_error": init_rear_error,
    "init_loop_error": init_loop_error,
    "position_rmse": float(np.sqrt(np.mean(position_error**2))),
    "velocity_rmse": float(np.sqrt(np.mean(velocity_error**2))),
  }


def write_results(path: Path, phases: list[dict[str, object]]) -> None:
  columns = (
    "phase", "time_s", "front_command_nm", "rear_command_nm",
    "real_front_relative_rad", "sim_front_relative_rad",
    "real_rear_relative_rad", "sim_rear_relative_rad",
    "real_front_speed_rad_s", "sim_front_speed_rad_s",
    "real_rear_speed_rad_s", "sim_rear_speed_rad_s",
  )
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(columns)
    for phase in phases:
      count = len(phase["time"])
      for index in range(count):
        writer.writerow(
          [
            phase["motor"], phase["time"][index],
            phase["front_command"][index], phase["rear_command"][index],
            phase["real_front"][index], phase["sim_front"][index],
            phase["real_rear"][index], phase["sim_rear"][index],
            phase["real_front_speed"][index], phase["sim_front_speed"][index],
            phase["real_rear_speed"][index], phase["sim_rear_speed"][index],
          ]
        )


def main() -> None:
  args = parse_args()
  log = read_log(args.csv)
  model = load_fixed_base_model(args.xml.resolve(), args.timestep)
  configure_model(model, args)
  phases = [
    replay_phase(model, log, phase_indices(log.active_motor, motor_id), args)
    for motor_id in (1, 2)
  ]
  write_results(args.output, phases)
  for phase in phases:
    print(
      f"{phase['motor']}: initial pose error front/rear="
      f"{phase['init_front_error']:.6f}/{phase['init_rear_error']:.6f} rad, "
      f"max loop residual={phase['init_loop_error']:.3e} m"
    )
    print(
      f"{phase['motor']}: position RMSE={phase['position_rmse']:.6f} rad, "
      f"velocity RMSE={phase['velocity_rmse']:.6f} rad/s"
    )
  print(f"aligned trajectories: {args.output.resolve()}")


if __name__ == "__main__":
  main()
