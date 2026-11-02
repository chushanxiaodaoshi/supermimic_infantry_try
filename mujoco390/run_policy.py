"""Play a wheel-legged ONNX policy in independent MuJoCo 3.9.0."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from collections import deque
import ctypes
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
import time

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
from pynput import keyboard


ROOT = Path(__file__).resolve().parent

# Debug settings: edit these two values directly when needed.
DEBUG_POLICY = True
DEBUG_PRINT_INTERVAL_S = 1.0
JUMP_DURATION_S = 0.55
JUMP_TRIGGER_DISTANCE_M = 0.7
SLOW_LINEAR_SPEED = 1.0
SLOW_YAW_SPEED = 2.0
FAST_LINEAR_SPEED = 2.0
FAST_YAW_SPEED = 4.0
NUMPAD_SCAN_CODES = {
  0x4F: "1", 0x50: "2", 0x51: "3", 0x4B: "4",
  0x4C: "5", 0x4D: "6", 0x47: "7", 0x48: "8",
  0x49: "9",
}
LEG_JOINTS = (
  "Left_front_joint",
  "Left_rear_joint",
  "Right_front_joint",
  "Right_rear_joint",
)
POLICY_JOINTS = (
  "Left_front_joint",
  "Left_rear_joint",
  "Left_Wheel_joint",
  "Right_front_joint",
  "Right_rear_joint",
  "Right_Wheel_joint",
)
ACTION_TARGETS = (
  "Left_rear_joint",
  "Left_front_joint",
  "Left_Wheel_joint",
  "Right_rear_joint",
  "Right_front_joint",
  "Right_Wheel_joint",
)
ACTION_SCALES = np.array((0.5, 0.5, 10.0, 0.5, 0.5, 10.0))
POLICY_TORQUE_LIMITS = np.array((28.0, 28.0, 4.0, 28.0, 28.0, 4.0))

# Explicit stand-up Teacher reset.  Do not inherit this pose from the scene
# keyframe or initial_state.npz: the experiment requires both legs folded
# behind the chassis before the six policy actuators are enabled.
STANDUP_INITIAL_ROOT_POSITION = (0.0, 0.0, 0.20)
STANDUP_INITIAL_ROOT_QUATERNION_WXYZ = (1.0, 0.0, 0.0, 0.0)
STANDUP_INITIAL_JOINT_POSITIONS = {
  "Left_rear_joint": 0.0,
  "Left_rear_child1_joint": 0.0,
  "Left_Wheel_joint": 0.0,
  "Right_rear_joint": 0.0,
  "Right_rear_child1_joint": 0.0,
  "Right_Wheel_joint": 0.0,
  "Left_front_joint": 0.0,
  "Left_front_child1_joint": 0.0,
  "Left_front_child2_joint": 0.0,
  "Left_front_child3_joint": 0.0,
  "Right_front_joint": 0.0,
  "Right_front_child1_joint": 0.0,
  "Right_front_child2_joint": 0.0,
  "Right_front_joint3_joint": 0.0,
}
STANDUP_DISABLED_DURATION_S = 1.0
STANDUP_STABLE_DURATION_S = 0.20

jump_onnx = "./onnx/jump.onnx"
onnx = "./onnx/rough.onnx"

def configure_motor_torque_limits(model, actuator_ids: np.ndarray) -> None:
  """Enforce current motor limits even when loading an older exported XML."""
  model.actuator_forcelimited[actuator_ids] = True
  model.actuator_forcerange[actuator_ids] = np.column_stack(
    (-POLICY_TORQUE_LIMITS, POLICY_TORQUE_LIMITS)
  )


def sync_torque_controls(viewer, model, data, actuator_ids: np.ndarray) -> None:
  """Show generated torques in Control without changing the PD simulation.

  The passive viewer does not step physics. Its Control sliders are monitoring
  values while a policy is running; slider edits are deliberately not applied.
  Restore servo targets even if viewer synchronization fails.
  """
  targets = data.ctrl[actuator_ids].copy()
  control_ranges = model.actuator_ctrlrange[actuator_ids].copy()
  control_limited = model.actuator_ctrllimited[actuator_ids].copy()
  try:
    # Limits here are GUI metadata only: no physics step uses these values.
    model.actuator_ctrlrange[actuator_ids] = model.actuator_forcerange[actuator_ids]
    model.actuator_ctrllimited[actuator_ids] = True
    data.ctrl[actuator_ids] = data.actuator_force[actuator_ids]
    viewer.sync()
  finally:
    data.ctrl[actuator_ids] = targets
    model.actuator_ctrlrange[actuator_ids] = control_ranges
    model.actuator_ctrllimited[actuator_ids] = control_limited


def print_rm_terrain_summary(model: mujoco.MjModel) -> None:
  """Report loaded custom course lanes once, without creating duplicate geoms."""
  markers = {
    "rm_double_step_base": "二级台阶：20cm + 10cm，上层入口凸坎5cm（顶高35cm）",
    "standalone_ramp_17deg_35cm": "飞坡：17度，出口离地35cm",
    "classic_slope_up_00": "连续坡隆起（上坑）",
    "classic_slope_down_00": "连续坡凹坑（下坑）",
    "classic_waves_00": "波浪起伏路段",
    "classic_stairs_up_12": "上阶梯金字塔",
    "classic_stairs_down_12": "下阶梯金字塔",
    "classic_discrete_00": "离散障碍路段",
    "classic_rough_00": "训练随机起伏路段",
  }
  for name, description in markers.items():
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id >= 0:
      x, y = model.geom_pos[geom_id, :2]
      print(f"[RM terrain] {description}; geom x={x:.2f}, y={y:.2f}m")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--onnx", "--rough-onnx", type=Path, default=onnx)
  parser.add_argument("--jump-onnx", type=Path, default=jump_onnx)
  parser.add_argument("--xml", type=Path, default=ROOT / "mjcf/scene.xml")
  parser.add_argument("--state", type=Path, default=None)
  parser.add_argument("--vx", type=float, default=0.0)
  parser.add_argument("--yaw", type=float, default=0.0)
  parser.add_argument("--height", type=float, default=0.2)
  parser.add_argument("--height-range", type=float, nargs=2, default=(0.1, 0.30))
  parser.add_argument("--timestep", type=float, default=0.0025)
  parser.add_argument("--control-decimation", type=int, default=4)
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--duration", type=float, default=0.0)
  parser.add_argument(
    "--standup-teacher-output",
    type=Path,
    default=None,
    help=(
      "Record an automatic disabled-to-rough stand-up trajectory to NPZ. "
      "The explicit stand-up reset is used instead of --state."
    ),
  )
  parser.add_argument(
    "--standup-disabled-duration",
    type=float,
    default=STANDUP_DISABLED_DURATION_S,
    help="Seconds to settle with policy actuators disabled before rough starts.",
  )
  parser.add_argument(
    "--standup-stable-duration",
    type=float,
    default=STANDUP_STABLE_DURATION_S,
    help="Continuous stable time required to finish the stand-up trajectory.",
  )
  parser.add_argument(
    "--debug",
    action=argparse.BooleanOptionalAction,
    default=DEBUG_POLICY,
  )
  parser.add_argument(
    "--debug-interval",
    type=float,
    default=DEBUG_PRINT_INTERVAL_S,
  )
  return parser.parse_args()


def _all_names(model: mujoco.MjModel, obj: mujoco.mjtObj, count: int) -> list[str]:
  names = []
  for index in range(count):
    name = mujoco.mj_id2name(model, obj, index)
    names.append(name or "")
  return names


def _suffix_id(
  model: mujoco.MjModel,
  obj: mujoco.mjtObj,
  count: int,
  suffix: str,
) -> int:
  matches = [
    index
    for index, name in enumerate(_all_names(model, obj, count))
    if name == suffix or name.endswith("/" + suffix)
  ]
  if len(matches) != 1:
    raise KeyError(f"Expected one {obj.name} ending in {suffix!r}, got {matches}")
  return matches[0]


def set_explicit_standup_initial_pose(
  model: mujoco.MjModel,
  data: mujoco.MjData,
) -> None:
  """Apply the named legs-behind reset used by the stand-up Teacher."""
  data.qpos[:] = model.qpos0
  data.qvel[:] = 0.0
  free_joint = _suffix_id(
    model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt, "base_free_joint"
  )
  if model.jnt_type[free_joint] != mujoco.mjtJoint.mjJNT_FREE:
    raise ValueError("base_free_joint must be a free joint")
  root_address = model.jnt_qposadr[free_joint]
  data.qpos[root_address : root_address + 3] = STANDUP_INITIAL_ROOT_POSITION
  data.qpos[root_address + 3 : root_address + 7] = (
    STANDUP_INITIAL_ROOT_QUATERNION_WXYZ
  )
  for name, position in STANDUP_INITIAL_JOINT_POSITIONS.items():
    joint_id = _suffix_id(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt, name)
    if model.jnt_type[joint_id] not in (
      mujoco.mjtJoint.mjJNT_HINGE,
      mujoco.mjtJoint.mjJNT_SLIDE,
    ):
      raise ValueError(f"stand-up joint {name} must have one qpos value")
    data.qpos[model.jnt_qposadr[joint_id]] = position
  mujoco.mj_forward(model, data)


def make_shift_state_reader() -> Callable[[], bool]:
  """Poll current Shift state outside the hook, independent of key order."""
  get_state = ctypes.windll.user32.GetAsyncKeyState
  get_state.argtypes = [ctypes.c_int]
  get_state.restype = ctypes.c_short
  return lambda: bool(get_state(0xA0) & 0x8000 or get_state(0xA1) & 0x8000)


@dataclass
class KeyboardState:
  vx_default: float
  yaw_default: float
  height: float
  height_range: tuple[float, float]
  pressed: set[str] = field(default_factory=set)
  torque_enabled: bool = True
  jump_requested: bool = False
  shift_state_reader: Callable[[], bool] | None = None
  lock: Lock = field(default_factory=Lock)

  def on_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
    char = getattr(key, "char", None)
    if key == keyboard.Key.space or char == " ":
      with self.lock:
        if "space" not in self.pressed:
          self.jump_requested = True
          self.pressed.add("space")
      return
    if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
      with self.lock:
        self.pressed.add(f"shift:{key}")
      return
    vk = getattr(key, "vk", None)
    if vk not in range(0x61, 0x6A):
      return  # Never accept the main keyboard's digit row or digit characters.
    digit = str(vk - 0x60)
    with self.lock:
      if digit in self.pressed:
        return
      self.pressed.add(digit)
      if digit == "1":
        self.torque_enabled = False
        print("[control] joint torque OFF; gas springs remain enabled")
      elif digit == "2":
        self.torque_enabled = True
        print("[control] policy control restored")

  def on_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
    char = getattr(key, "char", None)
    if key == keyboard.Key.space or char == " ":
      with self.lock:
        self.pressed.discard("space")
      return
    if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
      with self.lock:
        self.pressed.discard(f"shift:{key}")
      return
    vk = getattr(key, "vk", None)
    if vk in range(0x61, 0x6A):
      with self.lock:
        self.pressed.discard(str(vk - 0x60))

  def on_win32_event(self, msg, event):
    """Normalize physical keypad keys, including Shift/NumLock navigation VKs.

    Returning False filters only pynput callbacks, not delivery to other apps.
    Dedicated navigation keys have the extended flag and are not keypad keys.
    """
    if msg not in (0x100, 0x101, 0x104, 0x105):
      return True
    pressed = msg in (0x100, 0x104)
    callback = self.on_press if pressed else self.on_release
    if event.vkCode in (0x10, 0xA0, 0xA1):
      # Physical Shift uses side-specific VKs. Generic/injected transitions
      # around Shift+NumLock keypad navigation must not clear held Shift.
      if event.vkCode != 0x10 and not event.flags & 0x12:
        key = (
          keyboard.Key.shift_r if event.scanCode == 0x36 else keyboard.Key.shift_l
        )
        callback(key)
      return False
    digit = NUMPAD_SCAN_CODES.get(event.scanCode)
    if digit is not None and not event.flags & 1:
      callback(keyboard.KeyCode.from_vk(0x60 + int(digit)))
      return False
    return True

  def command(
    self, dt: float
  ) -> tuple[float, float, float, bool]:
    with self.lock:
      pressed = self.pressed.copy()
      enabled = self.torque_enabled
      height_direction = float("7" in pressed) - float("9" in pressed)
      requested_height = self.height + 0.15 * dt * height_direction
      self.height = float(np.clip(requested_height, *self.height_range))
    fast = (
      self.shift_state_reader() if self.shift_state_reader is not None
      else any(key.startswith("shift:") for key in pressed)
    )
    linear_speed = FAST_LINEAR_SPEED if fast else SLOW_LINEAR_SPEED
    yaw_speed = FAST_YAW_SPEED if fast else SLOW_YAW_SPEED
    vx_direction = float("8" in pressed) - float("5" in pressed)
    vx = self.vx_default + linear_speed * vx_direction
    yaw = self.yaw_default + yaw_speed * (float("4" in pressed) - float("6" in pressed))
    return vx, yaw, self.height, enabled

  def consume_jump_request(self) -> bool:
    with self.lock:
      requested = self.jump_requested
      self.jump_requested = False
    return requested


def load_policy_session(policy: Path):
  """Validate each policy independently before starting playback."""
  session = ort.InferenceSession(str(policy), providers=["CPUExecutionProvider"])
  if len(session.get_inputs()) != 1:
    raise ValueError(f"{policy}: expected one ONNX observation input")
  info = session.get_inputs()[0]
  if len(info.shape) != 2 or info.type != "tensor(float)":
    raise ValueError(f"{policy}: expected float32 input [batch, 150]")
  batch = info.shape[0]
  if info.shape[-1] != 150 or not (
    batch in (1, None) or isinstance(batch, str)
  ):
    raise ValueError(f"{policy}: only [1,150] policies supported, got {info.shape}")
  outputs = session.get_outputs()
  if len(outputs) != 1 or (
    len(outputs[0].shape) != 2
    or outputs[0].shape[-1] != 6
    or outputs[0].type != "tensor(float)"
  ):
    raise ValueError(f"{policy}: expected one 6-D action output")
  return session


class WheellegRuntime:
  def __init__(
    self,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    policy: Path,
    debug_interval: float | None = None,
    jump_policy: Path | None = None,
  ):
    self.model = model
    self.data = data
    self.sessions = {"rough": load_policy_session(policy)}
    if jump_policy is not None:
      self.sessions["jump"] = load_policy_session(jump_policy)
    self.active_policy = "rough"
    self.jump_prepare = False
    self.jump_until = None
    self.session = self.sessions["rough"]
    self.input_name = self.session.get_inputs()[0].name
    self.expected_obs_dim = 150
    self.current_obs_dim = 25
    print("[policy] observation width: 150 = current 25 + history 125")
    self.base_id = _suffix_id(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody, "base_link")
    try:
      self.forward_sensor_id = _suffix_id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, model.nsensor,
        "forward_rangefinder_sensor",
      )
    except KeyError:
      self.forward_sensor_id = None
    if self.forward_sensor_id is not None:
      sensor_id = self.forward_sensor_id
      if (model.sensor_type[sensor_id] != mujoco.mjtSensor.mjSENS_RANGEFINDER
          or model.sensor_dim[sensor_id] != 1):
        raise ValueError("forward_rangefinder_sensor must be a scalar rangefinder")
    self.joint_ids = {
      name: _suffix_id(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt, name)
      for name in set((*LEG_JOINTS, *POLICY_JOINTS))
    }
    self.leg_site_ids = {
      name: _suffix_id(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite, name)
      for name in (
        "Left_virtual_hip_site",
        "Left_wheel_axis_site",
        "Right_virtual_hip_site",
        "Right_wheel_axis_site",
      )
    }
    self.loop_site_pairs = np.array(
      [
        [
          _suffix_id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            model.nsite,
            f"{side}_{link}_site{index}",
          )
          for link in ("front", "rear")
        ]
        for side in ("Left", "Right")
        for index in (1, 2)
      ],
      dtype=np.int32,
    )
    self.actuator_ids = {
      name: _suffix_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu, name)
      for name in ACTION_TARGETS
    }
    self.gas_ids = [
      _suffix_id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        model.nu,
        name,
      )
      for name in ("Left_gas_spring_tendon", "Right_gas_spring_tendon")
    ]
    self.policy_actuator_ids = np.array(
      [self.actuator_ids[name] for name in ACTION_TARGETS],
      dtype=np.int32,
    )
    for name, actuator_id in self.actuator_ids.items():
      bias = model.actuator_biasprm[actuator_id]
      correct_feedback = bias[2] < 0.0 if "Wheel" in name else bias[1] < 0.0
      if not correct_feedback:
        raise ValueError(f"XML actuator {name} must be a PD/velocity servo")
    configure_motor_torque_limits(model, self.policy_actuator_ids)
    self.policy_gainprm = model.actuator_gainprm[
      self.policy_actuator_ids
    ].copy()
    self.policy_biasprm = model.actuator_biasprm[
      self.policy_actuator_ids
    ].copy()
    self.policy_actuators_enabled = True
    self.history: dict[str, deque[np.ndarray]] = {}
    self.last_action = np.zeros(6)
    self.debug_interval = debug_interval
    self.next_debug_time = 0.0

  def debug_policy_step(self, obs: np.ndarray, action: np.ndarray) -> None:
    """Print the effective 25-D observation and raw action at a low rate."""
    if self.debug_interval is None:
      return
    now = time.monotonic()
    if now < self.next_debug_time:
      return
    self.next_debug_time = now + self.debug_interval

    obs25 = obs[0, : self.current_obs_dim]
    obs_text = np.array2string(
      obs25,
      precision=4,
      suppress_small=True,
      separator=", ",
      max_line_width=240,
    )
    action_text = np.array2string(
      action,
      precision=4,
      suppress_small=True,
      separator=", ",
      max_line_width=120,
    )
    left_length, right_length = self._virtual_leg_lengths()
    wheel_ids = np.array(
      [
        self.actuator_ids["Left_Wheel_joint"],
        self.actuator_ids["Right_Wheel_joint"],
      ]
    )
    wheel_targets = self.data.ctrl[wheel_ids]
    wheel_torques = self.data.actuator_force[wheel_ids]
    loop_gaps_text = np.array2string(
      self._closed_chain_gaps_mm(), precision=3, separator=", "
    )
    print(
      f"[DEBUG {self.mode}] tof={self.forward_distance():.3f} m | "
      f"obs25={obs_text} | raw_action={action_text} | "
      f"wheel_vel_target={wheel_targets} rad/s | "
      f"wheel_torque={wheel_torques} N*m | "
      f"L_left={left_length:.4f} m | L_right={right_length:.4f} m | "
      f"loop_gap_mm[L1,L2,R1,R2]={loop_gaps_text}"
    )

  def _closed_chain_gaps_mm(self) -> np.ndarray:
    """Measure the four connect-site separations in millimeters."""
    front_pos = self.data.site_xpos[self.loop_site_pairs[:, 0]]
    rear_pos = self.data.site_xpos[self.loop_site_pairs[:, 1]]
    return np.linalg.norm(front_pos - rear_pos, axis=1) * 1000.0

  def _virtual_leg_lengths(self) -> tuple[float, float]:
    """Return hip-to-wheel-axis distances from the local XML sites."""
    left_vector = (
      self.data.site_xpos[self.leg_site_ids["Left_wheel_axis_site"]]
      - self.data.site_xpos[self.leg_site_ids["Left_virtual_hip_site"]]
    )
    right_vector = (
      self.data.site_xpos[self.leg_site_ids["Right_wheel_axis_site"]]
      - self.data.site_xpos[self.leg_site_ids["Right_virtual_hip_site"]]
    )
    return float(np.linalg.norm(left_vector)), float(np.linalg.norm(right_vector))

  def _joint_position(self, name: str) -> float:
    joint_id = self.joint_ids[name]
    return float(self.data.qpos[self.model.jnt_qposadr[joint_id]])

  def _joint_velocity(self, name: str) -> float:
    joint_id = self.joint_ids[name]
    return float(self.data.qvel[self.model.jnt_dofadr[joint_id]])

  def _terms(
    self,
    vx: float,
    yaw: float,
    height: float,
  ) -> dict[str, np.ndarray]:
    rotation = self.data.xmat[self.base_id].reshape(3, 3)
    ang_vel_b = rotation.T @ self.data.cvel[self.base_id, :3]
    projected_gravity = rotation.T @ np.array((0.0, 0.0, -1.0))
    command = np.array((vx, yaw, height))
    command_scale = np.array((3.0, 0.25, 5.0))
    return {
      "base_ang_vel": np.clip(ang_vel_b, -20.0, 20.0) * 0.25,
      "projected_gravity": projected_gravity,
      "command": command * command_scale,
      "leg_joint_pos": np.array([self._joint_position(n) for n in LEG_JOINTS]),
      "joint_vel": np.clip(
        np.array([self._joint_velocity(n) for n in POLICY_JOINTS]),
        -200.0,
        200.0,
      ) * 0.05,
      "last_action": np.clip(self.last_action, -10.0, 10.0),
    }

  def observation(
    self,
    vx: float,
    yaw: float,
    height: float,
  ) -> np.ndarray:
    terms = self._terms(vx, yaw, height)
    for name, value in terms.items():
      if name not in self.history:
        self.history[name] = deque(
          (value.copy() for _ in range(5)), maxlen=5
        )
      else:
        self.history[name].append(value.copy())
    current = np.concatenate(list(terms.values()))
    term_major_history = np.concatenate(
      [np.concatenate(tuple(self.history[name])) for name in terms]
    )
    obs = np.concatenate((current, term_major_history)).astype(np.float32)
    if obs.shape != (self.expected_obs_dim,) or not np.isfinite(obs).all():
      raise RuntimeError(f"Invalid policy observation: {obs.shape}")
    return obs[None, :]

  def set_policy_actuators_enabled(self, enabled: bool) -> None:
    """Enable policy actuators or force their generated torques to zero."""
    if enabled == self.policy_actuators_enabled:
      return
    if enabled:
      self.model.actuator_gainprm[self.policy_actuator_ids] = (
        self.policy_gainprm
      )
      self.model.actuator_biasprm[self.policy_actuator_ids] = (
        self.policy_biasprm
      )
    else:
      # ctrl=0 alone is a zero position/velocity target and still generates a
      # PD force. Zeroing both gain and bias disables all six policy actuators.
      self.model.actuator_gainprm[self.policy_actuator_ids] = 0.0
      self.model.actuator_biasprm[self.policy_actuator_ids] = 0.0
      self.data.ctrl[self.policy_actuator_ids] = 0.0
    self.policy_actuators_enabled = enabled

  def step_control(
    self,
    vx: float,
    yaw: float,
    height: float,
    torque_enabled: bool,
    jump_requested: bool = False,
  ) -> tuple[np.ndarray, np.ndarray]:
    self.update_policy(jump_requested)
    if self.active_policy == "jump":
      yaw, height = 0.0, 0.0
    obs = self.observation(vx, yaw, height)
    action = self.session.run(None, {self.input_name: obs})[0][0]
    if action.shape != (6,) or not np.isfinite(action).all():
      raise RuntimeError(f"Invalid ONNX action: {action}")
    action = np.clip(action, -100.0, 100.0)
    self.last_action = action.copy()
    self.set_policy_actuators_enabled(torque_enabled)
    if torque_enabled:
      for name, value in zip(
        ACTION_TARGETS,
        action * ACTION_SCALES,
        strict=True,
      ):
        self.data.ctrl[self.actuator_ids[name]] = value
    self.data.ctrl[self.gas_ids] = 300.0
    self.debug_policy_step(obs, action)
    return obs[0].copy(), action.copy()

  @property
  def mode(self) -> str:
    """Preparation uses the rough session and its unmodified command."""
    return "jumpprepare" if self.jump_prepare else self.active_policy

  def forward_distance(self) -> float:
    """Read the XML rangefinder; missing sensors return an invalid distance."""
    if self.forward_sensor_id is None:
      return float("nan")
    address = self.model.sensor_adr[self.forward_sensor_id]
    return float(self.data.sensordata[address])

  def update_policy(self, jump_requested: bool) -> None:
    """Arm with Space, trigger by distance, and time only the actual jump."""
    target = self.active_policy
    if self.jump_until is not None and self.data.time >= self.jump_until - 1e-9:
      target = "rough"
      self.jump_until = None
    if jump_requested and self.active_policy == "rough" and not self.jump_prepare:
      if "jump" not in self.sessions:
        print("[policy] Space ignored: supply --jump-onnx to enable jumping")
      elif self.forward_sensor_id is None:
        print("[policy] Space ignored: XML has no forward_rangefinder_sensor")
      else:
        self.jump_prepare = True
        print(
          "[policy] jumpprepare: keep rough until forward distance < "
          f"{JUMP_TRIGGER_DISTANCE_M} m"
        )
    if self.jump_prepare:
      distance = self.forward_distance()
      if np.isfinite(distance) and 0.0 <= distance < JUMP_TRIGGER_DISTANCE_M:
        target = "jump"
        self.jump_prepare = False
        self.jump_until = self.data.time + JUMP_DURATION_S
    if target != self.active_policy:
      self.active_policy = target
      self.session = self.sessions[target]
      self.input_name = self.session.get_inputs()[0].name
      # Keep the real previous action and robot state; only history is replaced.
      # observation() backfills all five frames with the target policy command.
      self.history.clear()
      print(f"[policy] switched to {target}; history backfilled with current frame")


@dataclass
class StandupTeacherRecorder:
  """Record disabled settling followed by rough-policy stand-up."""

  output: Path
  disabled_duration: float
  stable_duration: float
  enabled_at: float | None = None
  stable_since: float | None = None
  completed: bool = False
  saved: bool = False
  rows: dict[str, list[np.ndarray | float]] = field(
    default_factory=lambda: {
      "timestamp": [],
      "qpos": [],
      "qvel": [],
      "observation": [],
      "action": [],
      "applied_control": [],
      "next_qpos": [],
      "next_qvel": [],
      "actuator_force": [],
    }
  )

  def torque_enabled(self, elapsed: float) -> bool:
    if self.enabled_at is None and elapsed >= self.disabled_duration - 1e-9:
      self.enabled_at = elapsed
      print(
        "[standup teacher] rough control enabled after "
        f"{elapsed:.3f}s disabled settling"
      )
    return self.enabled_at is not None

  def command(self, controls: KeyboardState) -> tuple[float, float, float, bool]:
    return 0.0, 0.0, controls.height, self.enabled_at is not None

  def is_stable(self, runtime: WheellegRuntime) -> bool:
    rotation = runtime.data.xmat[runtime.base_id].reshape(3, 3)
    velocity = runtime.data.cvel[runtime.base_id]
    return bool(
      runtime.data.qpos[2] >= 0.18
      and rotation[2, 2] >= np.cos(np.deg2rad(10.0))
      and np.linalg.norm(velocity[:3]) <= 0.25
      and np.linalg.norm(velocity[3:]) <= 0.25
    )

  def append(
    self,
    runtime: WheellegRuntime,
    timestamp: float,
    qpos: np.ndarray,
    qvel: np.ndarray,
    observation: np.ndarray,
    action: np.ndarray,
  ) -> None:
    if self.enabled_at is None or self.completed:
      return
    control = np.concatenate(
      (
        runtime.data.ctrl[runtime.policy_actuator_ids].copy(),
        runtime.data.ctrl[runtime.gas_ids].copy(),
      )
    )
    values: dict[str, np.ndarray | float] = {
      "timestamp": timestamp - self.enabled_at,
      "qpos": qpos,
      "qvel": qvel,
      "observation": observation,
      "action": action,
      "applied_control": control,
      "next_qpos": runtime.data.qpos.copy(),
      "next_qvel": runtime.data.qvel.copy(),
      "actuator_force": runtime.data.actuator_force[
        runtime.policy_actuator_ids
      ].copy(),
    }
    for name, value in values.items():
      self.rows[name].append(value)
    if self.is_stable(runtime):
      if self.stable_since is None:
        self.stable_since = timestamp
      elif timestamp - self.stable_since >= self.stable_duration - 1e-9:
        self.completed = True
        self.save(success=True)
        print(
          "[standup teacher] stable standing reached in "
          f"{timestamp - self.enabled_at:.3f}s; trajectory={self.output.resolve()}"
        )
    else:
      self.stable_since = None

  def save(self, *, success: bool) -> None:
    if self.saved:
      return
    if not self.rows["timestamp"]:
      raise RuntimeError("stand-up Teacher produced no enabled-control transitions")
    self.output.parent.mkdir(parents=True, exist_ok=True)
    if self.output.exists():
      raise FileExistsError(f"refusing to overwrite trajectory: {self.output}")
    payload = {
      name: np.asarray(values)
      for name, values in self.rows.items()
    }
    payload.update(
      {
        "success": np.asarray(success),
        "disabled_duration": np.asarray(self.disabled_duration),
        "stable_duration": np.asarray(self.stable_duration),
        "policy_action_names": np.asarray(ACTION_TARGETS),
        "applied_control_names": np.asarray(
          (*ACTION_TARGETS, "Left_gas_spring_tendon", "Right_gas_spring_tendon")
        ),
        "initial_root_position": np.asarray(STANDUP_INITIAL_ROOT_POSITION),
        "initial_root_quaternion_wxyz": np.asarray(
          STANDUP_INITIAL_ROOT_QUATERNION_WXYZ
        ),
        "initial_joint_names": np.asarray(
          tuple(STANDUP_INITIAL_JOINT_POSITIONS)
        ),
        "initial_joint_positions": np.asarray(
          tuple(STANDUP_INITIAL_JOINT_POSITIONS.values())
        ),
      }
    )
    np.savez_compressed(self.output, **payload)
    self.saved = True


def run_loop(
  runtime,
  controls,
  decimation,
  duration,
  viewer=None,
  standup_teacher: StandupTeacherRecorder | None = None,
) -> np.ndarray:
  """Run policy inference and physics without any environment or training code."""
  model, data = runtime.model, runtime.data
  control_dt = model.opt.timestep * decimation
  start_time = data.time
  max_torque = np.zeros(6)
  while viewer is None or viewer.is_running():
    if duration > 0.0 and data.time - start_time >= duration - 1.0e-9:
      break
    start = time.perf_counter()
    elapsed = data.time - start_time
    if standup_teacher is None or standup_teacher.completed:
      command = controls.command(control_dt)
      jump_requested = controls.consume_jump_request()
    else:
      standup_teacher.torque_enabled(elapsed)
      command = standup_teacher.command(controls)
      jump_requested = False
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()
    observation, action = runtime.step_control(
      *command,
      jump_requested=jump_requested,
    )
    for _ in range(decimation):
      mujoco.mj_step(model, data)
      max_torque = np.maximum(
        max_torque, np.abs(data.actuator_force[runtime.policy_actuator_ids])
      )
    if standup_teacher is not None and standup_teacher.enabled_at is not None:
      standup_teacher.append(
        runtime,
        data.time - start_time,
        qpos,
        qvel,
        observation,
        action,
      )
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
      raise RuntimeError("MuJoCo state contains NaN/Inf; stopping playback")
    if viewer is not None:
      sync_torque_controls(viewer, model, data, runtime.policy_actuator_ids)
      remaining = control_dt - (time.perf_counter() - start)
      if remaining > 0.0:
        time.sleep(remaining)
  print(f"[play] simulated {data.time - start_time:.2f}s; max |torque|={max_torque}")
  if standup_teacher is not None and not standup_teacher.saved:
    standup_teacher.save(success=standup_teacher.completed)
  return max_torque


def main() -> None:
  args = parse_args()
  if mujoco.__version__ != "3.9.0":
    raise RuntimeError("Use conda activate mujoco390 with mujoco==3.9.0")
  if min(args.debug_interval, args.timestep) <= 0.0:
    raise ValueError("Debug interval and timestep must be positive")
  if args.control_decimation <= 0 or args.duration < 0.0:
    raise ValueError("Decimation must be positive; duration must be nonnegative")
  if min(args.standup_disabled_duration, args.standup_stable_duration) <= 0.0:
    raise ValueError("Stand-up Teacher durations must be positive")
  if args.headless and args.duration <= 0.0:
    raise ValueError("--headless requires --duration with a positive value")
  if args.standup_teacher_output is not None and args.state is not None:
    raise ValueError("--standup-teacher-output uses its explicit pose, not --state")
  if (
    args.headless
    and args.standup_teacher_output is not None
    and args.duration
    <= args.standup_disabled_duration + args.standup_stable_duration
  ):
    raise ValueError("Headless duration is too short for the stand-up Teacher")
  height_range = tuple(args.height_range)
  if height_range[0] > height_range[1]:
    raise ValueError("Height range must be in ascending order")
  for path in (args.xml, args.onnx):
    if not path.is_file():
      raise FileNotFoundError(path)
  if args.jump_onnx is not None and not args.jump_onnx.is_file():
    raise FileNotFoundError(args.jump_onnx)
  initial_height = float(np.clip(args.height, *height_range))
  model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
  model.opt.timestep = args.timestep
  data = mujoco.MjData(model)
  if args.standup_teacher_output is not None:
    set_explicit_standup_initial_pose(model, data)
  else:
    state = args.state
    if state is None and args.xml.resolve() == (ROOT / "mjcf/scene.xml").resolve():
      state = ROOT / "initial_state.npz"
    if state is not None:
      with np.load(state) as initial:
        if initial["qpos"].shape != data.qpos.shape:
          raise ValueError("Initial qpos shape does not match XML")
        if initial["qvel"].shape != data.qvel.shape:
          raise ValueError("Initial qvel shape does not match XML")
        data.qpos[:] = initial["qpos"]
        data.qvel[:] = initial["qvel"]
    elif model.nkey:
      mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
  runtime = WheellegRuntime(
    model, data, args.onnx, args.debug_interval if args.debug else None,
    jump_policy=args.jump_onnx,
  )
  controls = KeyboardState(
    args.vx, args.yaw, initial_height, height_range
  )
  standup_teacher = None
  if args.standup_teacher_output is not None:
    standup_teacher = StandupTeacherRecorder(
      output=args.standup_teacher_output,
      disabled_duration=args.standup_disabled_duration,
      stable_duration=args.standup_stable_duration,
    )
    print(
      "[standup teacher] explicit legs-behind reset; policy actuators disabled, "
      "gas springs remain enabled"
    )
  print(f"MuJoCo {mujoco.__version__}; ONNX {args.onnx.resolve()}")
  print_rm_terrain_summary(model)
  if args.jump_onnx is not None:
    print(f"Jump ONNX {args.jump_onnx.resolve()}; duration {JUMP_DURATION_S}s")
  print(f"Height command {initial_height:.3f}m in {height_range}")
  print(
    f"Physics dt={args.timestep}s; "
    f"control dt={args.timestep * args.control_decimation}s"
  )
  print("[motor limits] leg +/-28 N*m; wheel +/-4 N*m; gas spring 300 N")
  if args.headless:
    run_loop(
      runtime,
      controls,
      args.control_decimation,
      args.duration,
      standup_teacher=standup_teacher,
    )
    return
  print("Numpad 8/5 forward/back; 4/6 left/right; 7/9 raise/lower; 1 off, 2 restore")
  print(
    f"Numpad slow: vx +/-{SLOW_LINEAR_SPEED}m/s, yaw +/-{SLOW_YAW_SPEED}rad/s; "
    f"Shift: +/-{FAST_LINEAR_SPEED}m/s, +/-{FAST_YAW_SPEED}rad/s"
  )
  print(
    f"Space: jumpprepare; distance < {JUMP_TRIGGER_DISTANCE_M}m triggers jump "
    f"for {JUMP_DURATION_S}s, then returns to rough"
  )
  print("[GUI Control] motor torque (N*m), monitor only; gas spring force (N)")
  listener = keyboard.Listener(
    on_press=controls.on_press, on_release=controls.on_release, suppress=False,
    win32_event_filter=controls.on_win32_event,
  )
  controls.shift_state_reader = make_shift_state_reader()
  listener.start()
  try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
      run_loop(
        runtime,
        controls,
        args.control_decimation,
        args.duration,
        viewer,
        standup_teacher,
      )
  finally:
    listener.stop()
    listener.join(timeout=1.0)


if __name__ == "__main__":
  main()
