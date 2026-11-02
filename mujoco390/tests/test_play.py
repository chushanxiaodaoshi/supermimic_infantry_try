"""Local 150-D playback tests without PyTorch or an external ONNX file."""

import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("local_play", ROOT / "run_policy.py")
PLAY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLAY
SPEC.loader.exec_module(PLAY)


def fake_session(width=150, action=None):
  result = np.zeros((1, 6), dtype=np.float32) if action is None else action
  info = SimpleNamespace(name="obs", shape=[1, width], type="tensor(float)")
  output = SimpleNamespace(name="actions", shape=[1, 6], type="tensor(float)")
  return SimpleNamespace(
    get_inputs=lambda: [info], get_outputs=lambda: [output],
    run=lambda *_: [result],
  )


def make_runtime(action=None, jump_policy=None, scene="scene.xml"):
  model = mujoco.MjModel.from_xml_path(str(ROOT / "mjcf" / scene))
  data = mujoco.MjData(model)
  with np.load(ROOT / "initial_state.npz") as initial:
    data.qpos[:] = initial["qpos"]
    data.qvel[:] = initial["qvel"]
  mujoco.mj_forward(model, data)
  with patch.object(
    PLAY.ort, "InferenceSession", return_value=fake_session(150, action)
  ):
    return PLAY.WheellegRuntime(
      model, data, Path("unused.onnx"), jump_policy=jump_policy,
    )


def test_standup_teacher_uses_explicit_named_legs_behind_pose():
  runtime = make_runtime()
  PLAY.set_explicit_standup_initial_pose(runtime.model, runtime.data)
  np.testing.assert_allclose(
    runtime.data.qpos[:3], PLAY.STANDUP_INITIAL_ROOT_POSITION
  )
  np.testing.assert_allclose(
    runtime.data.qpos[3:7], PLAY.STANDUP_INITIAL_ROOT_QUATERNION_WXYZ
  )
  for name, expected in PLAY.STANDUP_INITIAL_JOINT_POSITIONS.items():
    joint_id = PLAY._suffix_id(
      runtime.model,
      mujoco.mjtObj.mjOBJ_JOINT,
      runtime.model.njnt,
      name,
    )
    assert runtime.data.qpos[runtime.model.jnt_qposadr[joint_id]] == expected
  left_hip = runtime.data.site_xpos[runtime.leg_site_ids["Left_virtual_hip_site"]]
  left_wheel = runtime.data.site_xpos[runtime.leg_site_ids["Left_wheel_axis_site"]]
  right_hip = runtime.data.site_xpos[runtime.leg_site_ids["Right_virtual_hip_site"]]
  right_wheel = runtime.data.site_xpos[runtime.leg_site_ids["Right_wheel_axis_site"]]
  assert left_wheel[0] < left_hip[0]
  assert right_wheel[0] < right_hip[0]


def test_standup_teacher_record_has_replay_contract_fields():
  runtime = make_runtime()
  PLAY.set_explicit_standup_initial_pose(runtime.model, runtime.data)
  with TemporaryDirectory() as directory:
    output = Path(directory) / "standup.npz"
    recorder = PLAY.StandupTeacherRecorder(
      output=output, disabled_duration=1.0, stable_duration=0.2
    )
    assert not recorder.torque_enabled(0.99)
    assert recorder.torque_enabled(1.0)
    observation = np.zeros(150, dtype=np.float32)
    action = np.zeros(6, dtype=np.float32)
    qpos, qvel = runtime.data.qpos.copy(), runtime.data.qvel.copy()
    command = (0.0, 0.0, 0.2, True)
    recorder.append(runtime, 1.0, qpos, qvel, observation, action, command)
    recorder.append(runtime, 1.2, qpos, qvel, observation, action, command)
    assert recorder.completed and output.is_file()
    with np.load(output) as trajectory:
      assert trajectory["success"]
      assert trajectory["qpos"].shape == (2, runtime.model.nq)
      assert trajectory["qvel"].shape == (2, runtime.model.nv)
      assert trajectory["observation"].shape == (2, 150)
      assert trajectory["action"].shape == (2, 6)
      assert trajectory["applied_control"].shape == (2, 8)
      assert trajectory["command"].shape == (2, 3)
      assert trajectory["torque_enabled"].tolist() == [1.0, 1.0]
      assert trajectory["initial_joint_names"].shape == (14,)


def test_randomized_standup_teacher_uses_reproducible_one_second_outages():
  runtime = make_runtime()
  PLAY.set_explicit_standup_initial_pose(runtime.model, runtime.data)
  with TemporaryDirectory() as directory:
    recorder = PLAY.StandupTeacherRecorder(
      output=Path(directory) / "randomized_standup.npz",
      disabled_duration=1.0,
      stable_duration=0.2,
      randomize=True,
      seed=7,
      command_interval=(1.0, 1.0),
      disable_interval=(2.0, 2.0),
      vx_range=(-0.5, 0.5),
      yaw_range=(-1.0, 1.0),
      height_range=(0.18, 0.22),
    )
    observation = np.zeros(150, dtype=np.float32)
    action = np.zeros(6, dtype=np.float32)
    qpos, qvel = runtime.data.qpos.copy(), runtime.data.qvel.copy()
    assert recorder.torque_enabled(1.0)
    command = recorder.command(KeyboardStateStub(), 1.0, True)
    recorder.append(runtime, 1.0, qpos, qvel, observation, action, command)
    recorder.append(runtime, 1.2, qpos, qvel, observation, action, command)
    assert recorder.first_stable_at == 1.2 and not recorder.completed

    enabled = recorder.torque_enabled(1.21)
    random_command = recorder.command(KeyboardStateStub(), 1.21, enabled)
    assert -0.5 <= random_command[0] <= 0.5
    assert -1.0 <= random_command[1] <= 1.0
    assert 0.18 <= random_command[2] <= 0.22
    assert recorder.torque_enabled(3.19)
    assert not recorder.torque_enabled(3.20)
    assert not recorder.torque_enabled(4.19)
    assert recorder.torque_enabled(4.20)


class KeyboardStateStub:
  height = 0.2


def test_default_training_scene_is_one_infinite_flat_plane():
  runtime = make_runtime()
  model = runtime.model
  terrain = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
  assert model.geom_type[terrain] == mujoco.mjtGeom.mjGEOM_PLANE
  np.testing.assert_allclose(model.geom_pos[terrain], [0.0, 0.0, 0.0])
  assert model.nhfield == 0
  names = {
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
    for index in range(model.ngeom)
  }
  assert not any(name.startswith(("rm_", "classic_")) for name in names)


def test_rm_course_heights_collision_masks_and_gaps():
  runtime = make_runtime(scene="scene_validation_terrain.xml")
  model, data = runtime.model, runtime.data
  terrain_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "terrain")
  for index in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
    if name.startswith("rm_"):
      assert model.geom_bodyid[index] == terrain_id
      assert model.geom_contype[index] == model.geom_conaffinity[index] == 1
  samples = [
    (2.2, 0, 0), (2.8, 0, 0.20), (3.4, 0, 0.20),
    (3.56, 0, 0.35), (4.0, 0, 0.30), (5.2, 0, 0.20), (5.8, 0, 0),
    (2.8, -1.65, 0.05), (3.8, -1.65, 0.10), (4.8, -1.65, 0.20),
    (8.0, 0, 0.175), (8.5, 0, 0.175 + 0.5 * 0.35 / 1.144798),
    (8.8, 0, 0), (-1, 1, 0), (15, 0, 0),
  ]
  for x, y, expected_height in samples:
    hit = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(
      model, data, np.array([x, y, 2.0]), np.array([0.0, 0.0, -1.0]),
      None, True, -1, hit,
    )
    assert distance >= 0 and hit[0] >= 0
    np.testing.assert_allclose(2.0 - distance, expected_height, atol=1e-8)
  for name in (
    "standalone_step_05cm", "standalone_step_10cm", "standalone_step_20cm",
    "standalone_ramp_17deg_35cm",
  ):
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0


def test_classic_terrain_pits_are_not_filled_by_ground_and_tiles_are_compact():
  runtime = make_runtime(scene="scene_validation_terrain.xml")
  model, data = runtime.model, runtime.data

  def height(x, y):
    hit = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(
      model, data, np.array([x, y, 2.0]), np.array([0.0, 0.0, -1.0]),
      None, True, -1, hit,
    )
    assert hit[0] >= 0
    return 2.0 - distance

  np.testing.assert_allclose(height(4, 4), 0.57, atol=2e-6)
  np.testing.assert_allclose(height(8, 4), -0.57, atol=2e-6)
  np.testing.assert_allclose(height(4, -4), 0.441, atol=2e-6)
  np.testing.assert_allclose(height(8, -4), -0.441, atol=2e-6)
  wave_heights = [height(x, y) for x in (11, 12, 13) for y in (3, 4, 5)]
  assert min(wave_heights) < -0.01 and max(wave_heights) > 0.01
  for x in np.arange(2.1, 14.0, 0.4):
    for y in np.arange(-5.9, 6.0, 0.4):
      assert height(x, y) > -0.8  # No unfilled cracks down to the safety plane.
  terrain_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "terrain")
  for index in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
    if name.startswith("classic_"):
      assert model.geom_bodyid[index] == terrain_id
      assert model.geom_contype[index] == model.geom_conaffinity[index] == 1
  assert model.nhfield == 5
  for x in np.arange(10.1, 14, 0.4):
    for y in np.arange(-1.9, 2, 0.4):
      np.testing.assert_allclose(height(x, y), 0.0, atol=1e-8)
  rough_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "classic_rough_00")
  np.testing.assert_allclose(model.geom_pos[rough_id, :2], [0.0, 4.0])


def test_only_150_is_accepted():
  with patch.object(PLAY.ort, "InferenceSession", return_value=fake_session(162)):
    with np.testing.assert_raises(ValueError):
      PLAY.WheellegRuntime(None, None, Path("unused.onnx"))


def test_observation_has_current25_and_term_major_history125():
  runtime = make_runtime()
  obs = runtime.observation(0.2, 0.3, 0.13)
  assert obs.shape == (1, 150) and obs.dtype == np.float32
  np.testing.assert_allclose(obs[0, 6:9], [0.6, 0.075, 0.65])
  np.testing.assert_array_equal(obs[0, 25:40], np.tile(obs[0, :3], 5))
  np.testing.assert_array_equal(obs[0, 55:70], np.tile(obs[0, 6:9], 5))


def test_real_force_limits_and_torque_off_keep_gas_spring():
  runtime = make_runtime(np.full((1, 6), 100.0, dtype=np.float32))
  np.testing.assert_array_equal(PLAY.POLICY_TORQUE_LIMITS, [28, 28, 4, 28, 28, 4])
  runtime.step_control(0.0, 0.0, 0.0, True)
  mujoco.mj_forward(runtime.model, runtime.data)
  np.testing.assert_allclose(
    runtime.data.actuator_force[runtime.policy_actuator_ids],
    PLAY.POLICY_TORQUE_LIMITS,
  )
  runtime.step_control(0.0, 0.0, 0.0, False)
  mujoco.mj_forward(runtime.model, runtime.data)
  np.testing.assert_array_equal(
    runtime.data.actuator_force[runtime.policy_actuator_ids], np.zeros(6)
  )
  np.testing.assert_allclose(runtime.data.actuator_force[runtime.gas_ids], 300.0)
  runtime.step_control(0.0, 0.0, 0.0, True)
  mujoco.mj_forward(runtime.model, runtime.data)
  np.testing.assert_allclose(
    runtime.data.actuator_force[runtime.policy_actuator_ids],
    PLAY.POLICY_TORQUE_LIMITS,
  )


def test_keyboard_release_and_height_lock():
  controls = PLAY.KeyboardState(0.0, 0.0, 0.0, (0.0, 0.0))
  key = PLAY.keyboard.KeyCode.from_vk(0x68)
  controls.on_press(key)
  assert controls.command(0.01)[0] == PLAY.SLOW_LINEAR_SPEED
  controls.on_release(key)
  assert controls.command(0.01)[0] == 0.0
  controls.on_press(PLAY.keyboard.KeyCode.from_vk(0x67))
  assert controls.command(0.01)[2] == 0.0


def test_only_keypad_digits_control_motion_height_and_torque():
  controls = PLAY.KeyboardState(0, 0, 0.2, (0.1, 0.3))
  for digit in "123456789":
    key = PLAY.keyboard.KeyCode.from_char(digit, vk=ord(digit))
    controls.on_press(key)
    controls.on_release(key)
  assert controls.command(0.1) == (0, 0, 0.2, True)
  controls.on_press(PLAY.keyboard.KeyCode.from_vk(0x61))
  assert not controls.command(0.1)[3]
  controls.on_press(PLAY.keyboard.KeyCode.from_char("2", vk=0x32))
  assert not controls.command(0.1)[3]
  controls.on_press(PLAY.keyboard.KeyCode.from_vk(0x62))
  assert controls.command(0.1)[3]
  controls.on_press(PLAY.keyboard.KeyCode.from_vk(0x67))
  assert controls.command(0.1)[2] > 0.2


def test_new_keypad_motion_layout_and_height_directions():
  controls = PLAY.KeyboardState(0, 0, 0.2, (0.1, 0.3))
  for vk, expected in (
    (0x68, (1, 0)), (0x65, (-1, 0)), (0x64, (0, 2)), (0x66, (0, -2)),
  ):
    key = PLAY.keyboard.KeyCode.from_vk(vk)
    controls.on_press(key)
    assert controls.command(0.01)[:2] == expected
    controls.on_release(key)
    assert controls.command(0.01)[:2] == (0, 0)
  key = PLAY.keyboard.KeyCode.from_vk(0x67)
  controls.on_press(key)
  raised = controls.command(0.1)[2]
  assert raised > 0.2
  controls.on_release(key)
  assert controls.command(0.1)[2] == raised
  page_up = SimpleNamespace(vkCode=0x21, scanCode=0x49, flags=0)
  assert not controls.on_win32_event(0x100, page_up)  # Physical keypad9.
  lowered = controls.command(0.1)[2]
  assert lowered < raised
  controls.on_win32_event(0x101, page_up)
  assert controls.command(0.1)[2] == lowered


def test_keypad_one_zeros_real_motor_torque_and_two_restores():
  runtime = make_runtime(np.full((1, 6), 100.0, dtype=np.float32))
  controls = PLAY.KeyboardState(0, 0, 0.2, (0.1, 0.3))
  controls.on_press(PLAY.keyboard.KeyCode.from_vk(0x61))
  runtime.step_control(*controls.command(0.01))
  mujoco.mj_forward(runtime.model, runtime.data)
  np.testing.assert_allclose(
    runtime.data.actuator_force[runtime.policy_actuator_ids], np.zeros(6),
  )
  np.testing.assert_allclose(runtime.data.actuator_force[runtime.gas_ids], 300.0)
  controls.on_press(PLAY.keyboard.KeyCode.from_vk(0x62))
  runtime.step_control(*controls.command(0.01))
  mujoco.mj_forward(runtime.model, runtime.data)
  np.testing.assert_allclose(
    runtime.data.actuator_force[runtime.policy_actuator_ids], PLAY.POLICY_TORQUE_LIMITS,
  )


def test_shift_changes_held_keypad_speeds_and_release_always_stops():
  controls = PLAY.KeyboardState(0, 0, 0.2, (0.1, 0.3))
  keys = [PLAY.keyboard.KeyCode.from_vk(vk) for vk in (0x68, 0x64)]
  for key in keys:
    controls.on_press(key)
  assert controls.command(0.01)[:2] == (1.0, 2.0)
  controls.on_press(PLAY.keyboard.Key.shift_l)
  assert controls.command(0.01)[:2] == (2.0, 4.0)
  controls.on_press(PLAY.keyboard.Key.shift_r)
  controls.on_release(PLAY.keyboard.Key.shift_l)
  assert controls.command(0.01)[:2] == (2.0, 4.0)
  controls.on_release(PLAY.keyboard.Key.shift_r)
  assert controls.command(0.01)[:2] == (1.0, 2.0)
  controls.on_press(PLAY.keyboard.Key.shift_l)
  for key in keys:
    controls.on_release(key)
  assert controls.command(0.01)[:2] == (0, 0)


def test_shift_first_then_any_motion_key_accelerates_with_live_shift_state():
  for vk, expected in (
    (0x68, (2, 0)), (0x65, (-2, 0)), (0x64, (0, 4)), (0x66, (0, -4)),
  ):
    controls = PLAY.KeyboardState(0, 0, 0.2, (0.1, 0.3))
    held = {"shift": True}
    controls.shift_state_reader = lambda: held["shift"]
    assert controls.command(0.01)[:2] == (0, 0)
    key = PLAY.keyboard.KeyCode.from_vk(vk)
    controls.on_press(key)
    assert controls.command(0.01)[:2] == expected
    held["shift"] = False
    # Live polling must override stale hook state on Shift release.
    controls.on_press(PLAY.keyboard.Key.shift_l)
    slow = tuple(value / 2 for value in expected)
    assert controls.command(0.01)[:2] == slow
    held["shift"] = True
    assert controls.command(0.01)[:2] == expected
    controls.on_release(key)
    assert controls.command(0.01)[:2] == (0, 0)


def test_raw_shift_first_then_keypad_ignores_generic_fake_shift_release():
  controls = PLAY.KeyboardState(0, 0, 0.2, (0.1, 0.3))
  event = lambda vk, scan: SimpleNamespace(vkCode=vk, scanCode=scan, flags=0)
  controls.on_win32_event(0x100, event(0xA0, 0x2A))
  controls.on_win32_event(0x101, event(0x10, 0x2A))  # Generic fake Shift up.
  controls.on_win32_event(0x100, event(0x26, 0x48))  # Shift+keypad8 -> Up.
  assert controls.command(0.01)[:2] == (2, 0)
  controls.on_win32_event(0x101, event(0x68, 0x48))
  assert controls.command(0.01)[:2] == (0, 0)


def test_raw_keypad_scan_codes_survive_shift_and_numlock_navigation_events():
  controls = PLAY.KeyboardState(0, 0, 0.2, (0.1, 0.3))
  event = lambda vk, scan, flags=0: SimpleNamespace(
    vkCode=vk, scanCode=scan, flags=flags,
  )
  assert not controls.on_win32_event(0x100, event(0x26, 0x48))  # Keypad Up/8.
  assert controls.command(0.01)[0] == 1.0
  assert not controls.on_win32_event(0x100, event(0xA0, 0x2A))
  assert controls.command(0.01)[0] == 2.0
  controls.on_win32_event(0x101, event(0x10, 0x2A, 0x10))  # Synthetic Shift up.
  assert controls.command(0.01)[0] == 2.0
  controls.on_win32_event(0x101, event(0x68, 0x48))  # Different release VK.
  assert controls.command(0.01)[0] == 0
  controls.on_win32_event(0x101, event(0xA0, 0x2A))
  assert controls.on_win32_event(0x100, event(0x26, 0x48, 1))  # Dedicated Up.
  assert controls.command(0.01)[0] == 0


def test_invalid_action_stops_before_control_is_applied():
  runtime = make_runtime(np.full((1, 6), np.nan, dtype=np.float32))
  before = runtime.data.ctrl.copy()
  with np.testing.assert_raises(RuntimeError):
    runtime.step_control(0.0, 0.0, 0.0, True)
  np.testing.assert_array_equal(runtime.data.ctrl, before)


def test_gui_force_display_restores_servo_targets_and_ranges():
  runtime = make_runtime()
  model, data = runtime.model, runtime.data
  ids = runtime.policy_actuator_ids
  controls = data.ctrl.copy()
  ranges = model.actuator_ctrlrange.copy()
  limited = model.actuator_ctrllimited.copy()

  def sync():
    np.testing.assert_array_equal(data.ctrl[ids], data.actuator_force[ids])
    np.testing.assert_array_equal(
      model.actuator_ctrlrange[ids, 1], PLAY.POLICY_TORQUE_LIMITS
    )
    data.ctrl[ids] = 123.0

  PLAY.sync_torque_controls(SimpleNamespace(sync=sync), model, data, ids)
  np.testing.assert_array_equal(data.ctrl, controls)
  np.testing.assert_array_equal(model.actuator_ctrlrange, ranges)
  np.testing.assert_array_equal(model.actuator_ctrllimited, limited)


def assert_history_is_current_frame(obs):
  offset, history_offset = 0, 25
  for width in (3, 3, 3, 4, 6, 6):
    np.testing.assert_array_equal(
      obs[0, history_offset:history_offset + 5 * width],
      np.tile(obs[0, offset:offset + width], 5),
    )
    offset += width
    history_offset += 5 * width


def test_switches_both_sessions_commands_and_all_five_history_frames():
  runtime = make_runtime(jump_policy=Path("jump.onnx"))
  runtime.data.sensordata[runtime.model.sensor_adr[runtime.forward_sensor_id]] = (
    PLAY.JUMP_TRIGGER_DISTANCE_M * 0.5
  )
  rough_obs, jump_obs = [], []
  rough = runtime.sessions["rough"]
  jump = fake_session(action=np.full((1, 6), 2.0, dtype=np.float32))
  rough.run = lambda _, feed: (
    rough_obs.append(feed["obs"].copy()) or [np.ones((1, 6), dtype=np.float32)]
  )
  jump.run = lambda _, feed: (
    jump_obs.append(feed["obs"].copy()) or
    [np.full((1, 6), 2.0, dtype=np.float32)]
  )
  runtime.sessions["jump"] = jump
  runtime.step_control(0.2, 1.0, 0.2, True)
  runtime.data.time = 1.0
  qpos = runtime.data.qpos.copy()
  runtime.step_control(0.3, 4.0, 0.3, True, jump_requested=True)
  assert runtime.active_policy == "jump"
  assert len(rough_obs) == 1 and len(jump_obs) == 1
  np.testing.assert_allclose(jump_obs[-1][0, 6:9], [0.9, 0, 0])
  np.testing.assert_array_equal(jump_obs[-1][0, 19:25], np.ones(6))
  assert_history_is_current_frame(jump_obs[-1])
  np.testing.assert_array_equal(runtime.data.qpos, qpos)
  deadline = 1.0 + PLAY.JUMP_DURATION_S
  assert runtime.jump_until == deadline
  runtime.data.time = deadline - 0.01
  runtime.step_control(0.3, -4.0, 0.1, True, jump_requested=True)
  assert runtime.active_policy == "jump" and runtime.jump_until == deadline
  runtime.data.time = deadline
  runtime.step_control(0.4, -4.0, 0.18, True, jump_requested=True)
  assert runtime.active_policy == "rough" and runtime.jump_until is None
  np.testing.assert_allclose(rough_obs[-1][0, 6:9], [1.2, -1.0, 0.9])
  np.testing.assert_array_equal(rough_obs[-1][0, 19:25], np.full(6, 2.0))
  assert_history_is_current_frame(rough_obs[-1])


def test_space_is_edge_triggered_and_does_not_repeat_while_held():
  controls = PLAY.KeyboardState(0, 0, 0.2, (0.1, 0.3))
  controls.on_press(PLAY.keyboard.Key.space)
  assert controls.consume_jump_request()
  for _ in range(100):
    controls.on_press(PLAY.keyboard.Key.space)
    assert not controls.consume_jump_request()
  controls.on_release(PLAY.keyboard.Key.space)
  controls.on_press(PLAY.keyboard.Key.space)
  assert controls.consume_jump_request()
  controls.on_press(PLAY.keyboard.KeyCode.from_vk(0x64))
  assert controls.command(0.01)[1] == PLAY.SLOW_YAW_SPEED


def test_jump_uses_simulation_time_for_configured_duration():
  runtime = make_runtime(jump_policy=Path("jump.onnx"))
  runtime.data.sensordata[runtime.model.sensor_adr[runtime.forward_sensor_id]] = (
    PLAY.JUMP_TRIGGER_DISTANCE_M * 0.5
  )
  steps = round(PLAY.JUMP_DURATION_S / 0.01)
  for index in range(steps + 1):
    runtime.data.time = index * 0.01
    runtime.step_control(0, 4, 0.2, True, jump_requested=index == 0)
    assert runtime.active_policy == ("jump" if index < steps else "rough")


def test_prepare_waits_for_valid_strict_threshold_and_times_from_trigger():
  runtime = make_runtime(jump_policy=Path("jump.onnx"))
  address = runtime.model.sensor_adr[runtime.forward_sensor_id]
  threshold = PLAY.JUMP_TRIGGER_DISTANCE_M
  runtime.data.sensordata[address] = threshold * 0.5
  runtime.step_control(0.2, 1, 0.2, True)
  assert runtime.mode == "rough"  # Nearby obstacle alone must not arm jumping.
  runtime.data.sensordata[address] = 1.2
  runtime.step_control(0.2, 1, 0.2, True, jump_requested=True)
  assert runtime.mode == "jumpprepare" and runtime.active_policy == "rough"
  assert runtime.jump_until is None
  invalid = [-1, float("nan"), float("inf"), threshold, threshold + 0.1]
  for index, distance in enumerate(invalid):
    runtime.data.time = index + 1.0
    runtime.data.sensordata[address] = distance
    runtime.step_control(0.3, 4, 0.3, True, jump_requested=True)
    assert runtime.mode == "jumpprepare" and runtime.jump_until is None
    assert runtime.session is runtime.sessions["rough"]
    np.testing.assert_allclose(runtime.history["command"][-1], [0.9, 1.0, 1.5])
  runtime.data.time = 10.0
  runtime.data.sensordata[address] = threshold - 0.001
  runtime.step_control(0.3, 4, 0.3, True)
  assert runtime.mode == "jump" and not runtime.jump_prepare
  deadline = 10.0 + PLAY.JUMP_DURATION_S
  assert runtime.jump_until == deadline
  runtime.data.sensordata[address] = -1
  runtime.data.time = deadline - 0.01
  runtime.step_control(0.3, 4, 0.3, True, jump_requested=True)
  assert runtime.mode == "jump" and runtime.jump_until == deadline
  runtime.data.time = deadline
  runtime.step_control(0.3, 4, 0.3, True)
  assert runtime.mode == "rough" and runtime.jump_until is None
  runtime.data.sensordata[address] = threshold * 0.5
  runtime.step_control(0.3, 4, 0.3, True)
  assert runtime.mode == "rough"  # No automatic repeat after one armed jump.


def test_prepare_does_not_clear_rough_history_or_change_commands():
  runtime = make_runtime(jump_policy=Path("jump.onnx"))
  runtime.data.sensordata[runtime.model.sensor_adr[runtime.forward_sensor_id]] = 2.0
  runtime.step_control(0.2, 1, 0.2, True)
  runtime.step_control(0.3, 4, 0.3, True, jump_requested=True)
  assert runtime.mode == "jumpprepare"
  np.testing.assert_allclose(runtime.history["command"][-2], [0.6, 0.25, 1.0])
  np.testing.assert_allclose(runtime.history["command"][-1], [0.9, 1.0, 1.5])


def test_missing_forward_sensor_cannot_arm_jumping():
  runtime = make_runtime(jump_policy=Path("jump.onnx"))
  runtime.forward_sensor_id = None
  runtime.step_control(0.3, 4, 0.3, True, jump_requested=True)
  assert runtime.mode == "rough" and runtime.jump_until is None


def test_forward_distance_reads_real_mujoco_rangefinder():
  model = mujoco.MjModel.from_xml_string('''
    <mujoco>
      <worldbody>
        <geom type="box" pos="1.2 0 0" size="0.1 0.1 0.1"/>
        <body name="sensor_body">
          <site name="front" quat="0.70710678 0 0.70710678 0"/>
        </body>
      </worldbody>
      <sensor><rangefinder site="front" name="forward_rangefinder_sensor"/></sensor>
    </mujoco>
  ''')
  runtime = object.__new__(PLAY.WheellegRuntime)
  runtime.model, runtime.data = model, mujoco.MjData(model)
  runtime.forward_sensor_id = 0
  mujoco.mj_forward(model, runtime.data)
  np.testing.assert_allclose(runtime.forward_distance(), 1.1)
  model.geom_pos[0, 0] = 0.8
  mujoco.mj_forward(model, runtime.data)
  np.testing.assert_allclose(runtime.forward_distance(), 0.7)
  model.geom_pos[0, 1] = 2.0
  mujoco.mj_forward(model, runtime.data)
  assert runtime.forward_distance() < 0


def test_invalid_jump_input_is_rejected_before_startup():
  with patch.object(
    PLAY.ort, "InferenceSession", side_effect=[fake_session(), fake_session(162)]
  ):
    with np.testing.assert_raises(ValueError):
      PLAY.WheellegRuntime(None, None, Path("rough"), jump_policy=Path("jump"))


def test_single_policy_ignores_jump_requests():
  runtime = make_runtime()
  runtime.step_control(0, 1, 0.2, True, jump_requested=True)
  assert runtime.active_policy == "rough"
  assert runtime.jump_until is None


def load_tests(loader, tests, pattern):
  del loader, tests, pattern
  return unittest.TestSuite(
    unittest.FunctionTestCase(value)
    for name, value in sorted(globals().items()) if name.startswith("test_")
  )
