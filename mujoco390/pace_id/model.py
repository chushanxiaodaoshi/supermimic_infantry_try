"""Strict fixed-base MuJoCo adapter for PACE baseline replay."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
import tempfile
from typing import Any

import mujoco
import numpy as np

from .core import Mapping, file_sha256, wrap_angle


POLICY_ACTUATOR_SUFFIXES = (
  "Left_rear_joint",
  "Right_rear_joint",
  "Left_front_joint",
  "Right_front_joint",
  "Left_Wheel_joint",
  "Right_Wheel_joint",
)
GAS_ACTUATOR_SUFFIXES = (
  "Left_gas_spring_tendon",
  "Right_gas_spring_tendon",
)
PASSIVE_JOINT_SUFFIXES = (
  "rear_child1_joint",
  "front_child1_joint",
  "front_child2_joint",
  "front_child3_joint",
)


def object_id(
  model: mujoco.MjModel,
  kind: mujoco.mjtObj,
  name: str,
) -> int:
  result = mujoco.mj_name2id(model, kind, name)
  if result < 0:
    raise ValueError(f"Required MuJoCo object is missing: {name}")
  return result


def _remove_free_joint_and_keyframe(text: str) -> str:
  fixed, free_count = re.subn(
    r'<joint\s+name="robot/base_free_joint"\s+type="free"\s*/>',
    "",
    text,
  )
  if free_count != 1:
    raise ValueError(
      "Expected exactly one robot/base_free_joint before fixing the base"
    )
  fixed = re.sub(r"\s*<keyframe>.*?</keyframe>", "", fixed, flags=re.DOTALL)
  return fixed


def load_fixed_base_model(xml_path: Path, timestep_s: float) -> mujoco.MjModel:
  xml_path = xml_path.resolve()
  fixed_xml = _remove_free_joint_and_keyframe(
    xml_path.read_text(encoding="utf-8")
  )
  temporary_path: Path | None = None
  try:
    with tempfile.NamedTemporaryFile(
      mode="w",
      suffix=".xml",
      prefix="pace_fixed_",
      dir=xml_path.parent,
      delete=False,
      encoding="utf-8",
    ) as stream:
      stream.write(fixed_xml)
      temporary_path = Path(stream.name)
    model = mujoco.MjModel.from_xml_path(str(temporary_path))
  finally:
    if temporary_path is not None:
      temporary_path.unlink(missing_ok=True)
  model.opt.timestep = timestep_s
  return model


def _is_descendant(model: mujoco.MjModel, body_id: int, root_id: int) -> bool:
  current = body_id
  while current > 0:
    if current == root_id:
      return True
    current = int(model.body_parentid[current])
  return False


@dataclass
class ModelAdapter:
  model: mujoco.MjModel
  mapping: Mapping
  xml_path: Path
  gas_force_n: float
  base_id: int
  can3_joint_id: int
  can2_joint_id: int
  can3_qpos: int
  can2_qpos: int
  can3_dof: int
  can2_dof: int
  hip_site_id: int
  wheel_site_id: int
  loop_site_pairs: tuple[tuple[int, int], ...]
  gas_actuator_ids: tuple[int, int]
  policy_actuator_ids: tuple[int, ...]
  wheel_dofs: tuple[int, int]
  passive_joint_ids: tuple[int, ...]
  passive_qpos: tuple[int, ...]
  passive_dofs: tuple[int, ...]
  validation: dict[str, Any]

  @classmethod
  def create(
    cls,
    xml_path: Path,
    mapping: Mapping,
    timestep_s: float = 0.0025,
    gas_force_n: float = 300.0,
  ) -> "ModelAdapter":
    model = load_fixed_base_model(xml_path, timestep_s)
    base_id = object_id(
      model, mujoco.mjtObj.mjOBJ_BODY, "robot/base_link"
    )
    can3_joint_id = object_id(
      model, mujoco.mjtObj.mjOBJ_JOINT, mapping.can3_joint
    )
    can2_joint_id = object_id(
      model, mujoco.mjtObj.mjOBJ_JOINT, mapping.can2_joint
    )
    can3_qpos = int(model.jnt_qposadr[can3_joint_id])
    can2_qpos = int(model.jnt_qposadr[can2_joint_id])
    can3_dof = int(model.jnt_dofadr[can3_joint_id])
    can2_dof = int(model.jnt_dofadr[can2_joint_id])
    hip_site_id = object_id(
      model,
      mujoco.mjtObj.mjOBJ_SITE,
      f"robot/{mapping.model_side}_virtual_hip_site",
    )
    wheel_site_id = object_id(
      model,
      mujoco.mjtObj.mjOBJ_SITE,
      f"robot/{mapping.model_side}_wheel_axis_site",
    )
    loop_site_pairs = tuple(
      (
        object_id(
          model,
          mujoco.mjtObj.mjOBJ_SITE,
          f"robot/{mapping.model_side}_front_site{index}",
        ),
        object_id(
          model,
          mujoco.mjtObj.mjOBJ_SITE,
          f"robot/{mapping.model_side}_rear_site{index}",
        ),
      )
      for index in (1, 2)
    )
    policy_actuator_ids = tuple(
      object_id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"robot/{suffix}"
      )
      for suffix in POLICY_ACTUATOR_SUFFIXES
    )
    gas_actuator_ids = tuple(
      object_id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"robot/{suffix}"
      )
      for suffix in GAS_ACTUATOR_SUFFIXES
    )
    wheel_dofs = tuple(
      int(model.jnt_dofadr[object_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        f"robot/{side}_Wheel_joint",
      )])
      for side in ("Left", "Right")
    )
    passive_joint_ids = tuple(
      object_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        f"robot/{mapping.model_side}_{suffix}",
      )
      for suffix in PASSIVE_JOINT_SUFFIXES
    )
    passive_qpos = tuple(
      int(model.jnt_qposadr[index]) for index in passive_joint_ids
    )
    passive_dofs = tuple(
      int(model.jnt_dofadr[index]) for index in passive_joint_ids
    )
    adapter = cls(
      model=model,
      mapping=mapping,
      xml_path=xml_path.resolve(),
      gas_force_n=gas_force_n,
      base_id=base_id,
      can3_joint_id=can3_joint_id,
      can2_joint_id=can2_joint_id,
      can3_qpos=can3_qpos,
      can2_qpos=can2_qpos,
      can3_dof=can3_dof,
      can2_dof=can2_dof,
      hip_site_id=hip_site_id,
      wheel_site_id=wheel_site_id,
      loop_site_pairs=loop_site_pairs,
      gas_actuator_ids=gas_actuator_ids,
      policy_actuator_ids=policy_actuator_ids,
      wheel_dofs=wheel_dofs,
      passive_joint_ids=passive_joint_ids,
      passive_qpos=passive_qpos,
      passive_dofs=passive_dofs,
      validation={},
    )
    adapter._validate_and_configure()
    return adapter

  def _validate_and_configure(self) -> None:
    model = self.model
    if mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_JOINT, "robot/base_free_joint"
    ) >= 0:
      raise ValueError("Base free joint is still present")
    if not np.allclose(model.opt.gravity, (0.0, 0.0, -9.81), atol=0.05):
      raise ValueError(f"Unexpected model gravity: {model.opt.gravity}")
    expected_actuators = len(POLICY_ACTUATOR_SUFFIXES) + len(
      GAS_ACTUATOR_SUFFIXES
    )
    if model.nu != expected_actuators:
      raise ValueError(
        f"Expected exactly {expected_actuators} actuators, found {model.nu}; "
        "refusing possible duplicate compensation"
      )
    gas_tendon_ids = []
    for actuator_id in self.gas_actuator_ids:
      if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_TENDON:
        raise ValueError("Gas-spring actuator is not tendon-transmitted")
      if not np.allclose(model.actuator_biasprm[actuator_id], 0.0):
        raise ValueError("Gas-spring actuator has an unexpected bias term")
      if not math.isclose(
        float(model.actuator_gainprm[actuator_id, 0]), 1.0, abs_tol=1e-12
      ):
        raise ValueError("Gas-spring actuator gain is not direct force")
      gas_tendon_ids.append(int(model.actuator_trnid[actuator_id, 0]))
    if np.any(model.tendon_stiffness[gas_tendon_ids] != 0.0):
      raise ValueError("Gas tendon has native stiffness plus force actuation")
    active_joint_ids = (self.can3_joint_id, self.can2_joint_id)
    if np.any(model.jnt_stiffness[list(active_joint_ids)] != 0.0):
      raise ValueError("Active joints have unexpected native spring stiffness")
    equality_names = {
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, index)
      for index in range(model.neq)
    }
    expected_loops = {
      f"robot/{self.mapping.model_side}_loop1",
      f"robot/{self.mapping.model_side}_loop2",
    }
    if not expected_loops <= equality_names:
      raise ValueError("Selected model side does not have both loop constraints")

    robot_geoms = 0
    for geom_id in range(model.ngeom):
      body_id = int(model.geom_bodyid[geom_id])
      if _is_descendant(model, body_id, self.base_id):
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0
        robot_geoms += 1
    model.actuator_gainprm[list(self.policy_actuator_ids)] = 0.0
    model.actuator_biasprm[list(self.policy_actuator_ids)] = 0.0
    self.validation = {
      "base_fixed": True,
      "gravity_m_s2": model.opt.gravity.tolist(),
      "robot_contact_geoms_disabled": robot_geoms,
      "wheel_joints_free": True,
      "closed_chain_equalities": sorted(expected_loops),
      "policy_actuators_disabled": [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in self.policy_actuator_ids
      ],
      "gas_spring": {
        "implementation": "constant tendon actuator force",
        "force_n": self.gas_force_n,
        "native_tendon_stiffness": [
          float(model.tendon_stiffness[index]) for index in gas_tendon_ids
        ],
        "firmware_370p1_times_length_added_in_sim": False,
      },
      "duplicate_compensation_check_passed": True,
      "source_xml": str(self.xml_path),
      "source_xml_sha256": file_sha256(self.xml_path),
      "mujoco_version": mujoco.__version__,
      "timestep_s": float(model.opt.timestep),
    }

  def new_data(self) -> mujoco.MjData:
    data = mujoco.MjData(self.model)
    data.ctrl[list(self.gas_actuator_ids)] = self.gas_force_n
    data.ctrl[list(self.policy_actuator_ids)] = 0.0
    mujoco.mj_forward(self.model, data)
    return data

  def loop_gap_m(self, data: mujoco.MjData) -> float:
    return max(
      float(np.linalg.norm(data.site_xpos[front] - data.site_xpos[rear]))
      for front, rear in self.loop_site_pairs
    )

  def closure_diagnostics(self, data: mujoco.MjData) -> dict[str, object]:
    """Describe the selected leg's geometric closure without fitting it."""
    residual, jacobian = self._closure_residual_and_jacobian(data)
    relevant = np.asarray(
      (self.can3_dof, self.can2_dof, *self.passive_dofs), dtype=np.int64
    )
    reduced = jacobian[:, relevant]
    singular_values = np.linalg.svd(reduced, compute_uv=False)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    threshold = max(1e-10, largest * 1e-8)
    effective = singular_values[singular_values > threshold]
    effective_min = float(effective[-1]) if effective.size else 0.0
    condition = (
      largest / effective_min if effective_min > 0.0 else math.inf
    )
    return {
      "loop_gap_m": self.loop_gap_m(data),
      "closure_residual_norm_m": float(np.linalg.norm(residual)),
      "singular_values": singular_values.tolist(),
      "effective_rank": int(effective.size),
      "structural_zero_count": int(singular_values.size - effective.size),
      "effective_min_singular_value": effective_min,
      "effective_condition_number": condition,
      "structural_zero_threshold": threshold,
      "jacobian_columns": [
        "CAN3",
        "CAN2",
        *[
          mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
          )
          for joint_id in self.passive_joint_ids
        ],
      ],
    }

  def passive_limit_diagnostics(
    self, data: mujoco.MjData
  ) -> dict[str, object]:
    """Return signed distances to all finite passive hinge limits."""
    joints = []
    for joint_id, qpos_id in zip(
      self.passive_joint_ids, self.passive_qpos, strict=True
    ):
      if not self.model.jnt_limited[joint_id]:
        continue
      lower, upper = self.model.jnt_range[joint_id]
      position = float(data.qpos[qpos_id])
      joints.append({
        "joint": mujoco.mj_id2name(
          self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        ),
        "position_rad": position,
        "lower_distance_rad": position - float(lower),
        "upper_distance_rad": float(upper) - position,
        "nearest_distance_rad": min(
          position - float(lower), float(upper) - position
        ),
      })
    nearest = min(
      (float(item["nearest_distance_rad"]) for item in joints),
      default=math.inf,
    )
    return {
      "limited_passive_joints": joints,
      "minimum_signed_distance_rad": nearest,
    }

  def virtual_state(
    self, data: mujoco.MjData
  ) -> tuple[float, float, float, float]:
    model = self.model
    vector_w = (
      data.site_xpos[self.wheel_site_id] - data.site_xpos[self.hip_site_id]
    )
    rotation = data.xmat[self.base_id].reshape(3, 3)
    vector_b = rotation.T @ vector_w
    length = float(np.linalg.norm(vector_b))
    phi = math.atan2(float(vector_b[0]), float(-vector_b[2]))

    jac_hip = np.zeros((3, model.nv))
    jac_wheel = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jac_hip, None, self.hip_site_id)
    mujoco.mj_jacSite(model, data, jac_wheel, None, self.wheel_site_id)
    velocity_w = (jac_wheel - jac_hip) @ data.qvel
    velocity_b = rotation.T @ velocity_w
    length_rate = float(np.dot(vector_b, velocity_b) / max(length, 1e-12))
    planar_sq = float(vector_b[0] ** 2 + vector_b[2] ** 2)
    phi_rate = float(
      (-vector_b[2] * velocity_b[0] + vector_b[0] * velocity_b[2])
      / max(planar_sq, 1e-12)
    )
    return length, phi, length_rate, phi_rate

  def initialize(
    self,
    data: mujoco.MjData,
    can3_qpos_target: float,
    can2_qpos_target: float,
    can3_speed_target: float,
    can2_speed_target: float,
  ) -> dict[str, float]:
    model = self.model
    qpos_ids = np.asarray((self.can3_qpos, self.can2_qpos))
    dof_ids = np.asarray((self.can3_dof, self.can2_dof))
    starts = data.qpos[qpos_ids].copy()
    raw_targets = np.asarray((can3_qpos_target, can2_qpos_target))
    targets = starts + wrap_angle(raw_targets - starts)
    for progress in np.linspace(0.0, 1.0, 41)[1:]:
      data.qpos[qpos_ids] = starts + progress * (targets - starts)
      self._solve_passive_position(data)
    data.qpos[qpos_ids] = targets
    self._solve_passive_position(data)
    position_error = wrap_angle(data.qpos[qpos_ids] - targets)
    if float(np.max(np.abs(position_error))) > 1e-8:
      raise RuntimeError(
        "Closed-chain initialization missed active target: "
        f"target={targets}, actual={data.qpos[qpos_ids]}, "
        f"error={position_error}"
      )

    desired = np.zeros(model.nv)
    desired[dof_ids] = (can3_speed_target, can2_speed_target)
    _, jacobian = self._closure_residual_and_jacobian(data)
    free = np.asarray(self.passive_dofs)
    right_hand_side = -jacobian[:, dof_ids] @ desired[dof_ids]
    desired[free] = np.linalg.lstsq(
      jacobian[:, free], right_hand_side, rcond=None
    )[0]
    constraint_velocity_error = (
      float(np.max(np.abs(jacobian @ desired))) if jacobian.size else 0.0
    )
    if constraint_velocity_error > 1e-5:
      raise RuntimeError(
        "Could not initialize a closure-consistent velocity: "
        f"{constraint_velocity_error}"
      )
    data.qvel[:] = desired
    data.time = 0.0
    mujoco.mj_forward(model, data)
    loop_gap = self.loop_gap_m(data)
    if loop_gap > 2e-4:
      raise RuntimeError(f"Closed-chain site gap is too large: {loop_gap} m")
    return {
      "can3_qpos_error_rad": float(position_error[0]),
      "can2_qpos_error_rad": float(position_error[1]),
      "constraint_velocity_error": constraint_velocity_error,
      "loop_gap_m": loop_gap,
    }

  def _closure_residual_and_jacobian(
    self, data: mujoco.MjData
  ) -> tuple[np.ndarray, np.ndarray]:
    mujoco.mj_forward(self.model, data)
    residuals = []
    jacobians = []
    for front_site, rear_site in self.loop_site_pairs:
      residuals.append(
        data.site_xpos[front_site] - data.site_xpos[rear_site]
      )
      front_jacobian = np.zeros((3, self.model.nv))
      rear_jacobian = np.zeros((3, self.model.nv))
      mujoco.mj_jacSite(
        self.model, data, front_jacobian, None, front_site
      )
      mujoco.mj_jacSite(
        self.model, data, rear_jacobian, None, rear_site
      )
      jacobians.append(front_jacobian - rear_jacobian)
    return np.concatenate(residuals), np.concatenate(jacobians, axis=0)

  def _solve_passive_position(self, data: mujoco.MjData) -> None:
    passive_qpos = np.asarray(self.passive_qpos)
    passive_dofs = np.asarray(self.passive_dofs)
    for _ in range(40):
      residual, jacobian = self._closure_residual_and_jacobian(data)
      norm = float(np.linalg.norm(residual))
      if norm <= 1e-9:
        return
      step = np.linalg.lstsq(
        jacobian[:, passive_dofs], -residual, rcond=None
      )[0]
      maximum = float(np.max(np.abs(step)))
      if maximum > 0.25:
        step *= 0.25 / maximum
      original = data.qpos[passive_qpos].copy()
      accepted = False
      for scale in (1.0, 0.5, 0.25, 0.1):
        data.qpos[passive_qpos] = original + scale * step
        candidate, _ = self._closure_residual_and_jacobian(data)
        if float(np.linalg.norm(candidate)) < norm:
          accepted = True
          break
      if not accepted:
        data.qpos[passive_qpos] = original
        break
    residual, _ = self._closure_residual_and_jacobian(data)
    raise RuntimeError(
      "Could not solve passive closed-chain pose; residual norm="
      f"{np.linalg.norm(residual):.6g} m"
    )

  def apply_logged_torque(
    self,
    data: mujoco.MjData,
    can3_command_nm: float,
    can2_command_nm: float,
  ) -> None:
    data.qfrc_applied[:] = 0.0
    data.qfrc_applied[self.can3_dof] = (
      self.mapping.can3_torque_sign * can3_command_nm
    )
    data.qfrc_applied[self.can2_dof] = (
      self.mapping.can2_torque_sign * can2_command_nm
    )
    data.ctrl[list(self.gas_actuator_ids)] = self.gas_force_n

  def measured_model_state(
    self,
    solver_q1: float,
    solver_q2: float,
    can3_speed: float,
    can2_speed: float,
  ) -> tuple[float, float, float, float]:
    q3, q2 = self.mapping.qpos_from_solver(solver_q1, solver_q2)
    return float(q3), float(q2), can3_speed, can2_speed
