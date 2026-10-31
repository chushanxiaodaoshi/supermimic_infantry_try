"""Read-only initial force balance in the closed-chain tangent coordinates.

Reuse saved initial states and stream only selected 100 ms log snippets.
No parameter changes, optimization, or differentiation of measured velocity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .core import (
  Mapping, create_output_dir, file_sha256, forward_kinematics,
  iter_samples, uint32_delta, wrap_angle, write_json,
)
from .model import ModelAdapter, object_id
from .replay import repository_state


def dense_constraint_jacobian(model, data):
  if not mujoco.mj_isSparse(model):
    return data.efc_J.reshape(data.nefc, model.nv).copy()
  result = np.zeros((data.nefc, model.nv))
  for row in range(data.nefc):
    start = data.efc_J_rowadr[row]
    stop = start + data.efc_J_rownnz[row]
    result[row, data.efc_J_colind[start:stop]] = data.efc_J[start:stop]
  return result


def restore(adapter, record):
  data = adapter.new_data()
  for joint in record["joints"]:
    jid = object_id(adapter.model, mujoco.mjtObj.mjOBJ_JOINT, joint["joint"])
    data.qpos[adapter.model.jnt_qposadr[jid]] = joint["qpos"]
    data.qvel[adapter.model.jnt_dofadr[jid]] = joint["qvel"]
  mujoco.mj_forward(adapter.model, data)
  return data


def tangent_basis(adapter, data):
  _, jac = adapter._closure_residual_and_jacobian(data)
  wheel = object_id(
    adapter.model, mujoco.mjtObj.mjOBJ_JOINT,
    f"robot/{adapter.mapping.model_side}_Wheel_joint",
  )
  free = [adapter.can3_dof, adapter.can2_dof,
          int(adapter.model.jnt_dofadr[wheel])]
  basis = np.zeros((adapter.model.nv, 3))
  basis[free] = np.eye(3)
  basis[list(adapter.passive_dofs)] = np.linalg.lstsq(
    jac[:, adapter.passive_dofs], -jac[:, free], rcond=None
  )[0]
  return basis, jac, free


def geometry_jacobian(adapter, data):
  # mj_fwdPosition keeps qvel and force decomposition untouched.
  mujoco.mj_fwdPosition(adapter.model, data)
  rows = []
  for front, rear in adapter.loop_site_pairs:
    ja = np.zeros((3, adapter.model.nv))
    jb = np.zeros_like(ja)
    mujoco.mj_jacSite(adapter.model, data, ja, None, front)
    mujoco.mj_jacSite(adapter.model, data, jb, None, rear)
    rows.append(ja - jb)
  return np.concatenate(rows)


def balance(adapter, record, sample):
  model = adapter.model
  data = restore(adapter, record)
  commands = [sample.value("can3_command_nm"), sample.value("can2_command_nm")]
  adapter.apply_logged_torque(data, *commands)
  basis, jac, free = tangent_basis(adapter, data)
  mass = np.zeros((model.nv, model.nv))
  mujoco.mj_fullM(model, mass, data.qM)
  reduced_mass = basis.T @ mass @ basis
  at_rest = restore(adapter, record)
  at_rest.qvel[:] = 0.0
  mujoco.mj_forward(model, at_rest)
  gravity_bias = at_rest.qfrc_bias.copy()
  constraint_jac = dense_constraint_jacobian(model, data)
  components = {
    "logged_motor_total": data.qfrc_applied.copy(),
    "gas_spring": data.qfrc_actuator.copy(),
    "gravity": -gravity_bias,
    "velocity_bias": gravity_bias - data.qfrc_bias,
    "passive_damping_and_springs": data.qfrc_passive.copy(),
  }
  for label, kinds in (
    ("equality", (0,)), ("friction", (1, 2)),
    ("limits", (3, 4)), ("contacts", (5, 6, 7)),
  ):
    mask = np.isin(data.efc_type, kinds)
    components[label] = constraint_jac[mask].T @ data.efc_force[mask]
  total = sum(components.values())
  projected = {key: (basis.T @ value) for key, value in components.items()}
  net = basis.T @ total
  # Reference approximation only, NOT another force applied to MuJoCo.
  q = data.qpos[free[:2]].copy()
  length_gradient = []
  for column in range(2):
    lengths = []
    for sign in (1, -1):
      pose = q.copy()
      pose[column] += sign * 1e-6
      q1 = (pose[0] - adapter.mapping.can3_qpos_offset) / (
        adapter.mapping.can3_qpos_scale
      )
      q2 = (pose[1] - adapter.mapping.can2_qpos_offset) / (
        adapter.mapping.can2_qpos_scale
      )
      lengths.append(float(forward_kinematics(q1, q2)[0]))
    length_gradient.append((lengths[0] - lengths[1]) / 2e-6)
  compensation = (
    -370.1 * sample.value("measured_leg_length_m") * np.array(length_gradient)
  )
  # Acceleration of the constraint manifold, without measured derivatives.
  epsilon = 1e-6
  jp = []
  for sign in (1, -1):
    probe = restore(adapter, record)
    mujoco.mj_integratePos(model, probe.qpos, data.qvel, sign * epsilon)
    jp.append(geometry_jacobian(adapter, probe))
  jdot_v = (jp[0] - jp[1]) @ data.qvel / (2 * epsilon)
  curvature = np.zeros(model.nv)
  curvature[list(adapter.passive_dofs)] = np.linalg.lstsq(
    jac[:, adapter.passive_dofs], -jdot_v, rcond=None
  )[0]
  ideal_acceleration = np.linalg.solve(
    reduced_mass, net - basis.T @ mass @ curvature
  )
  constraint_acceleration = jac @ data.qacc + jdot_v
  # Gas contribution independently checked via virtual work d(tendon)/d(q).
  gas_fd = []
  gas_id = adapter.gas_actuator_ids[0]
  if adapter.mapping.model_side != "Left":
    gas_id = adapter.gas_actuator_ids[1]
  tendon = int(model.actuator_trnid[gas_id, 0])
  for column in range(2):
    lengths = []
    for sign in (1, -1):
      probe = restore(adapter, record)
      mujoco.mj_integratePos(
        model, probe.qpos, basis[:, column], sign * epsilon
      )
      mujoco.mj_fwdPosition(model, probe)
      lengths.append(float(probe.ten_length[tendon]))
    gas_fd.append(adapter.gas_force_n * (lengths[0] - lengths[1]) / (2*epsilon))
  selected = [*free, *adapter.passive_dofs]
  other = [i for i in range(model.nv) if i not in selected]
  return {
    "window": record["window"], "workpoint": record["workpoint"],
    "phase": record["phase"], "sequence": sample.sequence,
    "L_m": sample.value("measured_leg_length_m"),
    "phi_rad": sample.value("measured_leg_phi_rad"),
    "active_velocity_rad_s": data.qvel[free[:2]].tolist(),
    "projected_Nm_CAN3_CAN2_wheel": {
      key: value.tolist() for key, value in projected.items()
    },
    "net_projected_Nm": net.tolist(),
    "reference_370p1L_compensation_Nm": compensation.tolist(),
    "model_gas_plus_reference_compensation_Nm": (
      projected["gas_spring"][:2] + compensation
    ).tolist(),
    "net_common_sum_Nm": float(np.sum(net[:2])),
    "initial_active_qacc_rad_s2": data.qacc[free[:2]].tolist(),
    "ideal_constraint_qacc_CAN3_CAN2_wheel": ideal_acceleration.tolist(),
    "reduced_mass_kg_m2": reduced_mass.tolist(),
    "loop_gap_m": adapter.loop_gap_m(data),
    "Jv_max_m_s": float(np.max(np.abs(jac @ data.qvel))),
    "JB_max": float(np.max(np.abs(jac @ basis))),
    "acceleration_closure_residual_m_s2": constraint_acceleration.tolist(),
    "full_force_equation_residual_max_Nm": float(np.max(np.abs(
      mass @ data.qacc - total
    ))),
    "gas_virtual_work_check_CAN3_CAN2_Nm": gas_fd,
    "gas_virtual_work_error_max_Nm": float(np.max(np.abs(
      np.array(gas_fd) - projected["gas_spring"][:2]
    ))),
    "other_leg_mass_coupling_max": float(np.max(np.abs(
      mass[np.ix_(selected, other)]
    ))),
    "all_qacc_peak_rad_s2": float(np.max(np.abs(data.qacc))),
    "selected_qacc_peak_rad_s2": float(np.max(np.abs(data.qacc[selected]))),
    "limited_passive": adapter.passive_limit_diagnostics(data),
    "initial_warning_counts": data.warning.number.tolist(),
    "tendon_length_m": float(data.ten_length[tendon]),
    "gas_actuator_force_N": float(data.actuator_force[gas_id]),
  }


def early_motion(adapter, record, snippet):
  data = restore(adapter, record)
  q0 = data.qpos[[adapter.can3_qpos, adapter.can2_qpos]].copy()
  qmeas0 = np.array(adapter.mapping.qpos_from_solver(
    snippet[0].value("solver_q1_rad"), snippet[0].value("solver_q2_rad")
  ))
  initial_v = data.qvel[[adapter.can3_dof, adapter.can2_dof]].copy()
  if np.max(np.abs(wrap_angle(q0 - qmeas0))) > 1e-8:
    raise ValueError("Saved initial state does not match the source CSV")
  result = []
  previous = snippet[0]
  for sample in snippet:
    target = uint32_delta(
      sample.device_time_us, snippet[0].device_time_us
    ) * 1e-6
    while data.time + adapter.model.opt.timestep / 2 < target:
      adapter.apply_logged_torque(
        data, previous.value("can3_command_nm"),
        previous.value("can2_command_nm"),
      )
      mujoco.mj_step(adapter.model, data)
      if np.any(data.warning.number) or not all(np.isfinite(x).all() for x in (
        data.qpos, data.qvel, data.qacc
      )):
        raise RuntimeError(f"Invalid early motion at {record['window']}")
    previous = sample
    mujoco.mj_forward(adapter.model, data)
    qmeas = np.array(adapter.mapping.qpos_from_solver(
      sample.value("solver_q1_rad"), sample.value("solver_q2_rad")
    ))
    if any(abs(target - t) < 0.00002 for t in (0.01, 0.02, 0.05, 0.10)):
      result.append({
        "device_time_s": target,
        "simulated_position_change_rad": wrap_angle(
          data.qpos[[adapter.can3_qpos, adapter.can2_qpos]] - q0
        ).tolist(),
        "measured_position_change_rad": wrap_angle(qmeas - qmeas0).tolist(),
        "simulated_velocity_change_rad_s": (
          data.qvel[[adapter.can3_dof, adapter.can2_dof]] - initial_v
        ).tolist(),
        "measured_velocity_change_rad_s": (
          np.array([sample.value("can3_speed_rad_s"),
                    sample.value("can2_speed_rad_s")]) - initial_v
        ).tolist(),
      })
  return result


def summary_text(results):
  lines = [
    "# PACE 初始力平衡诊断（第三次检查，无参数拟合）", "",
    "复用第二次测试10个初态：8个腿长扫频起点、2个摆角扫频中段。",
    "力矩通过闭链切空间投影到CAN3/CAN2及自由轮轴；同一状态分离",
    "总指令、气弹簧、重力、速度偏置、阻尼、摩擦、约束和限位。", "",
    "|工作点/阶段|L/phi|总指令 Nm|气弹簧 Nm|重力 Nm|净力矩 Nm|初始加速度 rad/s²|",
    "|---|---|---|---|---|---|---|",
  ]
  def pair(values):
    return "/".join(f"{v:.3f}" for v in values[:2])
  for row in results:
    p = row["projected_Nm_CAN3_CAN2_wheel"]
    lines.append(
      f"|{row['workpoint']}/{row['phase']}|"
      f"{row['L_m']:.3f}/{row['phi_rad']:.3f}|"
      f"{pair(p['logged_motor_total'])}|{pair(p['gas_spring'])}|"
      f"{pair(p['gravity'])}|{pair(row['net_projected_Nm'])}|"
      f"{pair(row['initial_active_qacc_rad_s2'])}|"
    )
  lines.extend([
    "", "阶段4为摆角扫频，阶段7为腿长扫频；每对数字依次CAN3/CAN2。",
    "气弹簧力矩已用tendon长度有限差分虚功独立交叉验证。",
    "初态全部无warning，未改变物理参数、约束参数、轮自由度或气弹簧。",
    "", "详细力平衡、约束残差、质量矩阵及10/20/50/100 ms实测对比",
    "保存在initial_balance.json；不对实测速度求导。",
  ])
  return "\n".join(lines) + "\n"


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--diagnosis", type=Path, required=True)
  parser.add_argument("--csv", type=Path, required=True)
  parser.add_argument("--xml", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  records = json.loads((args.diagnosis / "initialization_diagnostics.json")
                      .read_text(encoding="utf-8"))["windows"]
  workpoints = (0, 5, 12, 14, 15, 17, 24, 29)
  records = [r for r in records if (
    r["workpoint"] in workpoints and r["phase"] == 7
    and r["start_label"] == "early"
  ) or (
    r["workpoint"] in (0, 15) and r["phase"] == 4
    and r["start_label"] == "middle"
  )]
  if len(records) != 10:
    raise ValueError(f"Expected ten saved initial states, found {len(records)}")
  sequences = [r["start_sequence"] for r in records]
  snippets = {seq: [] for seq in sequences}
  for sample in iter_samples(args.csv):
    for seq in sequences:
      if seq <= sample.sequence <= seq + 10:
        snippets[seq].append(sample)
  for seq, snippet in snippets.items():
    if len(snippet) != 11 or snippet[0].sequence != seq:
      raise ValueError(f"Incomplete snippet at {seq}")
    for previous, current in zip(snippet, snippet[1:]):
      if (current.sequence != previous.sequence + 1
          or not 0 < uint32_delta(
            current.device_time_us, previous.device_time_us
          ) <= 50000):
        raise ValueError(f"Discontinuous snippet at {seq}")
  out = create_output_dir(args.out)
  adapter = ModelAdapter.create(args.xml, Mapping())
  results = []
  for record in records:
    snippet = snippets[record["start_sequence"]]
    item = balance(adapter, record, snippet[0])
    item["first_100ms"] = early_motion(adapter, record, snippet)
    item["command_peak_to_peak_100ms_Nm"] = [
      float(np.ptp([s.value(f"can{can}_command_nm") for s in snippet]))
      for can in (3, 2)
    ]
    results.append(item)
  write_json(out / "initial_balance.json", {
    "schema": "pace-initial-balance-v1", "states": results,
    "basis": "dq = B * [dq_CAN3, dq_CAN2, dq_wheel]; Q = B.T * qfrc",
    "assumptions": [
      "Saved active/passive/wheel initial states from second test reused.",
      "Wheel and opposite-leg unrecorded velocities remain zero.",
      "Free base removed; no contacts; original 300 N gas actuators.",
      "No parameter change or measured-velocity differentiation.",
      "100 Hz command snapshots held through each device interval.",
      "Ideal acceleration includes finite-difference constraint curvature.",
    ],
    "model": adapter.validation,
    "source_csv": str(args.csv), "source_diagnosis": str(args.diagnosis),
    "source_csv_sha256": file_sha256(args.csv),
    "source_initialization_sha256": file_sha256(
      args.diagnosis / "initialization_diagnostics.json"
    ),
    "analysis_script_sha256": file_sha256(Path(__file__)),
    "repository": repository_state(args.xml.resolve().parents[2]),
    "optimization_performed": False,
  })
  (out / "balance_table.md").write_text(
    summary_text(results), encoding="utf-8"
  )
  for row in results:
    p = row["projected_Nm_CAN3_CAN2_wheel"]
    compact = lambda x: "/".join(f"{v:.3f}" for v in x[:2])
    print(
      f"wp{row['workpoint']:02d} phase{row['phase']} "
      f"L/phi={row['L_m']:.3f}/{row['phi_rad']:.3f} "
      f"motor={compact(p['logged_motor_total'])} "
      f"gas={compact(p['gas_spring'])} grav={compact(p['gravity'])} "
      f"net={compact(row['net_projected_Nm'])} "
      f"qacc={compact(row['initial_active_qacc_rad_s2'])}"
    )
  print(f"output: {out}")


if __name__ == "__main__":
  main()
