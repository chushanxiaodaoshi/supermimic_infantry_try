"""Verified dimensions and names from the existing mjlab infantry task."""

from __future__ import annotations

from supermimic.core.spec import RobotSpec

INFANTRY_JOINT_NAMES = (
    "Left_rear_joint",
    "Left_rear_child1_joint",
    "Left_Wheel_joint",
    "Right_rear_joint",
    "Right_rear_child1_joint",
    "Right_Wheel_joint",
    "Left_front_joint",
    "Left_front_child1_joint",
    "Left_front_child2_joint",
    "Left_front_child3_joint",
    "Right_front_joint",
    "Right_front_child1_joint",
    "Right_front_child2_joint",
    "Right_front_joint3_joint",
)

# Exact concatenation order of the four mjlab action terms.
POLICY_ACTION_NAMES = (
    "Left_front_joint",
    "Left_rear_joint",
    "Left_Wheel_joint",
    "Right_front_joint",
    "Right_rear_joint",
    "Right_Wheel_joint",
)

# World-model inputs represent simulator controls after scaling/offset.  Gas
# spring effort is included even though it is not produced by the policy.
APPLIED_CONTROL_NAMES = (
    "Left_front_position_target",
    "Left_rear_position_target",
    "Left_Wheel_velocity_target",
    "Right_front_position_target",
    "Right_rear_position_target",
    "Right_Wheel_velocity_target",
    "Left_gas_spring_force",
    "Right_gas_spring_force",
)

CONTROL_TYPES = (
    "position",
    "position",
    "velocity",
    "position",
    "position",
    "velocity",
    "tendon_force",
    "tendon_force",
)

# These fields are direct simulator values.  They deliberately avoid lossy
# labels such as "motor strength scale", which cannot be recovered after
# several randomizers have modified the same gain matrix.
CONTEXT_NAMES = (
    "left_wheel_friction",
    "right_wheel_friction",
    "base_mass",
    "base_inertia_x",
    "base_inertia_y",
    "base_inertia_z",
    "base_ipos_x",
    "base_ipos_y",
    "base_ipos_z",
    *(f"{name}:gain" for name in POLICY_ACTION_NAMES),
    *(f"{name}:position_bias" for name in POLICY_ACTION_NAMES),
    *(f"{name}:velocity_bias" for name in POLICY_ACTION_NAMES),
    *(f"{name}:force_min" for name in POLICY_ACTION_NAMES),
    *(f"{name}:force_max" for name in POLICY_ACTION_NAMES),
    "left_gas_spring_force",
    "right_gas_spring_force",
    *(f"{name}:damping" for name in INFANTRY_JOINT_NAMES),
    *(f"{name}:frictionloss" for name in INFANTRY_JOINT_NAMES),
    "left_wheel_solref_0",
    "left_wheel_solref_1",
    "right_wheel_solref_0",
    "right_wheel_solref_1",
)

INFANTRY_SPEC = RobotSpec(
    name="fudan_2026_balance_infantry",
    joint_names=INFANTRY_JOINT_NAMES,
    policy_action_names=POLICY_ACTION_NAMES,
    control_names=APPLIED_CONTROL_NAMES,
    control_types=CONTROL_TYPES,
    context_names=CONTEXT_NAMES,
    # ang vel(3), gravity(3), command(3), active-leg q(4), all motor qd(6), action(6)
    actor_obs_dim=25,
    # Saved mjlab task: base v(3) + actor(25) + two older actions(12)
    # + qdd(6) + motor force(6) + mass(1) + CoM(3) + q0 offsets(6)
    # + wheel friction(2) + restitution(1)
    critic_obs_dim=64,
    command_dim=3,
    control_dt=0.01,
)

INITIAL_JOINT_POS = {
    "Left_rear_joint": 0.0,
    "Left_rear_child1_joint": 0.03192823492459914,
    "Left_Wheel_joint": 0.0,
    "Right_rear_joint": 0.0,
    "Right_rear_child1_joint": -0.03192823492459914,
    "Right_Wheel_joint": 0.0,
    "Left_front_joint": 0.0,
    "Left_front_child1_joint": 1.005950752066677e-09,
    "Left_front_child2_joint": -4.451650184898577e-05,
    "Left_front_child3_joint": 9.822788156748334e-05,
    "Right_front_joint": 0.0,
    "Right_front_child1_joint": -1.005950752066677e-09,
    "Right_front_child2_joint": 4.451650184898577e-05,
    "Right_front_joint3_joint": -9.822788156748334e-05,
}

LOOP_EQUALITY_NAMES = ("Left_loop1", "Left_loop2", "Right_loop1", "Right_loop2")
GAS_SPRING_TENDON_NAMES = ("Left_gas_spring_tendon", "Right_gas_spring_tendon")
