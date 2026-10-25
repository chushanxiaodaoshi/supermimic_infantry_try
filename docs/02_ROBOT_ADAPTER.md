# Phase 2：双轮足机器人 Adapter 设计与当前验证

## 1. 已确认模型契约

源资产：`Fudan-2026RoboMaster-Balance.xml`。

| 项目 | 实测值 |
| --- | --- |
| `nq / nv` | 21 / 20 |
| 关节 | 1 个 free joint + 14 个 hinge |
| Actor action | 6 |
| 实际控制输入 | 8 |
| 闭链约束 | 4 个 connect equality |
| 气弹簧 | 2 个 spatial tendon + 2 个 tendon motor |
| Actor 单帧观测 | 25 |
| Actor 历史 | 5 × 25 |
| Critic 特权观测 | 64 |
| World Model 完整状态 | 41 |
| 控制周期 | 0.01 s |

源 XML 的 timestep 是 0.002 s；当前 mjlab 任务使用 0.0025 s × 4。旧 flat checkpoint
保存的是 0.005 s × 2。两者控制周期相同，但物理子步不同，正式对照时必须把它们视为
不同仿真配置。

## 2. 关节与控制语义

四个主动腿关节：

- `Left_front_joint`
- `Left_rear_joint`
- `Right_front_joint`
- `Right_rear_joint`

两个轮关节：

- `Left_Wheel_joint`
- `Right_Wheel_joint`

其余八个腿 hinge 为闭链被动关节。完整状态保留全部 14 个 hinge 的位置和速度。

策略动作顺序为左前腿、左后腿、左轮、右前腿、右后腿、右轮。腿动作是默认角度加
`0.5 * action` 的位置目标；轮动作是 `10.0 * action` 的速度目标。气弹簧是独立的
tendon force，名义值左右各 300 N。World Model 因此读取 8 维实际控制，不直接把
6 维策略动作当作物理输入。

## 3. Adapter 边界

`supermimic/adapters/base.py` 规定以下接口：reset、step、actor observation、完整状态、
command、physics context、policy-action→applied-control、状态序列化和恢复。

每条 transition 保存：

- 当前和下一完整状态；
- 当前 Actor observation 与历史；
- 未缩放 command；
- Teacher action；
- 实际控制目标；
- Reward、terminated、truncated；
- 精确仿真字段组成的 physics context；
- timestamp、dt、task id、episode id。

数据集按 episode 分配 train/validation/test，窗口采样不跨 episode。

## 4. 闭链与气弹簧检查

现有 mjlab CUDA 冒烟中实际加载了 MuJoCo-Warp 的 `equality_connect`、
`friction_tendon` 和闭链求解 kernel，证明当前 mjlab 路径没有删除闭链或 tendon。
`get_spec()` 编译后的机器人仍有四个 equality 和两个 tendon。

Isaac Lab 3 路径必须单独验证。当前 Newton 文档说明 USD loop-closing joint 需要排除在
articulation tree 外，MJWarp 再作为 bilateral coupling 求解。离线 MJCF→USD importer
是否在固定的 Isaac Lab commit 上正确生成这些 joint，以及 spatial tendon actuator 是否
完整保留，尚未实测。兼容性失败时不得删除 equality 或 tendon 来换取启动成功。

需要记录的验收量：

- 四组闭链 site 对的最大位置残差；
- equality constraint force/solver residual；
- 左右 tendon 长度、速度和施加力；
- 主动/被动关节的短轨迹误差；
- root pose/velocity、轮接触和总能量差；
- 单环境与多环境有限数值检查。

## 5. 已实现文件

- `supermimic/infantry/spec.py`：真实关节、动作、控制、状态和 context 契约；
- `supermimic/infantry/control.py`：可微 6→8 控制映射；
- `supermimic/infantry/mjcf.py`：MJCF 编译与关键机制检查；
- `tools/import_infantry_asset.py`：从 mjlab 导入 XML/STL 到 gitignored data；
- `tools/mjcf_to_usd.py`：参数化 MJCF→USD，保留原 G1 默认入口；
- `tools/collect_mjlab_teacher.py`：真实 Teacher rollout collector；
- `tools/train_world_model.py`：多步共享 World Model 训练入口；
- `configs/infantry`：4060 debug、4090 train、teacher-guided/free 目标配置。

## 6. 尚未完成

- Isaac Lab 3 `RobotAdapter` 的具体 scene/spawn/control 实现；
- USD 内 equality/tendon 自动检查；
- Isaac Lab 与 mjlab 的固定动作短轨迹数值对照；
- 双轮足解析可微 Reward、soft termination 和真环境评测器；
- Teacher BC、reference tracking 与 imagined return 的完整联合训练 CLI。

这些项目依赖先验证固定 Isaac Lab/Newton 版本对闭链与 spatial tendon 的支持。

