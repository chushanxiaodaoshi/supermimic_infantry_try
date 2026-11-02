# Infantry 平地无 Teacher 策略训练

## 目标与边界

`tools/train_flat_policy.py` 从随机初始化的 Actor/Critic 开始，通过已经预训练并冻结的
Infantry World Model 做短时域可微展开。它不使用参考轨迹、行为克隆、DAgger 或 PPO
Teacher 动作标签。

训练使用 Teacher 数据集的训练集部分提供初始状态、五帧观测历史、速度命令和物理
context。Teacher 动作不进入策略目标。当前入口不会继续更新 World Model，也不会在
训练过程中生成新的真实仿真转移，因此模型外动作的可靠性必须通过独立 Newton 仿真
评测判断。

## 控制和气弹簧

Actor 始终输出六维动作：四个腿部位置控制和两个轮子速度控制。动作顺序、缩放与裁剪
从 `mujoco390/run_policy.py` 读取，不另写一份控制器配置。

已训练 World Model 保持原有八维输入兼容性。后两维气弹簧力从73维物理 context 复制并
停止梯度，整个 imagined rollout 中保持固定。策略不能调节气弹簧，气弹簧仍属于机器人
固有动力学。

## 奖励

解析奖励迁移 mjlab flat 配置中的以下分项：

- 前进速度与偏航速度跟踪；
- 左右腿名义构型与虚拟腿前后倾；
- 横滚/俯仰角速度与机身倾斜；
- 关节速度、加速度、执行器力矩；
- 动作变化率、腿动作二阶平滑和关节限位。

关节加速度用连续预测速度差分计算。执行器力矩根据 context 中记录的 MuJoCo
`gain/bias/forcerange` 做解析仿射计算及裁剪，不增加神经网络预测头。底盘接触力无法从
41维状态严格还原，imagined rollout 使用高度和倾角的可微软终止；真实仿真验收仍应
使用后端接触终止。

## 运行

先复制配置并设置数据集、World Model checkpoint 和全新输出目录，或直接通过 CLI 覆盖
checkpoint：

```bash
python tools/train_flat_policy.py \
  --config configs/infantry/flat_policy_from_world_model.yaml \
  --world-model /workspace/supermimic/logs/infantry_world_model/checkpoint_XXXXXXXX.pt \
  --output /workspace/supermimic/logs/infantry_flat_policy_run01
```

小规模冒烟测试建议先覆盖迭代数和 batch size（batch size 当前从 YAML 读取）：

```bash
python tools/train_flat_policy.py \
  --config configs/infantry/flat_policy_from_world_model.yaml \
  --world-model /path/to/world_model.pt \
  --output /workspace/supermimic/logs/infantry_flat_smoke \
  --iterations 10
```

恢复训练时输出目录保持不变：

```bash
python tools/train_flat_policy.py \
  --config configs/infantry/flat_policy_from_world_model.yaml \
  --world-model /path/to/world_model.pt \
  --resume /workspace/supermimic/logs/infantry_flat_policy_run01/checkpoint_00001000.pt
```

日志中的 `imagined_reward` 上升只说明策略利用了当前 World Model。必须同时监控
`wm_input_rms`；它明显持续增大表示策略正在离开模型训练分布。最终是否可用必须在独立
Newton/MuJoCo-Warp 仿真中验证站立、速度跟踪、跌倒率及完整 mjlab reward。
