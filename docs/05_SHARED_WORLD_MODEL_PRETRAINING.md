# 共享世界模型预训练

入口 `tools/train_world_model.py` 从 YAML 读取全部实际参数。默认云端配置混合 flat、
独立 rotation 和 jump 三个 Isaac Lab 采集目录，jump 权重为 2，权重只改变窗口采样
概率，不把 `teacher_id` 或 `task_id` 当作动力学输入。

训练目标是

```text
s_(t+1) = f(s_t, u_t, ξ)
```

其中 `u_t` 是已经过重放契约裁剪、缩放和控制类型转换的 8 维实际控制输入；奖励和
Teacher 原始动作不进入损失。网络保持现有 `WorldModel`/`StateCodec` 结构，使用多步
自回归误差，只用 train split 拟合输入和增量归一化统计。

数据读取按 `(run_id, episode_id)` 分组，因此不同采集运行中的 episode 编号可以重复。
train/validation/test 分配单位是完整 episode，采样窗口不能跨 episode 或采集运行。
验证窗口使用与训练采样器独立、由 seed 固定的 RNG，每次验证复用同一集合。

验证 JSONL 同时包含总体与各数据源的 1/4/8/16 步误差，误差拆分为 root 位置、姿态、
线速度、角速度、关节位置、关节速度；还记录相对单步的长时误差增长。数据含
`motion_phase` 时会输出阶段统计。当前 Isaac 采集的阶段标签是基于 root 垂直速度的
粗分类：`>0.15 m/s` 为上升，`<-0.15 m/s` 为下降，其余为接触/稳态；没有把它冒充
轮地接触传感器判定。

若配置 `validation.mjcf_path`，验证器把预测的 root pose 与 14 关节位置写回 MuJoCo，
报告四个闭链 equality 的残差。它是诊断项，不回传梯度。任何非有限 loss、梯度或
预测会立即报错。

checkpoint 保存模型（含归一化统计）、优化器、训练进度、Python/Torch/CUDA RNG、
混合采样器与每个数据源采样器状态、完整配置及累计耗时。Resume 会检查数据、模型、
batch、window 和 seed 契约，不重新拟合归一化，也不覆盖已有 checkpoint 或原始
`resolved_config.yaml`。

云端命令：

```bash
export MJLAB_ROOT=/path/to/mjlab
python tools/train_world_model.py --config configs/infantry/world_model_debug.yaml
python tools/train_world_model.py --config configs/infantry/world_model_cloud.yaml
python tools/train_world_model.py \
  --config configs/infantry/world_model_cloud.yaml \
  --resume results/world_model_shared/checkpoint_00005000.pt
```

训练 loss 下降不构成模型完成证明。是否扩大数据量应依据独立 validation 的各技能、
多步、闭链与有限性结果；这些 Teacher 只覆盖自己的动作分布，不能据此宣称模型能
可靠预测任意 Student 动作。
