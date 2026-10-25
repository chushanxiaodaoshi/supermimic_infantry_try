# 独立复核与最小修正

本轮只修改新增的数据管线、Teacher 采集器、测试与文档。上游 G1 训练代码、
通用 World Model/BPTT 算法以及现有 mjlab 源码均不修改。

## 修复内容

1. **数据覆盖和残留混入**：ChunkWriter 拒绝非空目录，chunk 使用独占创建；
   ChunkDataset 按 manifest 校验文件集合与转移数量。中断或损坏的数据集不会被
   静默当作完整数据集加载。目前不支持追加采集或自动修复旧数据。
2. **未观测动力学扰动**：采集配置关闭 `push_robots`、`body_torque`，每步检查
   external wrench 为零；`body_mass_inertia` 的惯量缩放范围设为 `(1, 1)`，保留
   context 已记录的基座 added mass；`wheel_friction` 和 `restitution` 仅随机化
   左右轮碰撞体。其余刚体质量/惯量及接触参数保持固定。context 在 step 前读取。
   配置变更和 seed 写入 manifest 的 `collection_profile`。
3. **BPTT 测试假阳性**：奖励仅依赖预测状态，检查 Actor 梯度有限且参数实际更新；
   同时注入全部预测状态 detach 的故障，确认奖励失去 Actor 梯度。
4. **CPU 缓存别名**：append 时强制拷贝到 CPU，后续原地更新不再修改历史记录。
5. **窗口采样种子**：每个数据集使用独立 `random.Random(seed)`，训练 CLI 显式
   传入 seed；采集 seed 同时写入 mjlab 环境配置。
6. **历史观测排列**：mjlab 先对每个观测项展开历史，再拼接各项；直接 reshape
   为 `(batch, 5, 25)` 会混淆时间与字段。采集器现先拆开六个观测项，再恢复逐帧、
   从旧到新的历史顺序，并将布局写入 manifest；添加已知数值的排列回归测试。

同时让实际执行动作遵守原 RSL-RL wrapper 的 clipping 设置；`teacher_action`
仍记录 Teacher 原始输出，`applied_control` 记录裁剪和缩放后的实际目标。

## 实验解释与未完成部分

当前模型没有地形输入，因此采集器仅开放平地任务并检查 plane 地形。
模型 41 维状态、8 维控制和 73 维 context 保持不变，不需要迁移网络结构。
这些状态是刚体运动状态，并非含求解器内部状态的精确可重放快照。

本轮收窄采集分布以匹配模型输入，不代表原随机扰动 PPO 训练条件下的性能。
扩大到 rough/jump 或恢复未记录随机化前，需要扩展状态/参数输入并重新验证。
旧数据保持原样；新实验需使用带 `flat_observed_dynamics_v1` 配置记录的新数据。

YAML 预设仍未接入 CLI；Resume、独立验证集评估、Isaac Lab 具体 Adapter 和
Student 完整联合训练均尚未实现。这些功能与本轮五项正确性修复分开推进。

## 本轮验证结果

- 14 项单元/回归测试通过，Ruff 和 Pyright 检查通过。
- 使用现有 flat `model_2700.pt`，2 环境 × 750 控制步 CUDA 采集，得到 1,500 条
  有限数值转移、11 个 episode id；核对逐帧历史末帧动作与 Actor 当前观测一致。
- 新数据保存于 gitignored `data/review_fix_teacher_flat_v2`。较早的
  `data/review_fix_teacher_flat` 在历史排列修正前生成，仅保留作调试记录。
- 新数据上运行两次相同 seed 的 CUDA World Model 冒烟：4 步预测、batch 8、
  hidden `[32, 32]`、3 iteration；两次训练日志及模型权重完全相同。
- 以上结果验证修复和流程，不代表 World Model 已收敛，也不是策略性能验收。
