# Isaac/Newton 平地在线同步闭环

闭环按固定顺序同步执行：当前 Actor 在真实 Isaac Lab/Newton 中主动采集，有限容量
的循环回放更新 World Model，Actor 再通过冻结的 World Model 更新，最后在独立真实
仿真中评测。Teacher 数据只保留为低权重动力学锚点，不再作为主导训练分布。这里没有
BC loss，也不会用 Actor 动作作为监督标签。

`tools/collect_isaaclab_student.py` 在 reset 前写入终止 transition，并记录实际状态、
实际控制、command、context、历史观测、Student 动作、episode/run id 和数据划分。
每次采集必须使用新的或空目录。编排器在更新 World Model 前会比较 Teacher/Student
的控制与动力学语义；文件迁移产生的路径和 SHA 变化不参与比较，关节、控制器、
timestep、context、闭链和气弹簧等语义变化会直接终止运行。

## 主动回放架构

每轮采集执行的是当前 Actor 动作，并在归一化动作空间加入可复现的高斯残差；残差
默认保持10个控制步（0.1秒），使扰动能够产生连续动力学响应，而不是逐帧相互抵消。
命令默认每500步（5秒）在不 reset 物理状态的情况下重新采样。数据同时保存 Actor
原始动作、探索残差、最终执行动作和8维实际控制。

训练混合不再累计所有历史 Actor 数据。`replay.recent_actor_cycles` 限制活跃 Actor
分片数量，最新分片权重最高，旧分片按 `actor_recency_decay` 衰减并最终退出活跃集合；
Teacher 权重统一乘 `teacher_anchor_scale`。旧采集目录仍保留用于审计，但不会无限
支配当前 World Model。

## 不确定性与终止模型

在线配置使用一个确定性主动力学模型和两个独立影子模型。三个成员共享输入/输出
归一化，但每次更新分别抽取训练窗口；主模型仍提供 Actor 的可微梯度，影子模型只
用于度量归一化状态增量的 epistemic disagreement。这样不会用随机输出替代可控的
确定性仿真动力学，同时可以识别 Actor 正在利用的数据覆盖盲区。

World Model 还从真实 `next_state` 与 `terminated` 标签学习终止概率。它不使用奖励，
也不把手写倾角阈值当成监督。验证报告在各数据源、各 horizon 下同时输出动力学误差、
模型分歧以及终止 BCE/Brier/正例召回率。旧的单模型 checkpoint 可直接作为新运行的
`--init-from`：主模型物理函数经归一化重参数化保持不变，影子模型由主模型加微小扰动
初始化，终止头从真实在线数据开始学习。Resume 则严格恢复全部成员、终止头、优化器、
归一化和随机状态。

Actor 的 32 步想象同时使用这两个信号：真实终止头与手写物理跌倒边界取较保守值；
模型分歧一方面作为显式可微成本，另一方面经平滑 trust gate 缩短有效想象范围。gate
在 Actor 更新时停止梯度，防止策略故意进入未知区域来屏蔽负奖励；Actor 只能通过显式
分歧成本学习回到数据支持区域。单成员旧模型的 disagreement 为零、trust 为一，因此
旧 checkpoint 的行为保持兼容，但在线配置默认启用三个成员。

`uncertainty_threshold` 使用的是归一化状态增量的 RMS 分歧，不是米、弧度等物理单位。
当前 0.15 是首轮云端校准起点，不是已由本地动态实验确认的常数。训练日志输出
`wm_ensemble_disagreement` 与 `wm_trust`；应结合独立真实 survival gap 调整阈值，不能
单独追求 trust 接近一。

小规模回归命令：

```bash
/opt/isaaclab-venv/bin/python tools/run_flat_online_cycle.py \
  --config configs/infantry/online_flat_sync_smoke.yaml \
  --actor /path/to/current_actor_checkpoint.pt \
  --output logs/infantry_flat_online_sync_NEW
```

该 smoke 配置只执行 1 轮：400 个真实 Student transition、10 次 World Model
更新、20 次 Actor 更新和 200 步单环境评测。产物写在每轮独立目录，
`online_manifest.json` 记录命令、阶段状态、checkpoint 和评测指标。若只检查计划，
添加 `--dry-run`。当前 smoke 跑通只证明闭环实现可用；跌倒率等真实评测指标没有
改善时，不能把训练 loss 或模型内 reward 上升表述为策略成功。

## 首轮 smoke 记录

2026-11-02 的端到端 smoke 完成了 400 条 Student 真实 transition，数据均为有限值，
共 18 个 episode，train/validation/test transition 数为 319/45/36。Teacher 与
Student 的动力学/控制语义检查通过。Student 验证集上的 World Model 全状态 RMSE
为 1 步 3.36、4 步 4.35、8 步 6.31；独立真实评测 200 步内完成 9 个 episode，
9 个均跌倒，平均长度 21 步，前向速度 MAE 0.646、yaw MAE 0.077。因此本轮验收的是
同步在线管线和数据契约，不是平地策略性能；没有继续启动长任务。

## 五轮有效性实验

`configs/infantry/online_flat_sync_5x.yaml` 每轮采集 2,000 步、更新 World Model
500 次、更新 reward-only Actor 500 次，并做 1,000 步单环境真实评测，共 5 轮。
第 2 轮起，采集器严格使用上一轮 Actor checkpoint；采集后会核对 checkpoint SHA，
并检查 6 维 Student/执行动作、8 维实际控制以及终止 reset 前的 41 维 next_state。

World Model 对每个 Teacher/Student 数据源分别报告 1/4/8/16/32 步验证误差。
短于 32 步的 episode 不会跨 reset 拼接；没有合法窗口时报告 `unavailable`。真实评测
还会从每个 episode 的同一初始状态、历史和 command 做一次模型内 32 步 rollout，
因此 `model_real_survival_gap_absolute` 与真实 survival 可以直接比较。逐轮摘要保存在
`online_manifest.json` 的 `summary` 中。

```bash
/opt/isaaclab-venv/bin/python tools/run_flat_online_cycle.py \
  --config configs/infantry/online_flat_sync_5x.yaml \
  --actor /path/to/current_actor_checkpoint.pt \
  --output logs/infantry_flat_online_sync_5x_NEW
```

第一道验收线是：真实平均 episode 长度持续上升、模型/真实 32 步 survival 差距持续
缩小、Student 32 步预测误差在出现合法窗口后持续下降。若 3–5 轮后寿命仍停留在
20 多步，应转向初始状态覆盖、显式终止成本或真实数据覆盖设计，不继续机械增加轮数。
