# Isaac/Newton 平地在线同步闭环

第一版闭环按固定顺序同步执行：Student 在真实 Isaac Lab/Newton 单环境中采集，
Teacher 与截至当前轮的全部 Student 数据混合更新 World Model，Actor 再通过冻结的
World Model 做 32 步 reward-only 更新，最后在独立的真实单环境中评测。这里没有
BC loss，也不会用 Student 动作作为 Actor 的监督标签。

`tools/collect_isaaclab_student.py` 在 reset 前写入终止 transition，并记录实际状态、
实际控制、command、context、历史观测、Student 动作、episode/run id 和数据划分。
每次采集必须使用新的或空目录。编排器在更新 World Model 前会比较 Teacher/Student
的控制与动力学语义；文件迁移产生的路径和 SHA 变化不参与比较，关节、控制器、
timestep、context、闭链和气弹簧等语义变化会直接终止运行。

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
