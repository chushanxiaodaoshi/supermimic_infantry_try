# Isaac Lab 3 多 Teacher 采集适配

本阶段的主采集入口是 `tools/collect_isaaclab_teacher.py`。mjlab 的
`mujoco390/run_policy.py` 仍是 ONNX 观测与控制契约的唯一来源；采集器通过 AST
读取动作关节顺序、缩放和力矩限制，不导入重放程序，也不在本仓库复制一套控制器。

## 已核对的 Teacher

三份模型都经过输入/输出图检查，要求输入 `[1, 150]`、输出 `[1, 6]`：

| skill | 文件身份 | SHA-256 |
|---|---|---|
| flat | `wheelleg-flat/和gym架构一样的非对称PPO平地/2026-09-10_11-31-50.onnx` | `5fbd941f7a30a6ae6be9196e26bbd6a43d6486667008cc315280295377bdf920` |
| rotation | 重放目录中的 `rough.onnx`；按实验约定登记为独立旋转 Teacher | `51932e80245ee235bdafd01f14be1672615560f007ce37cf4fe840e49af9df35` |
| jump | 重放目录中的 `jump.onnx`（元数据指向 jump-v8） | `94624b748f7e28b14b890b2ecd0b243050109c68b3d1c09cda4ab7a112e5109a` |

较新的 flat 导出 `2026-09-10_17-25-15.onnx` 是 162 维输入，不能套用当前
150 维重放历史，采集器会拒绝它。`rough.onnx` 的文件名及源 checkpoint 元数据不能
改变本实验中“独立旋转 Teacher”的身份；采集 manifest 同时保留路径、哈希和原始
ONNX 元数据，避免掩盖来源。

## Isaac Lab 3 适配边界

目标版本固定为 Isaac Lab 3 commit
`46d6cc3d7b115906fad12efd372fbf69f4e3d3ed`，使用 Newton 的 MuJoCo-Warp solver，
控制周期 0.01 s、每个控制步四个 0.0025 s 物理子步。环境关闭动力学随机化，
每条转移保存从重放 MJCF 读取的 73 维固定物理 context。

该版本 Articulation 的默认初始化会对 spatial tendon 抛出
`NotImplementedError`。适配类没有删除 tendon，而是保留 MJCF/USD 中的两根空间
气弹簧，并复用 Isaac Lab 已有的 MJWarp 原生 `mujoco.ctrl` tendon bridge。初始化时
必须同时找到两根气弹簧、它们的原生 actuator，以及四个闭链 equality；任何一项
缺失都会停止采集。六个关节 servo 的刚度与阻尼从导入 USD 继承，动作顺序、缩放、
力矩限制来自重放契约。

当前三份 ONNX 都写死 batch=1，因此配置默认单环境。代码可以对多个环境逐条推理，
但这只保证正确性，不代表高吞吐；云端扩容前应重新导出动态 batch ONNX，或启动多个
单环境采集进程，不能直接修改图形状并假设行为不变。

## 云端准备与命令

先导入原 MJCF/STL，再离线转换 USD：

```bash
python tools/import_infantry_asset.py "$MJLAB_ROOT/src/mjlab/asset_zoo/robots/infantry"
python tools/mjcf_to_usd.py \
  --src data/infantry_description/Fudan-2026RoboMaster-Balance.xml \
  --dst data/infantry_usd
```

只核对本机资产与 Teacher 契约（不需要 Isaac Lab runtime）：

```bash
python tools/collect_isaaclab_teacher.py \
  --config configs/infantry/collection/isaaclab_rotation.yaml --validate-only
```

在 Isaac Lab 3 kit-less 容器中采集：

```bash
python tools/collect_isaaclab_teacher.py \
  --config configs/infantry/collection/isaaclab_flat.yaml --headless
python tools/collect_isaaclab_teacher.py \
  --config configs/infantry/collection/isaaclab_rotation.yaml --headless
python tools/collect_isaaclab_teacher.py \
  --config configs/infantry/collection/isaaclab_jump.yaml --headless
```

输出目录必须不存在或为空。终止前的 `next_state` 先写入 chunk，随后才 reset；manifest
记录代码版本、ONNX/MJCF 哈希、完整有效配置、seed、运行 UUID、耗时和显存峰值。

## 尚未在本机完成的验证

本机没有运行目标 Isaac Lab 3 CUDA 容器，因此这里只完成静态编译、控制/数据单测、
三 Teacher ONNX 与 MJCF 契约检查。必须在云端先各跑一个短 episode，确认 USD importer
确实保留四个 equality、两个 spatial tendon 及原生 actuator，再扩大采集。当前不能
宣称 Isaac Lab 数据已经产生，也不能用 mjlab 冒烟结果代替这项验证。
