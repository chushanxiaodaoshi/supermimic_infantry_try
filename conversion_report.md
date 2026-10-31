# Infantry MJCF → USD 转换报告

## 结论

已从 mjlab 的**机器人描述文件**生成 Isaac Lab 3 可引用的 USD 包。场景重放文件
`mujoco390/mjcf/scene.xml` 没有作为机器人资产转换源；其中的控制器只作为运行时
控制契约来源。离线检查确认 USD 的 MuJoCo 物理变体包含四个闭链、两个 spatial
tendon、八个 actuator、十四个旋转关节和十五个有质量的机器人刚体。

本机没有完整 Isaac Lab 3 + Newton + MuJoCo-Warp 运行环境，因此没有宣称动态采集
通过。最终的 200 步动态检查必须在云端完成。

## 工具与输入

- 转换源：`mjlab/src/mjlab/asset_zoo/robots/infantry/Fudan-2026RoboMaster-Balance.xml`
- 源 SHA-256：`1c582c371a26b3e6502097748d414ea7eead1ba23f1ef6ee7ebc3b11ec011110`
- 转换器：`isaacsim-asset-isolated 6.2.0.0`
- `mujoco-usd-converter 0.6.1`
- `usd-exchange 3.0.1`
- `newton-usd-schemas 0.6.0`
- MuJoCo 编译验证：`mujoco 3.15.0`
- Python：WSL 隔离环境 `/opt/infantry-usd-importer`，未安装完整 Isaac Sim。
- 网络下载使用 Windows 代理端口 7897；WSL NAT 中实际代理地址为当时的
  `http://172.22.32.1:7897`。

源机器人 XML 同时带有课程场景 body。`tools/prepare_infantry_mjcf.py` 对预期关节、
闭链、tendon 和 actuator 做硬校验后，只提取 `base_link` 机器人子树和它引用的 35
个文件 mesh；移除了课程 body、13 个程序化课程 mesh、15 个课程接触对，以及默认
关闭且依赖场景支撑 body 的 `base_support_weld`。没有修改机器人质量、惯量、质心、
关节参数、碰撞参数、闭链参数或 tendon 路径。

## 输出结构

主文件：

`data/infantry_usd/Fudan-2026RoboMaster-Balance/Fudan-2026RoboMaster-Balance.usda`

主文件 SHA-256：

`608f65fa31e56044b5af88aba1283b362cb5048d481b18097e7fd2a0a99f84f2`

包内共 8 个 USD layer，总大小约 9.27 MB：

交付 ZIP 大小 6,563,888 bytes，SHA-256 为
`ec46e30822402ec43422ff2c2a0e6d2e354cfe8e22cb347f4ebcc4130681cba7`。

```text
Fudan-2026RoboMaster-Balance/
├── Fudan-2026RoboMaster-Balance.usda
└── payloads/
    ├── base.usda
    ├── geometries.usd
    ├── instances.usda
    ├── robot.usda
    └── Physics/
        ├── mujoco.usda
        ├── physics.usda
        └── physx.usda
```

主 layer 的默认 prim 是 `/tn__Fudan2026Balance_vNA`，机器人 articulation root 是
`/tn__Fudan2026Balance_vNA/Geometry/base_link`。运行时选择 `Physics=mujoco`，不是
PhysX 物理变体。所有依赖都是包内相对路径；没有绝对路径或未解析依赖。

## 机器人结构验证

MuJoCo 编译后的机器人模型：

- body：16（含 world），其中 15 个机器人刚体具有有效质量；
- joint：15（含 floating base），`nq=21`、`nv=20`；
- 旋转关节：14；
- mesh：35；geom：41（原 40 个物理/视觉 geom，加一个无质量、无碰撞的机身视觉实例）；
- actuator：8。

十四个旋转关节：

```text
Left_rear_joint
Left_rear_child1_joint
Left_Wheel_joint
Right_rear_joint
Right_rear_child1_joint
Right_Wheel_joint
Left_front_joint
Left_front_child1_joint
Left_front_child2_joint
Left_front_child3_joint
Right_front_joint
Right_front_child1_joint
Right_front_child2_joint
Right_front_joint3_joint
```

四个闭链均存在于 `physics.usda`，并在 `mujoco.usda` 上应用
`MjcEqualityConnectAPI`，原 `solref/solimp` 得到保留：

```text
Left_loop1
Left_loop2
Right_loop1
Right_loop2
```

两个气弹簧均以 `MjcTendon` 存在，类型为 `spatial`，每条 tendon 的两个 site 路径
完整，并各有一个指向 tendon 的 `MjcActuator`：

```text
Left_gas_spring_tendon
Right_gas_spring_tendon
```

`tools/validate_infantry_mjcf.py` 按名称比较提取前后的 body 质量、惯量、质心，关节
限位、阻尼、摩擦，以及 geom 摩擦和接触求解参数，结果完全一致。

云端现象表明 Newton 运行视图可能过滤机身原本唯一的 `base_link_collision` 表示，
因此转换准备阶段另外添加了复用同一 STL 的 `base_link_visual`。它设置为 `mass=0`、`contype=0`、
`conaffinity=0`，不参与质量计算和碰撞；离线验证同时要求该 USD prim 存在且没有
`CollisionAPI`。机身的原碰撞 geom 没有删除或修改。

## 资产与控制器分层

机器人 USD 不使用重放场景中的 PD 配置。Teacher 的动作裁剪、缩放、腿部位置
控制、轮部速度控制、力限制和气弹簧目标仍从重放工程读取。

为避免运行时依赖 USD drive，`extract_mjcf_physics()` 直接解析重放 `scene.xml` 的
现有 actuator 参数，并传给 Isaac Lab 适配层：

- 四个腿部主动关节：位置控制，stiffness 15、damping 1；
- 两个轮关节：速度控制，stiffness 0、damping 0.1；
- effort limit 继续来自 `run_policy.py`；
- tendon 路径属于机器人资产，气弹簧控制目标继续来自 `run_policy.py`。

这些值没有在 SuperMimic 中复制成第二份常量。若重放控制器不再是支持的仿射位置/
速度形式，启动校验会明确报错。

## 已完成验证

- 机器人本体 MJCF 可由 MuJoCo 3.15.0 编译；
- 提取前后机器人动力学参数逐名称相等；
- USD 可由 `pxr.Usd` 打开；
- articulation root、14 个关节、4 个闭链、2 个 tendon、8 个 actuator 存在；
- 15 个刚体质量均有限且为正；
- 8 个依赖 layer 全部可解析且位于包内；
- `Physics=mujoco` 变体存在；
- 重放控制器解析结果已核对；
- `pytest tests -q`：21 passed；
- 相关修改文件 Ruff 检查通过；
- `--steps 200` 已接入配置覆盖并有回归测试。

转换器提示 MJCF sensor 不受支持。当前采集运行时直接从 root body 和 joint state
构造 Teacher 观测，不依赖导入后的 MJCF sensor；这条限制仍记录在此，不将其描述为
“sensor 已转换”。

仓库全量 `pytest` 会错误收集 vendored `IsaacLab/conftest.py` 和需要 Isaac Lab 环境的
`tools/test_*.py`，因此本地使用明确的 `pytest tests -q`。仓库全量 Ruff 还包含上游
G1 文件的既有告警；本次相关文件单独检查通过，没有顺手修改上游核心代码。

## 尚未验证

- 本地没有执行 Isaac Lab 3 + Newton + MuJoCo-Warp 动态步进；
- 未验证 GPU 上闭链长期稳定性和 tendon 实际受力方向；
- 未验证 200 步 Teacher 动作下机器人姿态与原 MuJoCo 重放数值一致；
- 导入器生成的普通 `physics`/`physx` 变体不作为本项目动力学正确性的依据。

## 云端安装与 200 步采集

把 `infantry_usd_package.zip` 上传到 `/workspace/supermimic/` 后执行：

```bash
cd /workspace/supermimic
mkdir -p data/infantry_usd
unzip infantry_usd_package.zip -d data/infantry_usd

test -f data/infantry_usd/Fudan-2026RoboMaster-Balance/Fudan-2026RoboMaster-Balance.usda

/opt/isaaclab-venv/bin/python tools/collect_isaaclab_teacher.py \
  --config configs/infantry/collection/isaaclab_flat.yaml \
  --steps 200
```

配置默认输出目录必须不存在或为空；采集器会拒绝覆盖已有数据。如要看到画面，需要
云端 Isaac Lab 启动器支持非 headless 模式；当前采集步进代码以 `render=False` 运行，
本报告只验收实际物理采集，不把可视化窗口作为已验证功能。
