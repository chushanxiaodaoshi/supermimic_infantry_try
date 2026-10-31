# 独立 ONNX 播放器

仅使用 Python 3.11、MuJoCo 3.9.0、ONNX Runtime 和 pynput。
不导入 mjlab，不使用 PyTorch，不读取训练日志或配置，不接收 PT。

## 启动

```powershell
conda activate mujoco390
cd D:/mujoco390
python run_policy.py --onnx "D:/path/model.onnx"
# 双策略：默认rough；空格准备，车头测距小于触发阈值后跳跃。
python run_policy.py --rough-onnx "./onnx/rough.onnx" --jump-onnx "./onnx/jump.onnx"
```

默认使用本目录 `mjcf/scene.xml`、`mjcf/assets/` 和 `initial_state.npz`。
XML 是一份独立副本，包含原机器人、平面、台阶和斜坡，不会自动生成随机地形。
主场地是12×12m紧凑九宫格，范围x=2..14m、y=-6..6m，每块4×4m，
相邻地块直接拼接；随机起伏移到侧边(0,+4)，含侧边总范围16×12m。
出生点附近仍是平地。无需额外参数。

| 地块中心 | x=4m | x=8m | x=12m |
| --- | --- | --- | --- |
| y=+4m | 连续坡隆起（上坑） | 连续坡凹坑（下坑） | 波浪起伏 |
| y=0m | RM二级台阶 | 17°飞坡 | 空白平坦落地区 |
| y=-4m | 上阶梯金字塔 | 下阶梯金字塔 | 离散障碍 |

经典地形由训练中的同一生成函数离线导出，固定seed=42、difficulty=0.5，
非金字塔地形的高度/坡度范围乘height_scale=3.0，较上一版再翻倍。
上下阶梯金字塔保持stairs_height_scale=1.5，不再增加难度。
仅为紧凑缩小地块、中央平台和部分障碍宽度，并非某次训练地图的原样复制。
波浪路段使用mjlab提供的wave生成器。连续坡隆起/凹坑高差约±57cm；
金字塔单级14.7cm，总高差约±44.1cm。凹坑是真实负高程，未被z=0平面覆盖。
高度场分辨率沿用训练的0.2m，不使用高密度高度场。

RM二级台阶宽2.4m：x=2.4m进入20cm平台，x=3.5m跨35cm入口坎，
进入30cm上层平台，x=4.8m回到20cm平台，x=5.6m回到地面。
入口坎厚12cm、比上层高5cm；反向可以测试下台阶。
17°飞坡保留35cm出口高度，中心位于(8,0)，沿+X可飞出。
飞坡和RM二级台阶的指定尺寸保持不变。
x=10..14m、y=-2..2m为无障碍平坦落地区，之后仍是平地；
训练随机起伏路段改到中心(0,+4)，不再放在飞坡出口正前方。
5/10/20cm单台阶缩窄移至y=-1.65m的侧边测试道。
这些RM障碍按需求构造，不是官方场地尺寸。

自定义台阶和地面拼接在`mjcf/compact_rm_terrain.xml`；经典地形在
`mjcf/classic_assets.xml`、`mjcf/classic_geoms.xml`，生成参数记录在
`mjcf/classic_terrain.json`。旧版`rm_terrain.xml`保留但不再加载。
播放器只加载这些静态XML，不导入mjlab，也不在运行时重复生成地形。
机器人、执行器、策略输入及训练配置不改变。
若要重新生成经典地形，在mjlab环境执行（不用在播放环境安装mjlab）：

```powershell
conda activate mjlab
cd D:/mjlab
python -m export_play_terrain --difficulty 0.5 --seed 42 --height-scale 3.0 --stairs-height-scale 1.5
```

自定义台阶可直接修改`compact_rm_terrain.xml`的`pos`、`size`（半尺寸）。
重新导出或修改XML后，重启播放器生效。
复制整个文件夹到其他位置后，仍可通过任意 ONNX 路径播放。

只支持输入 `[1,150]`、输出 `[1,6]` 的轮腿策略，不支持 162 维。
ONNX 内部的归一化会由模型自身执行，不在播放器中重复归一化。

## 指令与按键

- 仅右侧小键盘8/5：前进/后退；小键盘4/6：左转/右转。默认松开为0。
  默认慢速线速度±1m/s、角速度±2rad/s；同时按住左/右Shift后，
  切换快速线速度±2m/s、角速度±4rad/s，松开Shift立即恢复慢速。
  不要求按键顺序：先Shift后方向键、先方向键后Shift均加速。
  可视化运行时每个控制帧读取Windows左右Shift实时状态，
  避免Shift+小键盘的临时Shift事件影响速度选择。
- 小键盘7/9：升高/降低高度指令；松开保持当前高度。
- 小键盘1：六个电机力矩直接为0，仍保持气弹簧；小键盘2：恢复控制。
  大键盘上方数字1～9不响应；小键盘3未绑定。
  按物理小键盘扫描码识别，支持NumLock开/关，
  Shift改变数字字符或导航VK时也能识别按下和松开；独立导航键不响应。
- 空格：进入 `jumpprepare`，继续 rough 策略及其正常指令。
  车头 `forward_rangefinder_sensor` 有效测距小于 `JUMP_TRIGGER_DISTANCE_M`
  （当前0.7m）时才切换 jump。
  无命中（负值）、NaN 或无穷大不触发；恰好等于阈值也不触发。
  jump 从实际触发时计时，维持 `JUMP_DURATION_S`（当前 0.55 秒）后返回 rough。
  若按空格时已经小于阈值，则当帧直接进入 jump；未按空格不会自动跳。
  需要指定 `--jump-onnx`；单策略模式按空格不会切换。
  按住不连跳，准备中再次按键不取消，跳跃中再次按键不延时、不排队。
  一次准备只触发一次跳跃，再次跳跃需要重新按空格；无传感器时忽略空格。
- 每秒打印一次 obs25、action、左右腿长与闭链间隙；`--no-debug` 可关闭。

默认rough高度指令0.2，可调整范围0.1～0.3；键盘速度设置显式写在
run_policy.py中的SLOW_LINEAR_SPEED、SLOW_YAW_SPEED、FAST_LINEAR_SPEED、
FAST_YAW_SPEED。CLI的--vx/--yaw仍为指令偏置（默认0）。
双策略中jump的yaw和height始终为0，vx沿用当前键盘指令。
返回rough后恢复键盘yaw及保留的rough高度指令。
仅选择ONNX，不读取训练配置或根据模型文件名推断任务。

```powershell
# 高度固定为0，小键盘7/9不会改变高度指令。
python run_policy.py --onnx "D:/path/model.onnx" --height 0 --height-range 0 0

# 指定高度及可调整范围。
python run_policy.py --onnx "D:/path/model.onnx" --height 0.13 --height-range 0.1 0.16

# 更换兼容 XML；不会给自定义 XML 自动套用默认 initial_state.npz。
python run_policy.py --onnx "D:/path/model.onnx" --xml "D:/path/scene.xml"

# 无头检查，不打开窗口或监听按键。
python run_policy.py --onnx "D:/path/model.onnx" --headless --duration 5 --no-debug
```

GUI Control 监视实际电机力矩（N·m），不用于手动控制。
腿电机硬限幅 ±28 N·m、轮子 ±4 N·m；气弹簧固定 300 N。
PD 参数沿用 XML：默认腿 kp=15、kd=1，轮子速度反馈增益 0.1。
物理步长默认 0.0025 s，每 4 步推理一次，即 100 Hz。

## 150 维接口

当前帧 25 维：机身角速度 3、重力投影 3、指令 3（vx/yaw/height）、
主动腿关节位置 4、电机关节速度 6、上一 action 6。
随后为这些观测项各自的 5 帧历史，共 125 维，按观测项分组排列。
初始历史重复当前帧；不包含接触、高程图、机身线速度或跳跃阶段。
每次策略切换，用切换后有效command对应的当前观测填满5帧历史。
不重置机器人状态或上一帧真实action，不增加任何跳跃观测槽。

action 顺序：左 rear、左 front、左轮、右 rear、右 front、右轮。
腿 action ×0.5 是角度目标；轮子 action ×10 是速度目标。
因此仅维度相同还不够，ONNX 的观测顺序、尺度及动作顺序也必须匹配。
自定义 XML 必须保留对应的关节、site 和 PD/速度执行器接口。

## 安装与测试

```powershell
conda activate mujoco390
cd D:/mujoco390
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```
