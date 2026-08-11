# OMG BUMI 原生模型接入 GMT 完整说明

本文档说明 BUMI 原生 OMG 模型如何实时生成动作、通过 Redis 向 GMT
传递参考轨迹，以及 GMT 如何把参考轨迹转换为 Gazebo 或实物 BUMI
的关节命令。

本文档基于以下实际代码：

- OMG：`/home/weili/OMG`，分支 `feat/bumi-native-omg`。
- GMT：`/home/weili/docker_projects/bumi_GMT_deployment_listao/bumi_GMT_deployment_listao`。
- GMT policy：`0724_lab_148500.onnx`，21 个关节。

## 1. 两代链路的区别

### 1.1 旧链路：GENMO/OMG G1 + GMR + GMT

旧链路生成的是 G1 动作，GMT 控制的却是 BUMI，因此中间必须经过 GMR：

```text
GENMO/OMG G1 qpos36
       ↓ Redis JSON: omg_online_frame_g1
GMR G1 → BUMI IK 重定向
       ↓ BUMI qpos28
GMR Redis Publisher
       ↓ legacy 35-float: gmt_online_frame_bumi
GMT policy
       ↓ 21 joint actions
Gazebo / 实物 BUMI
```

legacy packet 只表示一帧：

```text
timestamp                         1
root position world              3
root quaternion wxyz             4
root linear velocity world       3
root angular velocity world      3
joint position                  21
----------------------------------
total                           35 float32
```

单帧数据不能表达 GMT policy 所需的 `t-10 ... t+10` 时间窗。旧 GMT
实现只能在 command window 中重复这一帧。

### 1.2 新链路：BUMI 原生 OMG + GMT

现在 OMG 模型直接生成 BUMI qpos28，不再需要 G1 和 GMR：

```text
text / WAV / stand
       ↓ ZMQ JSON command
BUMI OMG Bridge
       ↓ ZMQ request: BUMI 已执行 history [10,28]
BUMI ONNX Planner
       ↓ BUMI plan [60,28] @ 30 Hz
Bridge 重采样、混合、缓冲
       ↓ trajectory_v1 [110,55] @ 50 Hz
Redis: gmt_online_frame_bumi
       ↓ GMT 取 t-10 ... t+10
GMT command_window [1,1092]
       ↓ GMT policy actions [1,21]
Gazebo / 实物 BUMI
```

新链路不运行 `run_g1_bumi3.sh`，也不运行旧 GMR Redis publisher。同一
Redis key 只能有一个运动发布者。

## 2. GMT 历史修改和本轮修改

GMT 工作区在本轮 OMG 接入开始前已经是 dirty worktree，且旧改动没有
独立 commit。因此下面的区分依据是“本轮开始时已存在”和“本轮新增”，
不是根据 Git commit 猜测。

### 2.1 GENMO/GMR/GMT 时期已经对 GMT 做过的改动

旧改动不是“只配了一个 Redis key”，而是完整的 BUMI 部署适配：

1. **BUMI Gazebo 与机器人模型**

   - 新增 `legged_robot/bumi` 机器人描述、URDF、meshes 和 Gazebo 配置。
   - 新增 `legged_robot/bumi_hw` 实物硬件层。
   - 新增 `bumi_empty_world.launch`、`bumi.yaml`、`bumi_ac.yaml`、
     `bumi_kd.yaml`。
   - 加入 BUMI walk/dance/GMT policy 和离线 NPZ 动作。

2. **启动方式由 E1 扩展到 BUMI**

   - `simulation.sh` 和 `real.sh` 的默认 `ROBOT_TYPE` 从 `e1` 改为
     `${ROBOT_TYPE:-bumi}`。
   - `ac_start.launch`、`ac_start_real.launch`、`load_ac_controller.launch`
     增加 BUMI/E1 分支。
   - BUMI 默认读取 `gmt_online_frame_bumi`和
     `0724_lab_148500.onnx`。

3. **控制器从固定 E1 关节扩展到 BUMI 21DOF**

   - `RLControllerBase` 增加 policy joint name 到硬件 joint name 的映射。
   - 根据 `robot_type` 选择 BUMI 21 关节或 E1 关节。
   - 站立角可从 policy ONNX `default_joint_pos` 初始化。
   - 关节状态、发布器和动作维度由固定 24 扩展为动态维度。

4. **GMT policy 加载泛化**

   - `AcController` 读取 ONNX input/output shape 和 metadata。
   - 根据 policy 推导 21D action、69D current policy input、690D history
     和 1092D command window。
   - 读取 `action_scale`、`joint_stiffness`、`joint_damping`、
     `default_joint_pos` 和 `joint_names`。
   - policy action 仍按
     `position_desired = action * action_scale + default_joint_pos`，然后通过原
     PD stiffness/damping 发送给 Gazebo/实物。

5. **离线动作输入泛化**

   - `MotionLoaderNPZ` 不再强制 24 关节。
   - 增加 `joint_vel`，缺失时由 joint position 差分得到。
   - 增加离线真实时间窗构造。

6. **legacy Redis 在线模式**

   - `MotionLoaderRedis` 读取 GMR 发布的 35-float BUMI 单帧。
   - 处理 root 位置、wxyz 四元数、速度、21 关节和首帧 XY/yaw 对齐。
   - Redis 地址、key 和 0.2 秒 timeout 通过 ROS 参数传入。

这些改动才使得旧 GMR 输出能够被 BUMI GMT policy 接收。

### 2.2 BUMI 原生 OMG + GMT 本轮新增的 GMT 改动

本轮只增加了新协议和真实窗口所需的最小接入：

1. `GmtTrajectoryProtocol.h`

   - 实现 `trajectory_v1` 二进制协议。
   - 校验 magic、version、尺寸、finite、root quaternion、CRC32
     和 GMT joint-order SHA256。
   - 从 21 个不同时间帧构造 21×52D feature。

2. `MotionLoaderRedis.h`

   - 自动识别 legacy 35-float 和 `trajectory_v1`。
   - 新增 `protocolKind()`、`sequence()`、`commandWindow()`、
     `hasFreshData()`。
   - 重复 Redis GET 同一 packet 时不重算速度。
   - 重复或倒序 sequence 不覆盖当前 policy window。
   - 保留 legacy 读取路径。

3. `AcController.cpp`

   - 创建 Redis loader 时传入 GMT policy `joint_names`。
   - 把“循环 21 次复制当前帧”替换为
     `motionRedisGmt_->commandWindow(10, 52)`。
   - 只对 `trajectory_v1` 启用 sequence 0.2 秒不更新则回 `DEFAULT`；
     legacy 保持兼容。

4. 测试和文档

   - CMake 增加 `test_gmt_trajectory_protocol`。
   - 新增 C++ 协议、CRC、hash、真实窗口和 Redis 交叉语言测试。
   - 新增 `docs/omg_bumi_trajectory_v1.md`。

本轮没有修改 GMT policy ONNX、action scale、PD、remote-B、emergency
stop、fall protection 或 `simulation.sh`。`simulation.sh` 显示为 modified 是本轮
开始前的历史改动。

## 3. OMG 输入语义

### 3.1 无命令和 `stand`

无命令、启动、`stand`、音乐结束或 Planner 失败时，Bridge 直接发送固定
BUMI qpos28，绝不请求 Planner 生成 `text: stand still`。

固定关节从 GMT policy ONNX 的 `default_joint_pos` 读取。当前 policy 关键值为：

```text
left/right hip pitch      -0.149 rad
left/right knee pitch     +0.322 rad
left/right ankle pitch    -0.172 rad
left shoulder roll        +0.300 rad
right shoulder roll       -0.300 rad
```

root z 由 BUMI sole proxy 自动落地。实际校验 root z 为约 `0.468382 m`，
最低足底代理点误差约 `-2.4e-8 m`。固定轨迹的根速度和关节速度为零。

回站时保留当前 root XY 和 yaw，不把机器人转回启动方向。root
roll/pitch、root z 和关节至少用 1 秒平滑恢复，默认最大关节恢复速度
`1.5 rad/s`。

### 3.2 文本

```text
text walk forward slowly
```

文本作为持续条件，Bridge 滚动生成，直到收到下一个 text/audio/stand。
新命令不清空当前轨迹；等新 plan 返回后用约 0.2 秒混合进入。

文本要求：

- 非空。
- 暂不允许换行或 `|`。
- 英文更接近当前 OMG 训练语料。

### 3.3 WAV 音乐

```text
audio /absolute/path/music.wav
```

只接受存在的绝对 `.wav` 路径。音乐时间轴从“第一帧音乐动作真正
发送给 GMT”开始，不从命令接受或 diffusion 开始时刻计时。

- WAV 推理期间保持固定站立。
- 第一帧音乐动作与 `ffplay` 在同一 50 Hz tick 启动。
- 音乐按 WAV 真实时长执行，不循环。
- 自然结束后平滑回固定站立。
- 新 text、stand 或新 audio 会终止旧 `ffplay`。
- 不加 `--play-audio` 时仍会生成音乐动作，但电脑不播放声音。

## 4. Planner 的输入和输出

### 4.1 历史输入

Planner 接收的不是“当前帧重复 10 次”，而是 Bridge 实际已经发送的
BUMI 历史：

```text
qpos_history: [10, 28] @ 30 Hz
```

qpos28 结构：

```text
root xyz                3
root quaternion wxyz    4
BUMI joint position    21
```

Bridge 从 50 Hz 实际执行参考中按时间采样出 30 Hz 的 10 帧。因此上一次
混合、回站或旧 plan 的真实输出都会进入下一次规划历史。

Planner 使用 BUMI representation/kinematics 把 qpos 编码为：

```text
history_features: [1, 10, 93]
```

93D 表示为 root local position 3、root rot6d 6、21 joint DOF 和
21 feature-body position。必须使用训练时匹配的 `bumi_93d_stats.json`。

### 4.2 文本、音频和 humanref 条件

当前 BUMI ONNX 图的条件输入包括：

```text
text_context       [2, 50, 768]
text_mask          [2, 50]
audio_features     [2, 60, 35]
audio_mask         [2, 60]
human_motion       [2, 60, 66]
human_motion_mask  [2, 60]
```

batch 2 用于 CFG 的 conditional/unconditional 联合推理。实时命令服务当前对外开放
text、WAV 和 stand；没有提供的模态使用 null condition。

### 4.3 Planner 输出

每次返回：

```text
qpos: [60, 28]
fps: 30
duration: 2 seconds
```

diffusion 使用 50 个 DDIM sampling step。Bridge 对 root position 和 joint position
做线性插值，对 root quaternion 做 SLERP，得到精确的：

```text
[100, 28] @ 50 Hz
```

## 5. 滚动规划、缓冲和切换延迟

默认参数：

```text
planner_frames=60 @ 30 Hz             2.0 s
tracker plan=100 @ 50 Hz              2.0 s
replan_remaining_frames=60            1.2 s remaining
```

一个 plan 开始后先执行 40 个 50 Hz frame，即约 0.8 秒，然后发起下一次
replan。此时旧 buffer 还有 60 frame，即约 1.2 秒的推理裕量。

因此“实时”不要求 diffusion 每 20 ms 生成一帧；它只需在旧 buffer 耗尽前
生成下一段 2 秒未来。默认设置的硬连续条件约为：

```text
rolling planning latency < 1.2 seconds
```

工程上建议 P95 小于 0.8 秒，保留调度、T5、Redis 和 GPU 抖动裕量。

新命令到达后：

- 如果没有 pending request，立即规划。
- 如果已有 pending request，等它返回后丢弃过时结果，立即请求最新命令。
- 等待期间保留当前 plan 或固定站立，不清空 buffer。
- 新 plan 生效时默认用 0.2 秒进入混合。

## 6. Redis `trajectory_v1`

### 6.1 packet 帧布局

每个 packet 含 110 帧：

```text
packet[0:10]      t-10 ... t-1
packet[10]        current t
packet[11:110]    t+1 ... t+99
```

GMT 的 21 帧 command window 严格使用：

```text
packet[0:21] = t-10 ... t ... t+10
```

不复制当前帧。

### 6.2 每帧 55D 数据

```text
offset  0..2    root_pos_w             3
offset  3..6    root_quat_wxyz         4
offset  7..9    root_lin_vel_b         3
offset 10..12   root_ang_vel_b         3
offset 13..33   joint_pos GMT order   21
offset 34..54   joint_vel GMT order   21
```

速度由完整 50 Hz 轨迹计算，不根据 GMT Redis GET 次数计算。

### 6.3 packet header 和带宽

header 记录：

- magic `OMGBT001` 和 version 1。
- stream ID 和递增 sequence。
- publish timestamp、FPS 和 current index。
- command revision 和 plan ID。
- idle/text/audio/transition/error flags。
- GMT joint-order SHA256。
- payload CRC32。

packet 大小为：

```text
104-byte header + 110 * 55 * 4 = 24,304 bytes
```

50 Hz 发布时约为 `1.22 MB/s`，对本机 Redis 开销很小。Bridge 使用
latest-only 后台 publisher，Redis 短暂阻塞不会阻塞 50 Hz 动作主循环。

### 6.4 GMT ACK 与音乐启动

运动 key 默认为 `gmt_online_frame_bumi`，独立 ACK key 默认为
`gmt_online_frame_bumi_ack`。第一帧音乐动作 packet 先进入 Redis；GMT 完成
magic、CRC、四元数和关节顺序校验并接受新sequence后，写入52字节
`trajectory_ack_v1`：

```text
magic/version/size
stream_id
sequence
command_revision
plan_id
gmt_received_unix_ns
```

OMG后台读取ACK，只有 stream ID、command revision 一致且ACK sequence不早于
第一条音乐packet时，才启动ffplay并建立音乐执行时钟。默认每5 ms轮询ACK，
等待上限2秒；超时后fail-closed切换固定站立，不会在GMT尚未接收动作时播放。

### 6.5 关节顺序

- OMG 顺序来自 `assets/robots/bumi/bumi_kinematics.json`。
- GMT 顺序来自 policy ONNX `joint_names`。
- Bridge 根据名称计算排列。
- 名称集合不完全一致时拒绝启动。
- GMT 再用 SHA256 校验发布者顺序。

这里不使用旧 GMR 的手写重排数组。

## 7. GMT policy 输入和输出

当前 `0724_lab_148500.onnx` 实际 contract：

```text
policy          [1, 69]
history_obs     [1, 690]
command_window  [1, 1092] = 21 * 52
actions         [1, 21]
```

52D command feature 每帧包含：

```text
root z                  1
projected gravity       3
root linear velocity    3
root angular velocity   3
joint position         21
joint velocity         21
--------------------------
total                  52
```

projected gravity 使用每一帧自己的 quaternion 计算。

Gazebo 配置中 control frequency 默认 2000 Hz，decimation 为 10，因此 GMT
policy 约为 200 Hz。OMG 参考为 50 Hz，GMT 在同一 packet 的 20 ms 周期内可
执行多次 policy，但不会把同一 packet 的速度重算为零。

## 8. 耗时和实时性

### 8.1 已实测部分

在当前本机 GPU 环境上，BUMI 100M ONNX + CUDA provider 实测：

- 首个文本 plan 约 `0.35 s`。
- 后续滚动 plan 约 `0.20–0.23 s`。
- 每次 plan 覆盖 2 秒动作。
- 默认 buffer 给 Planner 约 1.2 秒完成下一次规划。
- 因此 100M 在当前机器上已达到滚动实时条件。

其他微基准：

- 110 帧 packet 构造约 `3.6 ms/tick`。
- BUMI MuJoCo 1280×720 EGL 渲染约 `4.2 ms/frame`，网页默认只渲染
  20 Hz。
- Bridge 的50 Hz 主循环、Redis 二进制发布和 Planner 是分离的，
  Redis 写入在后台线程中执行。

上述数字是本机实测，会随 GPU、driver、provider、散热和其他 GPU 负载改变。

### 8.2 100M 和 300M 是否实时

| 模型 | 结构 | 本地文件 | 实时结论 |
|---|---|---|---|
| 100M | hidden 768, 10 layers, 12 heads | CKPT + BUMI ONNX | 已实测实时 |
| 300M | hidden 1152, 14 layers, 18 heads | 目前只有 CKPT | 未实测，不能直接宣称实时 |

当前本地文件：

```text
100M CKPT    /home/weili/OMG_models/bumi_100m/sstep=200000.ckpt       ~1.93 GB
100M ONNX    denoiser_step.onnx + .onnx.data                          ~508 MB
300M CKPT    /home/weili/OMG_models/bumi_300m/sstep=200000.ckpt       ~5.06 GB
300M ONNX    尚未导出
```

300M 能否实时必须先导出 ONNX，再在目标 GPU 上连续测量。验收建议：

```text
rolling latency P95 < 0.8 s       recommended
rolling latency max < 1.2 s       continuity boundary for current buffer
no buffer underrun
50 Hz Bridge loop does not miss deadlines continuously
```

300M 参数量和每层宽度都大于 100M，不能只按 checkpoint 文件大小线性推算
延迟。CUDA provider 不足时可再评估 TensorRT FP16，但必须重做 ONNX/TensorRT
数值和动作质量验证。

## 9. ONNX 与 CKPT

### 9.1 CKPT

CKPT 是 PyTorch Lightning 训练检查点，适合：

- 继续训练和保留 optimizer/scheduler 状态。
- Python/PyTorch 研究、损失计算和训练结构修改。
- 离线生成和准确重现训练模型。

缺点：

- 文件包含的状态多，更大。
- 依赖 PyTorch/Hydra/Lightning 完整模型构造。
- 对固定 shape 的重复 50-step denoiser 不如 ONNX Runtime/TensorRT 便于优化。
- **当前 `bumi_gmt_runtime` 不直接接受 CKPT**。

### 9.2 ONNX

实时 Planner 使用 ONNX denoiser step，适合：

- 固定 batch 2、sequence 60、feature 93 的部署。
- ONNX Runtime CUDA provider。
- 可选 TensorRT provider/FP16/engine cache。
- 启动后 ONNX、T5、stats 和 kinematics 常驻，不每条命令重新加载。

需要注意：

- ONNX 只是 diffusion denoiser；T5、BUMI representation/FK、历史编码和动作解码
  仍由 OMG Python/PyTorch 代码处理。
- `.onnx`、`.onnx.data`、`.onnx.meta.json`、匹配的 stats 和 kinematics
  必须一起保留。
- ONNX 和 PyTorch 不要求 bit-exact，但导出器会做 parity 校验。当前
  100M metadata 记录的 ONNX parity 约为 max abs `0.00144`、RMSE
  `0.000208`，在导出阈值内。

结论：**训练和研究保留 CKPT，实时 GMT 部署使用 ONNX。**

### 9.3 CUDA provider 与 TensorRT provider

CUDA provider：

- 部署简单，不需要 `tensorrt-cu12` Python package。
- 当前 100M 已实测达到实时。
- 推荐先用它做稳定性和 Gazebo 联调。

TensorRT provider：

- FP16 和 engine optimization 可能更快，对 300M 更有价值。
- 需要匹配 CUDA/TensorRT/ONNX Runtime 版本。
- 首次 engine build 可能很慢，之后从 engine cache 启动。
- engine 通常和 GPU 型号、TensorRT 版本相关。
- 启用 FP16 后需要重新比较动作质量。

当前推荐命令显式使用：

```text
--providers CUDAExecutionProvider,CPUExecutionProvider
--no-tensorrt-fp16
```

## 10. 导出 300M ONNX

导出时必须使用 full-data BUMI stats：

```bash
cd /home/weili/OMG
source .venv/bin/activate

export PYTHONPATH="$PWD/src:$PWD"
export CKPT=/home/weili/OMG_models/bumi_300m/sstep=200000.ckpt
export STATS=/home/weili/OMG_models/bumi_93d_stats.json
export T5=/home/weili/OMG_models/t5-base-local
export ONNX=/home/weili/OMG_models/bumi_300m/onnx/denoiser_step.onnx

CUDA_VISIBLE_DEVICES=0 \
python -m omg.cli.generation.export_onnx \
  --ckpt_path "$CKPT" \
  --exp 300m_bumi \
  --output "$ONNX" \
  --device cuda \
  --opset 18 \
  --batch_size 2 \
  representation.stats_path="$STATS" \
  model.text_encoder.model_name="$T5"
```

导出器会严格加载 checkpoint，并执行 PyTorch wrapper 与 ONNX parity
检查。不应使用 `--allow-partial-ckpt` 来隐藏结构不匹配。

## 11. 运行前准备

### 11.1 Redis

```bash
redis-cli -p 6379 ping
```

期望输出 `PONG`。如果 Redis 尚未启动，根据本机的 Redis 管理方式启动它。

### 11.2 一次性编译 GMT

```bash
cd /home/weili/docker_projects/bumi_GMT_deployment_listao/bumi_GMT_deployment_listao
catkin build rl_controllers
source devel/setup.bash
```

构建日志必须确认找到 hiredis。如果出现
`hiredis not found, GMT online mode disabled`，GMT 不会读取 Redis。

## 12. 运行命令

### 12.1 终端 1：推荐完整命令

```bash
cd /home/weili/OMG
source .venv/bin/activate

export PYTHONPATH="$PWD/src:$PWD"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl

BUMI_ONNX=/home/weili/OMG_models/bumi_100m/onnx/denoiser_step.onnx
BUMI_STATS=/home/weili/OMG_models/bumi_93d_stats.json
T5=/home/weili/OMG_models/t5-base-local
GMT_POLICY=/home/weili/docker_projects/bumi_GMT_deployment_listao/bumi_GMT_deployment_listao/src/legged_rl/rl_controller/rl_controllers/policy/bumi/0724_lab_148500.onnx

CUDA_VISIBLE_DEVICES=0 \
python -m omg.cli.realtime.bumi_gmt_runtime \
  --diffusion-onnx "$BUMI_ONNX" \
  --representation-stats-path "$BUMI_STATS" \
  --kinematics-path assets/robots/bumi/bumi_kinematics.json \
  --text-encoder-model "$T5" \
  --gmt-policy-onnx "$GMT_POLICY" \
  --providers CUDAExecutionProvider,CPUExecutionProvider \
  --no-tensorrt-fp16 \
  --planner-bind tcp://127.0.0.1:5571 \
  --command-bind tcp://127.0.0.1:5581 \
  --redis-host 127.0.0.1 \
  --redis-port 6379 \
  --redis-key gmt_online_frame_bumi \
  --tracker-fps 50 \
  --history-fps 30 \
  --history-frames 10 \
  --planner-frames 60 \
  --replan-remaining-frames 60 \
  --play-audio \
  --sim-stream-bind 127.0.0.1:7870 \
  --sim-stream-fps 20 \
  --sim-stream-width 1280 \
  --sim-stream-height 720 \
  --sim-camera-view iso \
  --sim-follow-mode xy \
  --continuous \
  --planner-log-jsonl outputs_realtime/bumi_gmt/planner.jsonl \
  --status-jsonl outputs_realtime/bumi_gmt/status.jsonl
```

参考动作网页：

```text
http://127.0.0.1:7870/
```

### 12.2 终端 1：最简实时命令

下面只保留四个必需路径和无限运行开关：

```bash
cd /home/weili/OMG
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD"

CUDA_VISIBLE_DEVICES=0 \
python -m omg.cli.realtime.bumi_gmt_runtime \
  --diffusion-onnx /home/weili/OMG_models/bumi_100m/onnx/denoiser_step.onnx \
  --representation-stats-path /home/weili/OMG_models/bumi_93d_stats.json \
  --text-encoder-model /home/weili/OMG_models/t5-base-local \
  --gmt-policy-onnx /home/weili/docker_projects/bumi_GMT_deployment_listao/bumi_GMT_deployment_listao/src/legged_rl/rl_controller/rl_controllers/policy/bumi/0724_lab_148500.onnx \
  --continuous
```

这条命令会使用 CUDA/CPU provider、默认 Redis 和默认端口，但不启动网页，
也不从电脑播放音乐。

### 12.3 终端 2：GMT Gazebo

```bash
cd /home/weili/docker_projects/bumi_GMT_deployment_listao/bumi_GMT_deployment_listao
./simulation.sh
```

`simulation.sh` 不需要附加 OMG 参数。进入 Gazebo 后继续使用 GMT 原有安全激活流程。

### 12.4 终端 3：交互命令

`command_client` 默认已连接 `tcp://127.0.0.1:5581`，因此可以省略
`--connect`：

```bash
cd /home/weili/OMG
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD"

python -m omg.cli.realtime.command_client --interactive
```

交互示例：

```text
text walk forward slowly
text turn left
audio /home/weili/datasets/AISTPP_official/music/wav/mHO1.wav
status
stand
quit
```

单次命令：

```bash
python -m omg.cli.realtime.command_client text "walk forward slowly"
python -m omg.cli.realtime.command_client audio /home/weili/datasets/AISTPP_official/music/wav/mHO1.wav
python -m omg.cli.realtime.command_client stand
python -m omg.cli.realtime.command_client status
```

`quit` 只退出命令客户端，不停止 Planner、Bridge 或 GMT。

## 13. `bumi_gmt_runtime` 参数和默认值

下表中除“必需”外都可以不写，使用当前代码默认值。

| 参数 | 默认值 | 是否可省略 | 作用 |
|---|---:|---|---|
| `--diffusion-onnx` | 无 | 否 | BUMI denoiser ONNX |
| `--representation-stats-path` | 无 | 否 | 匹配训练的 93D stats |
| `--text-encoder-model` | 无 | 否 | 本地 T5 |
| `--gmt-policy-onnx` | 无 | 否 | GMT policy，也提供 joint names/default pose |
| `--kinematics-path` | `assets/robots/bumi/bumi_kinematics.json` | 是 | BUMI FK/grounding |
| `--providers` | `CUDAExecutionProvider,CPUExecutionProvider` | 是 | ONNX Runtime provider |
| `--planner-bind` | `tcp://127.0.0.1:5571` | 是 | Planner ZMQ bind |
| `--planner-connect` | 由 bind 推导 | 是 | Bridge ZMQ connect |
| `--command-bind` | `tcp://127.0.0.1:5581` | 是 | 动态命令服务 |
| `--cfg-scale` | `None` | 是 | 统一 CFG，默认用分模态 scale |
| `--cfg-text-scale` | `2.5` | 是 | 文本 CFG |
| `--cfg-audio-scale` | `2.5` | 是 | 音频 CFG |
| `--torch-device` | `cuda` | 是 | T5/representation/FK 设备 |
| `--seed` | `0` | 是 | diffusion 随机种子 |
| `--tensorrt-fp16` | `true` | 是 | 只在 provider 包含 TensorRT 时生效 |
| `--no-tensorrt-fp16` | 关闭开关 | 是 | CUDA-only 运行时建议显式写 |
| `--dit-cache` / `--no-dit-cache` | 启用 | 是 | 扩散层 cache |
| `--redis-host` | `127.0.0.1` | 是 | Redis host |
| `--redis-port` | `6379` | 是 | Redis port |
| `--redis-db` | `0` | 是 | Redis DB |
| `--redis-key` | `gmt_online_frame_bumi` | 是 | GMT motion key |
| `--redis-ack-key` | `<redis-key>_ack` | 是 | GMT接收确认key |
| `--redis-ack-poll-ms` | `5` | 是 | OMG轮询ACK间隔 |
| `--audio-ack-timeout-ms` | `2000` | 是 | 音乐首帧等待GMT确认上限；超时回站 |
| `--redis-ttl-ms` | `500` | 是 | packet TTL |
| `--tracker-fps` | `50` | 是 | Bridge/Redis 参考帧率 |
| `--history-fps` | `30` | 是 | Planner history 帧率 |
| `--history-frames` | `10` | 是 | Planner 历史帧数 |
| `--planner-frames` | `60` | 是 | Planner 每次输出帧数 |
| `--replan-remaining-frames` | `60` | 是 | 还剩 60×50Hz 帧时规划 |
| `--condition-audio-step-frames` | 自动，当前为 `24` | 是 | 每次 replan 的音频时间轴步长 |
| `--play-audio` | 关闭 | 是 | 用 ffplay 同步播放音频 |
| `--ffplay` | `/usr/bin/ffplay` | 是 | ffplay 路径 |
| `--sim-stream-bind` | `None` | 是 | 不写则不启动 MuJoCo 网页 |
| `--sim-stream-fps` | `20` | 是 | 网页渲染帧率 |
| `--sim-stream-width` | `1280` | 是 | 网页宽度 |
| `--sim-stream-height` | `720` | 是 | 网页高度 |
| `--sim-camera-view` | `iso` | 是 | 相机视角 |
| `--sim-follow-mode` | `xy` | 是 | 相机跟随 |
| `--status-jsonl` | `None` | 是 | Bridge 状态日志 |
| `--planner-log-jsonl` | `None` | 是 | Planner 请求/耗时日志 |
| `--output` | `None` | 是 | 可选保存实际发送 qpos NPZ |
| `--continuous` | 关闭 | 无限运行必须写 | 持续运行 |
| `--num-frames` | `0` | 是 | 不用 continuous 时指定有限帧数 |
| `--startup-grace-seconds` | `1.0` | 是 | Planner 子进程启动检查时间 |

音频步长的默认计算：

```text
plan_tracker_frames = round(60 * 50 / 30) = 100
replan_interval_tracker_frames = 100 - 60 = 40
condition_audio_step_frames = round(40 * 30 / 50) = 24
```

Bridge 内部还有以下默认值：

| Bridge 参数 | 默认值 | 说明 |
|---|---:|---|
| `--blend-seconds` | `0.2` | 固定站立/旧 plan 进入新 plan 的混合时间 |
| `--return-seconds` | `1.0` | 回固定站立的最短时间 |
| `--return-max-joint-speed` | `1.5 rad/s` | 回站关节速度上限 |
| `--audio-fps` | `30` | current35 音频特征时间轴 |
| `--audio-type` | `audio` | 运行时读取 WAV |
| `--audio-feature-type` | `current35` | 35D 音频特征 |
| `--timeout-ms` | `120000` | 单次 Planner request 最大等待时间 |
| `--status-interval-seconds` | `1.0` | 周期状态记录间隔 |

这些高级参数当前属于 `omg_bumi_gmt_bridge` 直接 CLI，统一
`bumi_gmt_runtime` 使用它们的默认值。如需调整，可分别启动 Planner Server
和 Bridge；常规联调不需要改动。

## 14. 输出、日志和可视化

OMG 侧输出：

- Redis `gmt_online_frame_bumi`：GMT 实时参考轨迹。
- `status.jsonl`：command ID/revision、pending、buffer、plan、audio、Redis
  和 error 状态。
- `planner.jsonl`：每次 Planner 请求、条件、计划耗时和各阶段 timing。
- `--output xxx.npz`：保存 Bridge 真正发送的 50 Hz BUMI qpos28。
- `http://127.0.0.1:7870/`：MuJoCo 参考动作。
- 默认声卡：可选 ffplay 音乐。

GMT 侧输出：

- Gazebo 显示 GMT policy 真正执行的 BUMI。
- policy 输出 21D action。
- action 经 action scale、default pose 和原 PD 控制器变成关节命令。

网页是参考，Gazebo 是执行结果。当前两者通过并排人工观察比较，尚未
实现 Gazebo 关节状态回传和数值 RMSE。

## 15. 安全和失败行为

- OMG 停止后 Redis key 最多 500 ms 过期。
- GMT 对 `trajectory_v1` 要求 sequence 在 0.2 秒内推进，否则回 `DEFAULT`。
- CRC、尺寸、NaN/Inf、非法四元数或 joint hash 不匹配的 packet 会被拒绝。
- Planner 失败时 Bridge 平滑回固定站立，并对当前 revision 锁存错误，
  不产生失败重试风暴。新命令到达后才重试。
- `stand` 不发 Planner 请求。
- GMT remote-B、emergency stop、fall protection 和激活状态机保持原样。
- 先完成 MuJoCo 和 Gazebo 验证，再进入实物。

## 16. 已完成验证和尚未完成验证

已完成：

- OMG 全量测试：264 passed。
- OMG realtime 测试：84 passed。
- 启动无命令和 `stand` 的 Planner 请求数为 0。
- 60@30 Hz 到 100@50 Hz 重采样。
- 110 帧 packet 和 21 个不同时间帧。
- Python publisher → Redis → C++ GMT `trajectory_v1` 交叉语言验证。
- legacy 35-float Redis 兼容。
- GMT C++/catkin 编译和测试。
- 100M BUMI ONNX 文本滚动规划。
- BUMI MuJoCo EGL 1280×720 渲染。

尚需互动桌面环境完成：

- 实际启动 Gazebo 并按安全流程激活 GMT。
- 用主机真实声卡验证 ffplay 听感同步。
- 实物 BUMI 验证。
- 300M ONNX 导出、延迟 benchmark 和长时间稳定性测试。

## 17. 常见问题

### 固定站立时 Planner 没有日志

这是正常行为。无命令不调用 Planner。

### GMT 一直处于 DEFAULT

检查：

1. `redis-cli STRLEN gmt_online_frame_bumi` 是否返回正整数；
   `trajectory_v1` 当前应为 24304 bytes。
2. GMT 构建是否找到 hiredis。
3. OMG 和 GMT 是否使用同一 host/port/db/key。
4. 是否有旧 GMR 同时覆盖同一 key。
5. 是否报 joint-order SHA256 mismatch。

### 音乐动作有但没有声音

检查 `--play-audio`、`/usr/bin/ffplay`和终端 1 主机的默认音频输出设备。

### 提示缺少 TensorRT package

先使用：

```text
--providers CUDAExecutionProvider,CPUExecutionProvider
--no-tensorrt-fp16
```

只有在确实要使用 `TensorrtExecutionProvider` 时才安装和配置 TensorRT。

### 300M 动作质量更好就一定更适合实时吗

不一定。动作质量、条件理解和推理延迟是不同的指标。应使用同一组文本/
音乐、同一 GPU 和同一 provider，对比 P50/P95/max latency、buffer underrun 和 Gazebo
执行质量后再决定。
