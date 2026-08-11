# BUMI 原生 OMG 实时接入 GMT

更完整的历史改动、输入输出协议、参数默认值、耗时分析、
100M/300M 实时性与 ONNX/CKPT 对比见：
[OMG BUMI 原生模型接入 GMT 完整说明](omg_bumi_gmt_complete_guide.md)。

这条链路直接使用 BUMI qpos28，不经过 G1，也不经过 GMR：

```text
BUMI ONNX Planner → BUMI realtime bridge → Redis trajectory_v1 → GMT → Gazebo
                                      └→ BUMI MuJoCo 网页参考
```

Planner、ONNX 和 T5 在终端 1 中常驻。终端 2 仍只执行 GMT 原来的
`./simulation.sh`。终端 3 可以反复发送文本、WAV 或 `stand`。

Planner 历史来源可选。默认 `--history-source reference` 延续现有行为，10 帧
history 来自 Bridge 已发布的参考。增加 `--history-source lowstate` 后，10 帧
history 的根四元数和 21 个关节角来自 GMT 的真实 LowState，只有全局 root xyz
逐 tick 从当前 reference trajectory 补入。该模式使用独立 Redis key
`gmt_online_frame_bumi_lowstate`，不会改变 GMT motion 输入 key。

## 固定站立语义

启动、无命令、`stand`、音乐自然结束以及 Planner 错误都使用固定 BUMI
qpos28，不向 Planner 发送 `text: stand still`。固定关节角从 GMT policy
ONNX 的 `default_joint_pos` 元数据读取；当前 policy 的髋、膝、踝保持轻微
屈曲，左右肩 roll 为 `+0.3/-0.3 rad`。root z 由 BUMI kinematics 足底代理
自动落地。

回站时保留当前 root XY 和 yaw，只让 root roll/pitch、root z 和关节在至少
1 秒内平滑回固定姿态。默认关节恢复速度不超过 `1.5 rad/s`。Planner 失败后
当前命令会锁存为 `ERROR_IDLE`，收到下一条命令才重新尝试，避免失败重试风暴。

## 一次性编译 GMT

GMT 只增加 Redis 协议解析和 command-window 构造，不改 policy、控制输出、
remote-B、安全流程或 `simulation.sh`。修改代码后在 GMT 自己的 ROS 环境中
重新编译一次：

```bash
cd /home/weili/docker_projects/bumi_GMT_deployment_listao/bumi_GMT_deployment_listao
catkin build rl_controllers
source devel/setup.bash
```

必须确保构建日志没有 `hiredis not found, GMT online mode disabled`。若出现该
提示，请先安装 hiredis 开发包，再重新编译。

## 终端 1：OMG、Redis 发布、网页和可选音乐

先确认 Redis 已运行，并只保留一个 `gmt_online_frame_bumi` 发布者。运行本链路
时不要同时启动旧 GMR publisher。

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
  --status-jsonl outputs_realtime/bumi_gmt/status.jsonl
```

上面使用默认的 reference history。如需 LowState history，在终端 1 增加：

```bash
  --history-source lowstate \
  --redis-lowstate-key gmt_online_frame_bumi_lowstate \
  --lowstate-max-age-ms 200
```

GMT 以 50 Hz 发布实测反馈，OMG 融合后重采样为 10 帧 @ 30 Hz。首次规划前会
等待完整实测时间窗；反馈断流时暂停新规划且不自动退回 reference history，已有
计划仍连续执行，之后平滑回固定站立。默认模式不创建 LowState Redis 读取线程。

如不需要电脑音响播放音乐，删除 `--play-audio`。指定该参数时必须存在
`/usr/bin/ffplay`。声音从运行终端 1 的主机默认音频设备输出。

网页地址：

```text
http://127.0.0.1:7870/
```

网页显示 Bridge 实际发送给 GMT 的 50 Hz、重采样和混合后的 BUMI 参考，而
不是原始 30 Hz diffusion 输出。Gazebo 显示 policy 实际执行结果，可以将两个
窗口并排比较。

## 终端 2：GMT Gazebo

```bash
cd /home/weili/docker_projects/bumi_GMT_deployment_listao/bumi_GMT_deployment_listao
./simulation.sh
```

进入 Gazebo 后继续使用原有安全激活流程。先观察固定站立稳定，再发送命令。
新协议 sequence 超过 0.2 秒不更新时，GMT 自动切回 `DEFAULT`。

## 终端 3：动态命令

```bash
cd /home/weili/OMG
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD"

python -m omg.cli.realtime.command_client \
  --connect tcp://127.0.0.1:5581 \
  --interactive
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

`quit` 只退出命令客户端。文本持续滚动生成，直到下一条命令。WAV 从第一帧
音乐动作发布后等待 GMT 确认：GMT 成功读取并校验该 `sequence` 后写入独立的
`gmt_online_frame_bumi_ack` key，OMG 收到同一 stream/revision/sequence 的 ACK
才启动 ffplay 和音乐执行时钟。diffusion 推理耗时不计入音乐时间轴；自然结束
后平滑回固定站立。Bridge 默认用 20 ms RMS 窗检测低于 -50 dBFS、连续至少
0.5 秒的尾部静音；检测出的静音尾巴不计入有效音乐时长。到有效结束帧时会同时
停止 ffplay、丢弃尚未执行的音乐计划并开始平滑回站。新文本、`stand` 或新音乐
会终止旧 ffplay 进程。

## trajectory_v1

同一 Redis key 自动兼容两种格式：

- 旧 GMR 35-float 单帧继续按 legacy 方式处理。
- 新 OMG 二进制 packet 使用 magic `OMGBT001` 和版本 1。

每个新 packet 有 110 帧：过去 10 帧、当前帧、未来 99 帧。GMT policy 的
21 帧窗口严格取 packet 的第 0～20 帧，即 `t-10...t...t+10`，每帧独立计算
52D feature，不复制当前帧。每个轨迹帧包含 root position、wxyz quaternion、
body-frame root linear/angular velocity、21 个 joint position 和 21 个 joint
velocity。packet 同时校验尺寸、有限值、四元数、CRC32 和 GMT policy 关节顺序
SHA256。

GMT 每次接受新的 `trajectory_v1` sequence 后，将52字节二进制 ACK 写到独立
Redis key（默认 `<motion-key>_ack`）。ACK包含 stream ID、sequence、command
revision、plan ID和GMT接收时间。OMG拒绝旧stream、旧revision和过早sequence，
因此遗留ACK不能触发新音乐。启用 `--play-audio` 后若2秒内没有合法ACK，音乐
不会播放，当前音频命令会切换到固定站立。

OMG 读取 `bumi_kinematics.json` 的关节顺序，GMT 顺序来自 policy ONNX
`joint_names`，两边名称集合必须完全相同。不存在旧 GMR 的手写重排数组。

## 常见问题

- Planner 启动后没有 `[replan ...]`：处于固定站立时这是正确行为。
- Redis 连接失败：确认 Redis 已启动、host/port/db/key 一致。
- GMT 始终处于 DEFAULT：确认重新编译时检测到了 hiredis，并确认 OMG sequence
  正在递增。
- `joint-order SHA256 mismatch`：OMG 使用的 GMT policy 与 Gazebo 加载的 policy
  不是同一关节约定，必须停止运行并统一 policy。
- 网页端口占用：修改 `--sim-stream-bind`，命令服务端口则修改
  `--command-bind`。
- 音乐无声：确认保留 `--play-audio`、`/usr/bin/ffplay` 存在且终端 1 主机有可用
  默认音频设备；同时运行 `redis-cli STRLEN gmt_online_frame_bumi_ack`，合法 ACK
  长度应为 `52`，并确认 Gazebo 已进入 GMT 模式。GMT 未读取轨迹时不会播放音乐。
