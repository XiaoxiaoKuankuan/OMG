# OMG 实时生成 G1 并通过 GMR 重定向到 BUMI3

本文说明新增的独立 OMG→GMR 实时链路。它不会修改或替代现有
`holomotion_dry_run`、`holomotion_real_bridge`、固定
`--condition-sequence` 或 GEM+GMR 运行方式。

## 1. 数据链路

```text
command_client
      ↓ text / audio / stand
omg_gmr_bridge
      ├── 无命令或 stand：固定 G1 qpos36，不调用 Planner
      └── text / audio：常驻 Planner Server 滚动生成 G1 qpos36
      ↓ 固定 50 Hz，Redis JSON: omg_online_frame_g1
GMR G1MotionReader + G1MotionAdapter
      ↓ G1 MuJoCo FK，12 个 BodyMap 目标
现有 GMR IK
      ↓
BUMI3 MuJoCo Viewer
      ↓ 可选 Redis binary: gmt_online_frame_bumi
现有 GMT / BUMI3 控制链路
```

Planner Server、ONNX 和 T5 全程常驻。bridge 也是常驻进程；运行中可以
输入任意多条文本、音乐和站立命令。

## 2. 固定站姿语义

默认初始条件为 `text: stand still`，但在本 bridge 中它只是兼容的状态
名称，不会发送给扩散模型。

- 启动后没有命令：逐帧发布固定 G1 姿态。
- `stand`：停止申请生成计划并平滑返回固定姿态。
- 音乐结束：自动执行相同的固定站姿流程。
- `--idle-motion` 存在时，固定姿态取该文件最后一帧。
- 未设置 `--idle-motion` 时，默认 `--idle-preset neutral`：腿和腰关节
  全零、左右肩 roll 为 `+0.12/-0.12 rad`、左右 G1 肘为 `pi/2`，根为
  `[0, 0, 0.793]`；bridge 默认通过 `--initial-yaw-degrees 90` 将根绕世界
  Z 轴逆时针旋转 90 度。
- 兼容旧结果可显式使用 `--idle-preset seed-last`，此时才取
  `--seed-motion` 最后一帧。
- 默认保留动作结束时的根节点 X/Y 和当前 yaw；只将根高度、根 roll/pitch
  和 29 个关节平滑回到固定姿态，因此 `stand` 不会使机器人转回初始朝向。
- 返回至少使用 `--return-seconds`，大关节角差还会受
  `--return-max-joint-speed` 限制并自动延长，避免猛拉。
- 返回完成后每一帧完全相同，不会反复生成“stand still”。

### `seed_motion.npz` 不是安全站姿

`inputs/seed_motion.npz` 是把 `assets/stats/g1_125d_stats.json` 中的
`default_root_pos + default_root_quat + default_joint_dof` 拼成 qpos36，再重复
10 帧得到的。统计文件中的 default qpos 是 `compute_stats.py` 对训练集全部有效
qpos 求均值的结果，不是 G1 MuJoCo 零位，也不是人工验证的 stand keyframe。
因此它自然包含约 32～35 度屈膝、约 26～31 度肘角和倾斜根姿态。

这个文件仍可作为与训练分布相近的扩散历史样例，但不能同时承担固定 idle 的语义。
OMG→GMR bridge 已将两者分开：`--seed-motion` 仍保留兼容输入，固定站立默认使用
`neutral` preset。Planner 的第一次历史由实际发布的 neutral 帧构造，不会把两种
姿态混在一个 history 中。

`neutral` 是 MuJoCo/软件参考姿态。实机使用仍应通过 `--idle-motion` 指定经过
硬件负责人、限位、低速和吊装测试确认的安全姿态。

### 为什么 G1 idle 的肘不是 0

G1 肘部 qpos 为 0 时，前臂沿 elbow body 的局部 `+X`，画面里会向前平举；
`+pi/2` 才把前臂转为向下。BUMI3 的前臂在肘部零位沿局部 `-Z`，所以两者的
同姿态关系为：

```text
q_bumi_elbow = q_g1_elbow - pi/2
```

因此 neutral 中的 G1 肘为 `pi/2`，重定向后的 BUMI 肘接近 0；这两组不同的
关节数值表达的是相同的“手臂自然下垂”物理姿态。

## 3. 文本和音乐语义

文本命令持续有效。bridge 在计划剩余帧达到阈值时滚动重规划，直到收到
下一条文本、音乐或 `stand`。

音乐只执行 WAV 自身时长：

- 必须是存在的绝对 `.wav` 路径；
- 时间轴由 bridge 实际输出的 tracker frame 推进；
- 不按扩散推理次数推进；
- 新文本或新音乐会取消旧音乐的结束状态；
- 音乐不循环；
- 音乐结束后自动平滑返回固定站姿，不向 Planner 发送 stand 文本。

推理期间固定频率输出不会停止。初次推理时继续站立；滚动推理时继续执行
已有计划。Redis 连接和重连位于独立线程，Redis 暂时断开也不会阻塞动作
时钟。

## 4. 完整运行命令

以下路径与当前机器一致。

### 终端 0：确认 Redis

```bash
redis-cli -h 127.0.0.1 -p 6379 PING
```

正常返回：

```text
PONG
```

如果没有运行 Redis，可在单独终端启动：

```bash
redis-server --bind 127.0.0.1 --port 6379
```

### 终端 1：常驻 OMG Planner Server

```bash
cd /home/weili/OMG
source .venv/bin/activate

export PYTHONPATH=/home/weili/OMG/src
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ONNX=/home/weili/OMG_models/generation/onnx/100m_updated/denoiser_step.onnx
T5=/home/weili/OMG_models/t5-base-local

CUDA_VISIBLE_DEVICES=0 \
python -m omg.cli.realtime.planner_server \
  --bind tcp://127.0.0.1:5571 \
  --diffusion-onnx "$ONNX" \
  --text-encoder-model "$T5" \
  --providers CUDAExecutionProvider,CPUExecutionProvider \
  --no-tensorrt-fp16 \
  --dit-cache \
  --cfg-scale 2.5 \
  --torch-device cuda \
  --seed 0 \
  --log-jsonl outputs_realtime/omg_gmr/planner.jsonl
```

这个进程只启动一次。后续切换命令不重启 ONNX 或 T5。

### 终端 2：OMG G1 动作仲裁与 Redis bridge

```bash
cd /home/weili/OMG
source .venv/bin/activate

export PYTHONPATH=/home/weili/OMG/src
export MUJOCO_GL=egl

python -m omg.cli.realtime.omg_gmr_bridge \
  --connect tcp://127.0.0.1:5571 \
  --command-bind tcp://127.0.0.1:5581 \
  --seed-motion /home/weili/OMG/inputs/seed_motion.npz \
  --idle-preset neutral \
  --initial-yaw-degrees 90 \
  --initial-condition-sequence "text: stand still" \
  --history-frames 10 \
  --history-fps 30 \
  --tracker-fps 50 \
  --planner-frames 60 \
  --replan-remaining-frames 40 \
  --blend-seconds 0.20 \
  --return-seconds 1.0 \
  --return-max-joint-speed 1.5 \
  --audio-type audio \
  --audio-feature-type current35 \
  --audio-fps 30 \
  --redis-host 127.0.0.1 \
  --redis-port 6379 \
  --redis-db 0 \
  --redis-key omg_online_frame_g1 \
  --redis-ttl-ms 250 \
  --sim-stream-bind 127.0.0.1:7870 \
  --sim-stream-fps 20 \
  --sim-stream-width 1280 \
  --sim-stream-height 720 \
  --sim-camera-view iso \
  --sim-follow-mode xy \
  --status-jsonl outputs_realtime/omg_gmr/bridge.jsonl \
  --output outputs_realtime/omg_gmr/g1_executed.npz \
  --continuous
```

启动后应看到：

```text
[OMG→GMR] computed audio step frames=36
[OMG→GMR] stand/no command uses a fixed pose; no stand prompt is sent to Planner
[OMG G1 Redis] connected 127.0.0.1:6379/0 key=omg_online_frame_g1
```

`--output` 在无限运行时会把执行帧保存在内存中，长时间运行可去掉该参数，
仅保留 JSONL 状态日志。

### 终端 3：现有 GMR G1→BUMI3

```bash
cd /home/weili/GMR-CPP_e1jump_lowdpi

export OMG_G1_REDIS_KEY=omg_online_frame_g1
export BUMI3_REDIS_KEY=gmt_online_frame_bumi

./run_g1_bumi3.sh \
  --redis-host 127.0.0.1 \
  --redis-port 6379 \
  --redis-db 0 \
  --poll-hz 60 \
  --hz 50 \
  --ttl-ms 200 \
  --stale-ms 250 \
  --viewer-ground-penetration 0.005 \
  --viewer-width 1280 \
  --viewer-height 720 \
  --vis \
  --vis-targets
```

这个命令不改动原 GMR IK：

- BUMI3 完整模型显示在 GMR MuJoCo 窗口；
- `--vis-targets` 叠加 G1 FK 原始目标、缩放目标和 BUMI3 body skeleton；
- 默认同时向 `gmt_online_frame_bumi` 发布 BUMI3 GMT 格式。
- GMR 输出仍按每帧最低脚部几何严格贴地；
  `--viewer-ground-penetration 0.005` 只把 MuJoCo viewer 使用的 qpos 副本下移
  5 mm，减轻网格看起来悬空的缝隙，不改变发给 GMT 的根位姿和关节。设为 `0`
  可关闭；若仍有脚跟翘起，原因是脚掌姿态而不是根高度，需要另行增加足底姿态/
  接触约束。

如果只看重定向、不发布给 GMT，增加：

```bash
--no-output-redis
```

### 终端 4：交互命令客户端

```bash
cd /home/weili/OMG
source .venv/bin/activate
export PYTHONPATH=/home/weili/OMG/src

python -m omg.cli.realtime.command_client \
  --connect tcp://127.0.0.1:5581 \
  --interactive
```

交互示例：

```text
text walk forward slowly
text turn left slowly
text wave both arms
audio /home/weili/datasets/AISTPP_official/music/wav/mHO1.wav
status
stand
quit
```

`quit` 只退出客户端，不关闭 bridge、GMR 或 Planner Server。

## 5. 可视化

完整 G1 网格：

```text
http://127.0.0.1:7870/video.mjpg
```

bridge 命令已经将图像设为 `1280×720`。浏览器仍按页面布局缩放时，可以
直接打开该 URL 后使用浏览器全屏模式。摄像机默认 `xy` 跟随，G1 走远后
不会离开画面。

完整 BUMI3 网格在终端 3 启动的 GMR 原生窗口中。当前版本是两个同步视图，
并非把完整 G1 和完整 BUMI3 网格放入同一个窗口。

## 6. 单次命令

文本：

```bash
PYTHONPATH=/home/weili/OMG/src /home/weili/OMG/.venv/bin/python \
  -m omg.cli.realtime.command_client \
  --connect tcp://127.0.0.1:5581 \
  text "walk forward slowly"
```

音乐：

```bash
PYTHONPATH=/home/weili/OMG/src /home/weili/OMG/.venv/bin/python \
  -m omg.cli.realtime.command_client \
  --connect tcp://127.0.0.1:5581 \
  audio /home/weili/datasets/AISTPP_official/music/wav/mHO1.wav
```

固定站姿：

```bash
PYTHONPATH=/home/weili/OMG/src /home/weili/OMG/.venv/bin/python \
  -m omg.cli.realtime.command_client \
  --connect tcp://127.0.0.1:5581 \
  stand
```

状态：

```bash
PYTHONPATH=/home/weili/OMG/src /home/weili/OMG/.venv/bin/python \
  -m omg.cli.realtime.command_client \
  --connect tcp://127.0.0.1:5581 \
  status
```

## 7. Redis 协议

OMG→GMR key 默认为 `omg_online_frame_g1`，内容是 UTF-8 JSON：

```json
{
  "timestamp": 123.456,
  "root_pos": [0.0, 0.0, 0.786],
  "root_quat": [1.0, 0.0, 0.0, 0.0],
  "joints": ["29 个有限浮点数"]
}
```

四元数顺序是 `wxyz`。发布前会验证 36 维、NaN/Inf 和四元数范数，并将
四元数归一化。每次 `SET` 都带 `PX` TTL，bridge 停止后 key 会自动失效。

两个 Redis key 不要混用：

| Key | 方向 | 格式 |
|---|---|---|
| `omg_online_frame_g1` | OMG bridge → GMR | JSON，G1 qpos36 |
| `gmt_online_frame_bumi` | GMR → GMT | 35×float32，BUMI3 GMT 顺序 |

## 8. 关键参数

- `--idle-motion`：经过确认的固定 G1 姿态文件；使用最后一帧。
- `--initial-yaw-degrees`：内置 neutral 的初始世界 yaw，默认逆时针 90 度。
- `--blend-seconds`：新生成计划切入时间，默认 0.20 秒。
- `--return-seconds`：返回固定姿态的最短时间，默认 1.0 秒。
- `--return-max-joint-speed`：回站关节最大插值速度，默认 1.5 rad/s；必要时
  自动延长回站时间。
- `--no-preserve-idle-xy`：返回站姿时也返回 idle 文件的绝对 X/Y。
- `--reset-idle-heading`：显式要求回到初始 yaw；默认不使用，因而保持当前朝向。
- `--planner-frames`：每次扩散生成的原生帧数，默认 60。
- `--replan-remaining-frames`：未来计划达到该 tracker 帧数时重规划。
- `--condition-audio-step-frames`：未设置时自动计算；默认参数得到 36。
- `--redis-ttl-ms`：G1 输入 key 的 TTL，默认 250ms。
- `--timeout-ms`：Planner 请求超时。超时后重建 REQ transport，固定输出不中断。
- `--num-frames N`：有限帧调试；正常长期运行使用 `--continuous`。

## 9. 日志示例

文本开始：

```text
[dynamic-condition] accepted command_id=... type=text revision=1
[OMG→GMR replan request] frame=... type=text ... prompt='text: walk forward'
[OMG→GMR replan 0000] ... accepted=True stale_command=False
```

音乐结束：

```text
[dynamic-condition] audio ended command_id=... frame=...; switching to stand
[OMG→GMR fixed-idle] command_id=... revision=...; returning to fixed G1 pose
```

音乐结束后不应出现 `prompt='stand still'` 的 Planner 请求。

## 10. 常见问题

### Redis 连接失败

日志会显示 `connect failed ... will retry`。动作循环继续运行，Redis 后台线程
自动重连。检查 `redis-cli PING`、host、port 和防火墙。

### GMR 一直显示 input stale

检查：

```bash
redis-cli GET omg_online_frame_g1
```

如果为空，确认 bridge 仍在运行、两个进程使用相同 Redis DB/key，并适当增加
`--redis-ttl-ms` 和 GMR `--stale-ms`。

### 输入 stand 后 Planner 仍在计算旧请求

ZMQ REQ 已经发送后不能粗暴取消。bridge 会继续固定频率输出，正常接收并
丢弃过时响应；不会执行旧响应，也不会为 stand 创建新计划。

### 计划推理超过整个动作 horizon

该响应会标记为 expired 并丢弃，随后按当前命令重新请求。期间保持原动作或
安全返回固定姿态，不会因为 motion buffer 为空退出。

### G1 或 BUMI3 姿态不适合实机

先在两个 Viewer 中验证。实机前使用经过确认的 `--idle-motion`，并继续遵守
现有 GMT/BUMI3 控制器的激活、限位、急停和安全流程；本 bridge 不绕过这些
流程。
