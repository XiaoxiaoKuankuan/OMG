# 实时动态文本与音乐命令

动态命令模式让规划器服务器、ONNX、T5 和桥接器保持常驻，同时从另一个终端切换文本、音乐或站立条件。规划器服务器不保存当前提示词；桥接器在每个规划请求的 metadata 中发送当前条件。

## 行为语义

- 文本命令会持续生效，桥接器不断滚动生成对应动作，直到收到下一条命令。
- 音乐从第一次使用该命令规划时的机器人 tracker frame 开始，按机器人实际执行帧推进，而不是按扩散推理耗时或规划次数推进。
- 音乐只播放自身 WAV 时长，不循环；结束后自动切换到 `text: stand still` 并请求站立计划。
- `stand` 立即改变有效条件，但不会清空当前动作缓冲区。
- 所有切换采用 `next_replan`：没有 pending 请求时立即重规划；已有请求时先追加返回的旧计划保证连续性，随后立即用最新命令重规划。新计划到达时只裁剪尚未执行的未来动作。
- `quit` 只关闭命令客户端，不关闭 bridge 或 planner server。

模型每次仍输出原生规划长度。例如 `--planner-frames 60 --history-fps 30` 表示每次生成约 2 秒的模型动作；`--continuous` 使桥接器持续运行。

## 三进程运行：本机动态 Dry Run

### 终端 1：Planner Server

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
  --log-jsonl outputs_realtime/planner_100m.jsonl
```

Planner Server 只启动一次。后续文本和音乐切换不会重新加载 ONNX 或 T5。

### 终端 2：动态 Dry Run Bridge

```bash
cd /home/weili/OMG
source .venv/bin/activate

export PYTHONPATH=/home/weili/OMG/src
export MUJOCO_GL=egl

python -m omg.cli.realtime.holomotion_dry_run \
  --connect tcp://127.0.0.1:5571 \
  --seed-motion /home/weili/OMG/inputs/seed_motion.npz \
  --history-frames 10 \
  --history-fps 30 \
  --tracker-fps 50 \
  --planner-frames 60 \
  --continuous \
  --replan-remaining-frames 40 \
  --command-bind tcp://127.0.0.1:5581 \
  --initial-condition-sequence "text: stand still" \
  --audio-type audio \
  --audio-feature-type current35 \
  --audio-fps 30 \
  --sim-stream-bind 127.0.0.1:7870 \
  --sim-stream-fps 20 \
  --sim-camera-view iso \
  --sim-follow-mode xy \
  --sleep \
  --status-jsonl outputs_realtime/dynamic/status.jsonl \
  --output outputs_realtime/dynamic/bridge.npz
```

浏览器打开 `http://127.0.0.1:7870/video.mjpg` 查看视频流。

未显式设置 `--condition-audio-step-frames` 时，bridge 按下面的执行间隔自动计算：

```text
plan_tracker_frames = round(planner_frames * tracker_fps / history_fps)
replan_interval_tracker_frames = max(1, plan_tracker_frames - replan_remaining_frames)
condition_audio_step_frames = max(1, round(replan_interval_tracker_frames * audio_fps / tracker_fps))
```

上面的参数得到 `36`，启动时输出：

```text
[dynamic-condition] computed audio step frames=36
```

显式传入 `--condition-audio-step-frames N` 会覆盖自动计算值。

### 终端 3：交互命令客户端

```bash
cd /home/weili/OMG
source .venv/bin/activate

export PYTHONPATH=/home/weili/OMG/src

python -m omg.cli.realtime.command_client \
  --connect tcp://127.0.0.1:5581 \
  --interactive
```

交互输入示例：

```text
text walk forward slowly
text turn left slowly
text wave both arms
audio /home/weili/datasets/AISTPP_official/music/wav/mHO1.wav
status
stand
quit
```

## 单次命令

文本：

```bash
PYTHONPATH=src python -m omg.cli.realtime.command_client \
  --connect tcp://127.0.0.1:5581 \
  text "walk forward slowly"
```

音乐：

```bash
PYTHONPATH=src python -m omg.cli.realtime.command_client \
  --connect tcp://127.0.0.1:5581 \
  audio /home/weili/datasets/AISTPP_official/music/wav/mHO1.wav
```

站立：

```bash
PYTHONPATH=src python -m omg.cli.realtime.command_client \
  --connect tcp://127.0.0.1:5581 \
  stand
```

状态：

```bash
PYTHONPATH=src python -m omg.cli.realtime.command_client \
  --connect tcp://127.0.0.1:5581 \
  status
```

## 实机 Bridge

Planner Server 的 `--bind` 应改为机器人可访问的工作站地址，例如 `tcp://0.0.0.0:5571`。在 G1 的 HoloMotion/ROS 环境中运行：

```bash
cd /home/unitree/OMG

export PYTHONPATH=/home/unitree/OMG/src:$PYTHONPATH

/root/miniconda3/envs/holomotion_deploy/bin/python \
  -m omg.cli.realtime.holomotion_real_bridge \
  --connect tcp://10.0.20.14:5571 \
  --history-frames 10 \
  --history-fps 30 \
  --tracker-fps 50 \
  --planner-frames 60 \
  --continuous \
  --replan-remaining-frames 40 \
  --command-bind tcp://127.0.0.1:5581 \
  --initial-condition-sequence "text: stand still" \
  --audio-type audio \
  --audio-feature-type current35 \
  --audio-fps 30 \
  --holomotion-config /home/unitree/holomotion/deployment/unitree_g1_ros2_29dof/src/config/g1_29dof_holomotion.yaml \
  --publish-bind tcp://*:6001 \
  --activation-mode remote-b \
  --sleep \
  --status-jsonl /home/unitree/OMG/outputs_realtime/dynamic/status.jsonl \
  --output /home/unitree/OMG/outputs_realtime/dynamic/bridge.npz
```

命令客户端应连接实机 bridge 的 `--command-bind` 地址。若客户端不在 G1 本机，把 bridge 的 bind 改为可访问的接口，并使用 G1 IP 连接。动态命令不会绕过 `remote-b` 激活、lowstate 检查或急停流程。

## JSON 协议

命令服务使用 ZMQ REP 和 JSON。支持的请求为：

```json
{"type":"text","text":"turn left slowly","command_id":"optional-id"}
{"type":"audio","audio_path":"/absolute/path/music.wav","audio_type":"audio","on_end":"stand","command_id":"optional-id"}
{"type":"stand","command_id":"optional-id"}
{"type":"status"}
```

成功响应：

```json
{
  "ok": true,
  "accepted": true,
  "switch_policy": "next_replan",
  "active": {
    "command_id": "optional-id",
    "type": "text",
    "text": "turn left slowly",
    "audio_path": null,
    "revision": 3,
    "condition_session_id": "...",
    "condition_index": 0,
    "audio_duration_seconds": null,
    "audio_start_tracker_frame": null,
    "audio_end_tracker_frame": null
  }
}
```

错误响应不会终止 bridge：

```json
{"ok":false,"error":{"type":"ValueError","message":"..."}}
```

## 日志与切换延迟

命令接受、音乐开始和结束日志示例：

```text
[dynamic-condition] accepted command_id=abc type=audio revision=4
[dynamic-condition] audio start command_id=abc path=/absolute/path/music.wav duration=10.000000s start_frame=1250
[dynamic-condition] audio ended command_id=abc frame=1750; switching to stand
```

若新命令在一次推理期间到达，旧响应会记录 `stale_command=true`，然后输出：

```text
[dynamic-condition] stale plan completed; scheduling current command immediately
```

实际切换延迟是当前 pending 请求完成时间加最新请求返回时间；bridge 在此期间继续执行已有缓冲区，不会直接清空 buffer。`status.jsonl` 记录 command/session、音乐起止帧及 stale 状态。

## 固定条件兼容模式

不传 `--command-bind` 时，原有方式保持不变，并继续要求 `--condition-sequence`：

```bash
PYTHONPATH=src python -m omg.cli.realtime.holomotion_dry_run \
  --connect tcp://127.0.0.1:5571 \
  --seed-motion inputs/seed_motion.npz \
  --condition-sequence "text: walk forward" \
  --continuous \
  --sleep \
  --output outputs_realtime/fixed/bridge.npz
```

动态模式下若同时传入 `--condition-sequence`，它会覆盖 `--initial-condition-sequence`，仅作为启动时的初始条件；后续仍可动态切换。

## 常见错误和限制

- 文本不能为空，且第一版拒绝换行符和 `|`，以免破坏 condition-sequence 语法。
- 音频必须是 bridge 所在机器可读取的绝对 `.wav` 路径；不要求放在 `/tmp`。
- WAV 必须非空且采样率大于 0。第一版仅支持 `audio_type=audio`、`on_end=stand`，不支持音乐循环。
- `Address already in use` 表示 `--command-bind` 端口已占用，请关闭旧 bridge 或换端口。
- 客户端超时通常表示 connect 地址错误、bridge 未启动或防火墙未开放。
- 音乐文件必须存在于 bridge 运行的主机；Planner Server 也需要能够按请求中的同一路径读取该文件。跨机器实机部署时，应把 WAV 放在两台机器的相同绝对路径，或使用共享挂载。
