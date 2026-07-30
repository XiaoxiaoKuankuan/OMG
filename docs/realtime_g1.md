# G1 实时部署

实时部署使用三个进程：

1. G1 Orin 上的 HoloMotion 部署进程。
2. GPU 工作站上的 OMG 实时规划器服务器。
3. G1 Orin 上的 OMG 实机桥接器。

规划器服务器负责扩散推理。实机桥接器读取机器人 lowstate、构建历史、发送条件序列请求、
接收规划的未来动作，并发布 HoloMotion `obs65` 参考数据包。

## 运行环境

实时部署使用两台计算机：

- GPU 工作站：运行 `omg.cli.realtime.planner_server`。
- G1 Orin：运行 HoloMotion 部署进程和 `omg.cli.realtime.holomotion_real_bridge`。

G1 Orin 环境必须提供 Unitree ROS 消息，包括 `unitree_hg.msg.LowState`。
请在已加载这些 ROS 包的 HoloMotion 部署环境或容器中运行实机桥接器。

## 网络

G1 Orin 必须能够访问工作站规划器的绑定地址。条件允许时请使用有线以太网。
也可以使用 Wi-Fi，但实机测试前应检查规划器延迟和抖动。

工作站规划器地址示例：

```text
tcp://10.0.20.14:5571
```

## 终端 1：G1 上的 HoloMotion

在 G1 HoloMotion 部署目录中运行：

```bash
cd /home/unitree/holomotion/deployment/unitree_g1_ros2_29dof
./launch_holomotion_29dof_docker.sh
```

活动的启动配置必须将 HoloMotion 配置为订阅 OMG latest-obs ZMQ：

```yaml
latest_obs_zmq_uri: tcp://127.0.0.1:6001
latest_obs_zmq_topic: obs65
latest_obs_zmq_mode: connect
enable_teleop_reference: true
```

请将运行时和部署字段保留在启动配置中，而不要放入机器人配置 YAML。

## 终端 2：工作站上的规划器服务器

```bash
cd /path/to/OMG
source .venv/bin/activate

PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python -m omg.cli.realtime.planner_server \
  --bind tcp://0.0.0.0:5571 \
  --diffusion-onnx models/generation/onnx/50m/last_denoiser_step.onnx \
  --providers TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider \
  --tensorrt-engine-cache-path tensorrt_engine_cache/realtime_planner \
  --dit-cache \
  --log-jsonl outputs_realtime/planner.jsonl
```

规划器服务器不负责提示词。条件来自桥接器请求的元数据，
因此运行时条件变更保留在机器人侧的桥接器命令中。

## 终端 3：G1 上的实机桥接器

请在能够使用 Unitree ROS 消息的 HoloMotion 部署环境或容器中运行：

```bash
cd /home/unitree/OMG

PYTHONPATH=src:$PYTHONPATH /root/miniconda3/envs/holomotion_deploy/bin/python \
  -m omg.cli.realtime.holomotion_real_bridge \
  --connect tcp://10.0.20.14:5571 \
  --history-frames 10 \
  --history-fps 30 \
  --tracker-fps 50 \
  --continuous \
  --replan-remaining-frames 40 \
  --condition-sequence "text: walk forward" \
  --holomotion-config /home/unitree/holomotion/deployment/unitree_g1_ros2_29dof/src/config/g1_29dof_holomotion.yaml \
  --publish-bind tcp://*:6001 \
  --activation-mode remote-b \
  --sleep \
  --status-jsonl /home/unitree/OMG/outputs_realtime/real_demo/status.jsonl \
  --output /home/unitree/OMG/outputs_realtime/real_demo/bridge.npz
```

使用 `--activation-mode remote-b` 时，桥接器会等待按下 Unitree 遥控器的 B 键，
随后才开始发送主动重新规划请求。第一次主动重新规划使用实时 lowstate 历史。

## 空运行

实机测试前，请针对规划器运行桥接器空运行：

```bash
PYTHONPATH=src python -m omg.cli.realtime.holomotion_dry_run \
  --connect tcp://10.0.20.14:5571 \
  --seed-motion /path/to/seed_motion.npz \
  --history-frames 10 \
  --history-fps 30 \
  --tracker-fps 50 \
  --continuous \
  --replan-remaining-frames 40 \
  --condition-sequence "text: walk forward" \
  --publish-bind tcp://*:6001 \
  --sleep \
  --output outputs_realtime/dry_run/bridge.npz
```

添加 `--sim-stream-bind 127.0.0.1:7870` 后，可在
`http://127.0.0.1:7870/video.mjpg` 查看本地 MuJoCo 视频流。

## 日志

规划器日志行包括：

```text
[replan 0001] request=... frame=... buffer=... prompt='walk forward' latency=...
```

桥接器日志行包括：

```text
[real-bridge replan 0001] request=... append=... history=lowstate ...
```

请检查：

- 激活后出现 `history=lowstate`。
- lowstate 数据的时间戳延迟始终较小。
- 桥接器延迟接近服务器延迟与网络/请求开销之和。
- 参考缓冲区不会耗尽至零。

## 安全

进行实时扩散前，请先单独测试 HoloMotion。确保 Unitree 遥控器随时可用，
并在按下 B 键开始主动实时执行前验证紧急停止功能。
