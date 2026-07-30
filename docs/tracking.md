# 跟踪

OMG 将 HoloMotion 集成为下游 G1 动作跟踪器。跟踪既可以直接用于参考片段，
也可以作为同步/异步生成的一部分。

## 仅跟踪器模式

```bash
PYTHONPATH=src python -m omg.cli.tracking.holomotion \
  --reference /path/to/reference_motion.npz \
  --holomotion-onnx models/holomotion/motion_tracking/model.onnx \
  --target-fps 50 \
  --video \
  --video-path outputs_tracking/reference_tracker.mp4 \
  --output outputs_tracking/reference_tracker.npz
```

输入参考动作必须包含 `qpos_36`。如果参考文件不含 FPS 元数据，
请传入 `--reference-fps`。

## 流水线跟踪器模式

流水线命令也提供跟踪器模式：

```bash
PYTHONPATH=src python -m omg.cli.pipeline.main \
  --mode tracker-only \
  --seed-motion /path/to/reference_motion.npz \
  --holomotion-onnx models/holomotion/motion_tracking/model.onnx \
  --num-frames 300 \
  --video
```

当参考动作应来自扩散规划器时，请使用 `sync`、`async` 或 `offline-track`。

## 导出部署片段

要将生成的参考动作转换为 HoloMotion 部署所用的 `motion_data` 格式：

```bash
PYTHONPATH=src python -m omg.cli.tracking.export_holomotion_clip \
  --reference outputs_pipeline/run/reference_motion.npz \
  --target-fps 50 \
  --output /home/unitree/holomotion/deployment/unitree_g1_ros2_29dof/src/motion_data/01_reference.npz
```

更改部署动作片段后，请重启 HoloMotion 部署进程。

## 提供程序

跟踪器的默认提供程序：

```text
TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider
```

TensorRT 不可用时，请仅使用 CUDA：

```bash
--providers CUDAExecutionProvider,CPUExecutionProvider
```

## 输出

跟踪器输出包括：

- 跟踪器执行后的 `qpos_36`
- 参考动作元数据
- 可选视频
- 可用时的执行过程时序和质量元数据

使用跟踪器执行后的输出，评估生成的参考动作在多大程度上能够被下游策略实际跟踪。
