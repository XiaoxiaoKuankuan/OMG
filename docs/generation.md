# 生成

主要的离线流水线入口为：

```bash
PYTHONPATH=src python -m omg.cli.pipeline.main
```

## PyTorch checkpoint 推理的历史来源

`omg.cli.generation.generate` 支持两种明确的历史初始化方式。

### Standalone deployment（推荐）

`--history_source default` 是默认模式。它不会实例化 Hydra 数据配置，也不会读取
LeRobotDataset 或 materialized cache。初始历史由机器人表示直接构造：

1. `representation.get_default_prev_qpos()` 从匹配的 stats JSON 读取默认根姿态和关节姿态；
2. representation 已有的 grounding 逻辑保证默认足底落地；
3. 已有 FK 计算默认姿态的 body position/quaternion；
4. `codec.prev_state_features_from_history()` 生成与训练完全相同的历史特征和 canonical root。

因此 standalone 推理不需要 OMG-Data、BUMI LeRobotDataset、materialized cache 或 Dev32。
它仍然需要：

- OMG 代码、机器人 MJCF 和 kinematics；
- checkpoint；
- 与训练完全匹配的 stats JSON；
- 本地 T5（使用文本条件时）；
- 可选 WAV 或 humanref 文件。

stats 是训练/推理坐标系的一部分，不能省略或用其他数据集的 stats 替代。建议同时通过
环境变量和 Hydra override 明确指定：

```bash
export OMG_BUMI_STATS_PATH=/path/to/bumi_93d_stats.json

PYTHONPATH="$PWD/src:$PWD" python -m omg.cli.generation.generate \
  --ckpt_path /path/to/sstep=200000.ckpt \
  --exp 300m_bumi \
  --history_source default \
  --text "walk forward slowly and then stop" \
  --num_frames 180 \
  --seed 0 \
  --cfg_text_scale 2.5 \
  --render_video \
  --width 640 \
  --height 480 \
  --camera_view iso \
  --follow_mode xy \
  representation.stats_path="$OMG_BUMI_STATS_PATH" \
  model.text_encoder.model_name=/path/to/t5-base-local
```

纯音乐 standalone 推理：

```bash
PYTHONPATH="$PWD/src:$PWD" python -m omg.cli.generation.generate \
  --ckpt_path /path/to/sstep=200000.ckpt \
  --exp 300m_bumi \
  --history_source default \
  --music /path/to/test.wav \
  --music_mod raw \
  --music_feature_type current35 \
  --num_frames 300 \
  --seed 0 \
  --cfg_audio_scale 2.5 \
  --render_video \
  --width 640 \
  --height 480 \
  --camera_view iso \
  --follow_mode xy \
  representation.stats_path="$OMG_BUMI_STATS_PATH" \
  model.text_encoder.model_name=/path/to/t5-base-local
```

standalone 模式没有显式文本时使用 null text；音频或 humanref 没有提供时也使用对应的
null condition，不会为了补条件而加载数据集。外部 WAV、feature 和 humanref 都从第0帧开始。

### Dataset-backed debug/evaluation

`--history_source dataset` 保留原有行为：从 validation dataset 的
`--history_val_index` 取得历史，并允许使用数据集 caption、对齐音频和 humanref。该模式用于：

- GT comparison；
- dataset-aligned benchmark/debug；
- 重现旧 generation 历史；
- `--save_gt_motion`、`--render_comparison_video`、`--aligned_gt_comparison`。

示例：

```bash
PYTHONPATH="$PWD/src:$PWD" python -m omg.cli.generation.generate \
  --ckpt_path /path/to/sstep=200000.ckpt \
  --exp 300m_bumi \
  --history_source dataset \
  --history_val_index 0 \
  --text "walk forward" \
  --num_frames 120 \
  --save_gt_motion \
  --render_comparison_video \
  representation.stats_path="$OMG_BUMI_STATS_PATH" \
  model.text_encoder.model_name=/path/to/t5-base-local \
  data=omg_bumi_lerobot_omnimodal
```

在 `--history_source default` 下使用任何 GT comparison 参数会立即报错，不会隐式切换到
dataset 模式。每次生成的 `metadata.json` 和 `reference_motion.npz` 都记录
`history_source`、stats绝对路径及 SHA-256。

离线流水线入口支持五种模式：

- `diffusion-only`：生成参考动作，并可选择渲染。
- `tracker-only`：通过 HoloMotion 跟踪已有参考动作。
- `sync`：扩散模型规划一个片段，跟踪器执行整个片段，随后开始下一次规划。
- `async`：跟踪器持续执行参考缓冲区中的动作，扩散模型则在缓冲区耗尽前重新规划。
- `offline-track`：生成一次参考动作，然后离线跟踪。

## 条件序列

使用 `--condition-sequence` 指定片段级条件：

```text
text: walk forward
text[5]: walk forward | text[3]: turn around
audio: inputs/audio/demo.wav
humanref: inputs/humanref/sample.npz
text+audio: wave arms+/path/to/audio.wav
text+humanref: imitate this+/path/to/ref.npz
```

`text[5]` 会在五个扩散片段中重复相同的文本条件。未带 `[N]` 的音频片段会扩展至
wav 时长。异步模式使用音频运行时，音频时间线按照跟踪器执行时间推进，
而不是按照规划器延迟推进。

## 仅扩散模式

```bash
PYTHONPATH=src python -m omg.cli.pipeline.main \
  --mode diffusion-only \
  --diffusion-onnx models/generation/onnx/50m/last_denoiser_step.onnx \
  --seed-motion /path/to/seed_motion.npz \
  --condition-sequence "text: walk forward" \
  --num-frames 120 \
  --video \
  --output-root outputs_pipeline
```

输出目录包含生成的参考动作和元数据。

## 同步模式

```bash
PYTHONPATH=src python -m omg.cli.pipeline.main \
  --mode sync \
  --diffusion-onnx models/generation/onnx/50m/last_denoiser_step.onnx \
  --holomotion-onnx models/holomotion/motion_tracking/model.onnx \
  --seed-motion /path/to/seed_motion.npz \
  --condition-sequence "text[4]: walk forward | text[2]: turn around" \
  --num-frames 300 \
  --video \
  --output-root outputs_pipeline
```

同步模式会在跟踪器每执行完一个片段后重新规划。

## 异步模式

```bash
PYTHONPATH=src python -m omg.cli.pipeline.main \
  --mode async \
  --diffusion-onnx models/generation/onnx/50m/last_denoiser_step.onnx \
  --holomotion-onnx models/holomotion/motion_tracking/model.onnx \
  --seed-motion /path/to/seed_motion.npz \
  --condition-sequence "text: walk forward" \
  --num-frames 300 \
  --async-replan-remaining-frames 40 \
  --video \
  --output-root outputs_pipeline
```

当跟踪器参考缓冲区剩余帧数不超过 `--async-replan-remaining-frames` 时，
异步模式会开始重新规划。异步模式默认启用 TensorRT FP16 和 DiT 缓存。

## 音频

对于 wav 驱动的条件：

```bash
--condition-sequence "audio: inputs/audio/demo.wav" --audio-type audio
```

对于启动时预计算的 wav 特征：

```bash
--condition-sequence "audio: inputs/audio/demo.wav" --audio-type feature
```

两种形式都在条件字符串中接收 wav 路径。

## 导出 ONNX

默认导出路径兼容 TensorRT，并为批量无分类器引导使用固定批大小 2。

```bash
PYTHONPATH=src python -m omg.cli.generation.export_onnx \
  --exp 50m \
  --ckpt_path outputs/<run>/checkpoints/last.ckpt \
  --output models/generation/onnx/50m/last_denoiser_step.onnx \
  --batch_size 2 \
  --device cuda
```

导出器会在 ONNX 模型旁写入元数据伴随文件。规划器使用这些元数据恢复序列长度、
特征维度、文本/音频设置、表示、扩散约定和注意力架构。

新检查点带有架构约定，而旧检查点没有；仅凭参数名称或形状无法推断 QK 归一化变更。
因此，导出旧检查点时必须声明 `none`、`cross-only`、`self-only` 或
`self-and-cross` 之一，并实例化匹配的去噪器。例如：

```bash
PYTHONPATH=src python -m omg.cli.generation.export_onnx \
  --exp 100m_omnimodal \
  --ckpt_path /path/to/legacy.ckpt \
  --legacy-attention-contract cross-only \
  denoiser.self_attention_qk_norm=false \
  denoiser.cross_attention_qk_norm=true
```

导出器会验证训练去噪器与包装器之间，以及包装器与 ONNX 之间的数值一致性。
任一检查超过容差时，它都会删除已生成的计算图并报错。

## TensorRT 运行时

流水线和实时规划器可以通过 ONNX Runtime TensorRT 提供程序运行导出的 ONNX
去噪步骤。异步模式默认启用 TensorRT FP16 和 DiT 缓存。

常用提供程序顺序：

```bash
--providers TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider
```

实时规划器默认设置：

- 启用 TensorRT FP16。
- 启用 DiT 缓存。
- TensorRT 引擎缓存位于 `tensorrt_engine_cache/realtime_planner`。

## 渲染

常用渲染参数：

```bash
--video
--camera-view iso
--follow-mode xy
--scene-preset studio
--video-width 1280
--video-height 720
```

`--follow-mode xy` 是流水线渲染的默认设置，通常也是观察行走动作最实用的视角。
