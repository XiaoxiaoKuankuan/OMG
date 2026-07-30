# 数据与模型制品

## Hugging Face 发布资源

公开制品发布在 Hugging Face 上：

```text
OMG-Data:        https://huggingface.co/datasets/THU-MARS/OMG-Data
OMG checkpoints: https://huggingface.co/THU-MARS/OMG/tree/main/checkpoints
OMG evaluator:   https://huggingface.co/THU-MARS/OMG/blob/main/evaluator/step_004000.pt
```

## 默认本地布局

可移植的默认布局如下：

```text
OMG/
  data/
    OMG-Data/
      data/
      meta/
      materialized/
  models/
    generation/
    evaluator/
    holomotion/
```

Hydra 配置从 `configs/generation/paths/default.yaml` 读取这些路径。
可通过环境变量覆盖：

```bash
export OMG_DATA_ROOT=/path/to/OMG-Data
export OMG_MATERIALIZED_ROOT=/path/to/OMG-Data/materialized
export OMG_MODELS_ROOT=/path/to/OMG-models
```

## 必需的运行时制品

运行生成和跟踪演示前，请准备：

- 导出的 OMG 扩散 ONNX 模型及其元数据伴随文件；
- HoloMotion G1 跟踪器 ONNX 模型；
- 包含 `qpos_36` 的 G1 种子动作文件；
- 使用 `omg.cli.generation.compute_stats` 生成的
  `assets/stats/g1_125d_stats.json`。

对于基于评估器的基准指标，请准备：

- 从 OMG 评估器发布资源下载的 `models/evaluator/pretrained.ckpt`。

训练时，请准备 `OMG-Data/` 下的官方 LeRobot v3 数据集，或
`materialized/` 下的帧级片段运动学缓存。
