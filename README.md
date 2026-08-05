<div align="center">
  <p>
    <img src="assets/title.svg" alt="OMG：面向通用人形机器人控制的全模态动作生成" width="900">
  </p>
  <p>
    <strong>OMG：面向通用人形机器人控制的全模态动作生成</strong>官方仓库。
  </p>
  <p>
    <a href="https://arxiv.org/"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg" alt="arXiv"></a>
    <a href="https://tsinghua-mars-lab.github.io/OMG/"><img src="https://img.shields.io/badge/Website-Page-Green" alt="网站"></a>
    <a href="https://huggingface.co/THU-MARS/OMG"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-OMG-FFD21E" alt="Hugging Face 模型"></a>
    <a href="https://huggingface.co/datasets/THU-MARS/OMG-Data"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-OMG--Data-FFD21E" alt="Hugging Face 数据集"></a>
  </p>
  <p>
    <img src="assets/teaser.png" alt="OMG 预览图" width="900">
  </p>
</div>

BUMI 原生数据转换、训练、生成和渲染请参阅 [BUMI 原生训练文档](docs/bumi_native_training.md)，G1 兼容迁移说明见 [MIGRATION_NOTES.md](MIGRATION_NOTES.md)。

## 新闻

🚩 **2026 年 7 月**：OMG 荣获 **ExWBC@RSS 2026 Oral、RoboData@RSS 2026 Spotlight**，祝贺！<br>
🚩 **2026 年 6 月**：我们发布了 [OMG](https://arxiv.org/abs/2606.10340) 的预印本、代码和数据。

## 流程

通常的端到端工作流如下：

1. 安装环境。
2. 下载 OMG-Data、模型检查点和 HoloMotion 制品。
3. 物化 OMG-Data，以加速训练。
4. 计算归一化统计量。
5. 训练扩散模型。
6. 导出 ONNX，用于 TensorRT/CUDA 推理。
7. 运行生成模式或完整流水线模式。
8. 运行基准测试。
9. 按需部署到 G1 机器人。

## 1. 安装

```bash
cd /path/to/OMG
make venv
source .venv/bin/activate
make install

export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
```

中国大陆网络环境请使用：

```bash
make install-cn
```

手动 `uv` 命令和按任务选择的可选额外依赖，请参阅[安装](docs/installation.md)。

## 2. 下载数据和制品

OMG-Data 以官方 LeRobotDataset v3 数据集的形式发布。预训练模型检查点和基准测试评估器
可从官方 [THU-MARS/OMG Hugging Face 仓库](https://huggingface.co/THU-MARS/OMG)获取：

- [OMG-Data](https://huggingface.co/datasets/THU-MARS/OMG-Data)
- [物化的 OMG-Data]()（即将发布）
- [OMG 检查点和评估器](https://huggingface.co/THU-MARS/OMG/tree/main)

| 模型 | 训练步数 | 检查点 |
| --- | ---: | --- |
| OMG 50M | 90,000 | [`checkpoints/50m/sstep=090000.ckpt`](https://huggingface.co/THU-MARS/OMG/blob/main/checkpoints/50m/sstep%3D090000.ckpt) |
| OMG 100M | 100,000 | [`checkpoints/100m/sstep=100000.ckpt`](https://huggingface.co/THU-MARS/OMG/blob/main/checkpoints/100m/sstep%3D100000.ckpt) |
| OMG 300M | 55,000 | [`checkpoints/300m/sstep=055000.ckpt`](https://huggingface.co/THU-MARS/OMG/blob/main/checkpoints/300m/sstep%3D055000.ckpt) |
| OMG 500M | 50,000 | [`checkpoints/500m/sstep=050000.ckpt`](https://huggingface.co/THU-MARS/OMG/blob/main/checkpoints/500m/sstep%3D050000.ckpt) |

预训练基准测试评估器位于
[`evaluator/step_004000.pt`](https://huggingface.co/THU-MARS/OMG/blob/main/evaluator/step_004000.pt)。
发布文件的校验和列在
[`SHA256SUMS`](https://huggingface.co/THU-MARS/OMG/blob/main/SHA256SUMS)。

文本条件训练和生成需要 Hugging Face `t5-base` 文本编码器。默认情况下，配置从
`${OMG_MODELS_ROOT}/t5-base-local` 加载。离线运行时请下载
[t5-base](https://huggingface.co/google-t5/t5-base)，也可以用其他本地路径或
Hugging Face 模型 ID 覆盖 `model.text_encoder.model_name`。

OMG 不会再分发 HoloMotion 权重。请从 [HoloMotion 官方仓库](https://github.com/HorizonRobotics/HoloMotion)
或 [HoloMotion Hugging Face 制品](https://huggingface.co/HorizonRobotics/HoloMotion_models)
下载 HoloMotion 模型。

推荐的本地布局：

```text
data/OMG-Data/
  data/
  meta/
  materialized/
models/
  generation/
  evaluator/
  t5-base-local/
  holomotion/
    motion_tracking/model.onnx
    velocity_tracking/model.onnx
```

使用外部存储时，请显式设置根目录：

```bash
export OMG_DATA_ROOT=/path/to/OMG-Data
export OMG_MATERIALIZED_ROOT=/path/to/OMG-Data/materialized
export OMG_MODELS_ROOT=/path/to/OMG-models

hf download THU-MARS/OMG-Data \
  --type dataset \
  --revision 6e0dfbc1c5298bff14d4e2b1459ad678af0a38e7 \
  --local-dir "$OMG_DATA_ROOT"

hf cache verify THU-MARS/OMG-Data \
  --type dataset \
  --revision 6e0dfbc1c5298bff14d4e2b1459ad678af0a38e7 \
  --local-dir "$OMG_DATA_ROOT" \
  --fail-on-missing-files
```

数据配置固定了官方 Hub 提交和发布清单的 SHA-256。因此，`OMG_DATA_ROOT` 必须包含
该确切的 LeRobot v3 快照；初始化时会拒绝旧的或仅部分替换的本地数据集。

## 3. 物化数据

物化会预计算帧级片段运动学。完整训练建议使用物化数据，因为它能避免重复的源数据解析
和正向运动学（FK）计算，同时保留精确、完备的步长为 1 的窗口集合，
且不会重复存储相互重叠的窗口张量。

请从固定版本的源 OMG-Data 在本地生成缓存。只有当预计算缓存的 v2 源身份与活动配置
完全匹配时，该缓存才有效：

```bash
scripts/materialize_omg_data.sh --overwrite
```

使用前请验证每份清单、片段索引和帧张量：

```bash
PYTHONPATH=src python -m omg.cli.data.validate_episode_cache \
  "$OMG_MATERIALIZED_ROOT/omg_episode_cache_v2_rot6d_seq60_hist10_k1"
```

片段缓存 v2 会固定源 LeRobot 仓库版本。未固定版本的 v1 缓存必须重建；
它们不能作为发布版本的训练输入。

使用物化数据训练：

```bash
data=omg_data_materialized
```

进行小规模调试或使用自定义微型数据集时，可直接使用源数据：

```bash
data=omg_data_lerobot
```

## 4. 计算统计量

训练前请计算归一化统计量。默认表示配置要求生成的统计量文件位于：

```text
assets/stats/g1_125d_stats.json
```

物化读取器会枚举与源读取器相同的完备窗口，同时复用缓存的 FK 张量。

```bash
PYTHONPATH=src python -m omg.cli.generation.compute_stats \
  --data-config configs/generation/data/omg_data_materialized.yaml \
  --representation-config configs/generation/representation/125d.yaml \
  --paths-config configs/generation/paths/default.yaml \
  --device cuda \
  --output assets/stats/g1_125d_stats.json
```

要使用四块 GPU 计算精确统计量，请通过 torchrun 运行相同命令：

```bash
torchrun --standalone --nproc-per-node=4 -m omg.cli.generation.compute_stats \
  --data-config configs/generation/data/omg_data_materialized.yaml \
  --representation-config configs/generation/representation/125d.yaml \
  --paths-config configs/generation/paths/default.yaml \
  --device cuda \
  --output assets/stats/g1_125d_stats.json
```

只要训练数据、表示、序列长度或预处理发生变化，就应重新计算此文件。

## 5. 训练

50M 模型训练示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=src python -m omg.cli.generation.train \
  exp=50m \
  data=omg_data_materialized \
  trainer=4gpu \
  logger=wandb \
  exp_name=50m_release_train
```

模型规模配置位于 `configs/generation/exp/`：

```text
50m.yaml  100m.yaml  300m.yaml  500m.yaml  1b.yaml
```

恢复、初始化和配置详情请参阅[训练](docs/training.md)。

## 6. 导出 ONNX

从检查点导出兼容 TensorRT 的去噪步骤：

```bash
PYTHONPATH=src python -m omg.cli.generation.export_onnx \
  --exp 50m \
  --ckpt_path outputs/50m_release_train/checkpoints/last.ckpt \
  --output models/generation/onnx/50m/last_denoiser_step.onnx \
  --batch_size 2 \
  --device cuda
```

导出器会在 ONNX 文件旁写入元数据伴随文件。运行时规划器使用该文件恢复序列长度、
条件维度、表示和扩散设置。新检查点会记录其注意力架构，导出器会拒绝不匹配的配置。
对于在此约定建立之前创建的检查点，必须显式声明训练语义；例如，
仅使用交叉注意力 QK 归一化训练的检查点需要：

```bash
PYTHONPATH=src python -m omg.cli.generation.export_onnx \
  --exp 100m_omnimodal \
  --ckpt_path /path/to/legacy.ckpt \
  --legacy-attention-contract cross-only \
  denoiser.self_attention_qk_norm=false \
  denoiser.cross_attention_qk_norm=true
```

导出采用故障关闭策略：保留制品前，它会对训练去噪器、兼容 TensorRT 的包装器
以及生成的 ONNX 计算图进行数值比较。

## 7. 生成与跟踪

使用 HoloMotion 跟踪的离线异步生成：

```bash
PYTHONPATH=src python -m omg.cli.pipeline.main \
  --mode async \
  --diffusion-onnx models/generation/onnx/50m/last_denoiser_step.onnx \
  --holomotion-onnx models/holomotion/motion_tracking/model.onnx \
  --seed-motion /path/to/seed_motion.npz \
  --condition-sequence "text: walk forward" \
  --num-frames 300 \
  --video \
  --output-root outputs_pipeline
```

支持的流水线模式：

- `diffusion-only`
- `tracker-only`
- `sync`
- `async`
- `offline-track`

请参阅[生成](docs/generation.md)和[跟踪](docs/tracking.md)。

## 8. 基准测试

报告基于评估器的分布和检索指标的基准测试使用预训练评估器检查点：

```text
https://huggingface.co/THU-MARS/OMG/blob/main/evaluator/step_004000.pt
```

推荐的本地路径：

```text
models/evaluator/pretrained.ckpt
```

已验证的发布清单在 `assets/benchmarks/mixed_modalities_all_v2` 下进行版本管理。
仅在确实要定义新的基准测试版本时，才从固定版本的数据集重新生成：

```bash
PYTHONPATH=src python -m omg.cli.evaluation.prepare_samples \
  --data omg_data_lerobot_omnimodal \
  --output_dir outputs/benchmark_samples/mixed_modalities_all_v2
```

运行文本、音频、人体参考或制品基准测试：

```bash
PYTHONPATH=src python -m omg.cli.generation.benchmark text \
  --exp 50m \
  --ckpt_path outputs/50m_release_train/checkpoints/last.ckpt \
  --evaluator_checkpoint models/evaluator/pretrained.ckpt \
  --samples_path assets/benchmarks/mixed_modalities_all_v2/text_test_1024.jsonl \
  --output_dir outputs/benchmarks/50m_text
```

各模态专用命令和跟踪器执行评估请参阅[基准测试](docs/benchmark.md)。

## 9. 部署

实时部署使用：

- G1 Orin 上的 HoloMotion 部署进程。
- GPU 工作站上的 OMG 实时规划器服务器。
- G1 Orin 上的 OMG 实机桥接器。

实体机器人部署时，建议优先使用 HoloMotion 速度跟踪模型：

```text
models/holomotion/velocity_tracking/model.onnx
```

完整启动顺序请参阅 [G1 实时部署](docs/realtime_g1.md)。

## 文档

- [安装](docs/installation.md)
- [制品](docs/artifacts.md)
- [数据](docs/data.md)
- [训练](docs/training.md)
- [生成](docs/generation.md)
- [跟踪](docs/tracking.md)
- [G1 实时部署](docs/realtime_g1.md)
- [基准测试](docs/benchmark.md)
- [配置](docs/configuration.md)
- [开发](docs/development.md)

## 许可证

本项目基于 [MIT 许可证](LICENSE)发布。

## 引用

如果我们的代码对您有帮助，请考虑引用我们的工作：
```
@article{huang2026omg,
  title={OMG: Omni-Modal Motion Generation for Generalist Humanoid Control},
  author={Huang, Siqiao and Lee, Kun-Ying and Qiao, Dongming and He, Guanqi and Wang, Zhenyu and Li, Yitang and Zhu, Shaoting and Zhao, Hang},
  journal={arXiv preprint arXiv:2606.10340},
  year={2026}
}
```
