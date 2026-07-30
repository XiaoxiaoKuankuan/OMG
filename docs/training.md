# 训练

训练使用 PyTorch Lightning，并采用 `configs/generation` 下的 Hydra 配置。
请先下载 OMG-Data 并将其放在 `data/OMG-Data`，或设置
`OMG_DATA_ROOT` 和 `OMG_MATERIALIZED_ROOT`。

文本条件运行还需要 Hugging Face `t5-base` 文本编码器。默认配置要求本地副本位于：

```text
${OMG_MODELS_ROOT}/t5-base-local
```

可使用以下方式指定其他本地路径或 Hugging Face 模型 ID：

```bash
model.text_encoder.model_name=/path/to/t5-base
```

## 最简命令

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=src python -m omg.cli.generation.train \
  exp=50m \
  data=omg_data_materialized \
  trainer=4gpu \
  logger=wandb \
  exp_name=50m_release_train
```

主配置为：

```text
configs/generation/train.yaml
```

常用覆盖项：

```bash
trainer.max_steps=200000
trainer.val_check_interval=2000
callbacks.checkpoint.every_n_train_steps=2000
data.loader_opts.train.batch_size=64
```

## 模型规模

实验预设：

```text
configs/generation/exp/50m.yaml
configs/generation/exp/100m.yaml
configs/generation/exp/300m.yaml
configs/generation/exp/500m.yaml
configs/generation/exp/1b.yaml
```

每个实验都会选择一种 Transformer 去噪器规模和训练超参数。

## 恢复训练

使用 `ckpt_path` 完整恢复 Lightning 训练：

```bash
PYTHONPATH=src python -m omg.cli.generation.train \
  exp=50m \
  data=omg_data_materialized \
  trainer=4gpu \
  logger=wandb \
  ckpt_path=outputs/50m_release_train/checkpoints/last.ckpt
```

仅在只初始化模型权重、而不恢复优化器、调度器、数据加载器或全局步数状态时，
使用 `init_weights_only_ckpt`：

```bash
PYTHONPATH=src python -m omg.cli.generation.train \
  exp=50m \
  data=omg_data_materialized \
  trainer=4gpu \
  logger=wandb \
  init_weights_only_ckpt=outputs/source/checkpoints/last.ckpt
```

## W&B 日志

```bash
export WANDB_API_KEY="..."
export WANDB_MODE=online
```

使用以下配置禁用日志：

```bash
logger=none
```

## 检查点

查找最近的检查点：

```bash
find outputs/<exp_name>/checkpoints -maxdepth 1 -name "*.ckpt" | sort | tail -20
```

从检查点导出：

```bash
PYTHONPATH=src python -m omg.cli.generation.export_onnx \
  --exp 50m \
  --ckpt_path outputs/<exp_name>/checkpoints/last.ckpt \
  --output models/generation/onnx/50m/last_denoiser_step.onnx \
  --batch_size 2 \
  --device cuda
```

## 验证

Lightning 验证按照 `trainer.val_check_interval` 运行。快速调试时，
可同时缩短验证间隔和检查点间隔：

```bash
trainer.val_check_interval=200 callbacks.checkpoint.every_n_train_steps=200
```

短时调试运行仅用于代码检查。不要比较极短运行所生成动作的质量。
