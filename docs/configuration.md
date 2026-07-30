# 配置

OMG 使用以下目录中的 Hydra 配置：

```text
configs/generation/
```

训练入口配置为：

```text
configs/generation/train.yaml
```

## 顶层配置组

```text
callbacks/       Lightning 回调。
data/            数据集和数据加载器配置。
denoiser/        Transformer 去噪器规模。
diffusion/       扩散过程配置。
exp/             实验预设。
logger/          日志记录器预设。
loss/            训练损失配置。
model/           Lightning 模块配置。
paths/           可移植路径默认值。
representation/ 动作表示和统计量。
trainer/         Lightning 训练器预设。
```

## 默认表示

```text
configs/generation/representation/125d.yaml
```

重要字段：

```yaml
feat_dim: 125
sequence_length: 60
num_prev_states: 10
stats_path: assets/stats/g1_125d_stats.json
```

`sequence_length=60` 表示每个扩散请求都会按照规划器 FPS 生成一个 60 帧的未来片段。

`stats_path` 指向由 `omg.cli.generation.compute_stats` 生成的归一化文件；
该文件未在本仓库中预计算。

## 实验预设

使用 `exp=<name>`：

```text
50m
100m
300m
500m
1b
```

每个预设都会设置模型大小、训练规模和输出命名。发布实验应保留在这些预设中；
一次性的消融实验应放在单独分支中，或使用外部配置覆盖。

## 训练器预设

```text
trainer=1gpu
trainer=2gpu
trainer=4gpu
trainer=8gpu
```

Lightning DDP 通过这些预设进行配置。每个进程的批大小位于：

```text
data.loader_opts.train.batch_size
data.loader_opts.val.batch_size
```

## 日志记录器预设

```text
logger=none
logger=wandb
```

使用 `logger=wandb` 前，请先设置 W&B 环境变量。

## 常用覆盖项

```bash
trainer.max_steps=200000
trainer.val_check_interval=2000
callbacks.checkpoint.every_n_train_steps=2000
data.loader_opts.train.batch_size=64
ckpt_path=outputs/run/checkpoints/last.ckpt
```
