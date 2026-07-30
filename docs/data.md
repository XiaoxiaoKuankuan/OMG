# 数据

OMG 唯一支持的源数据集约定是官方
[THU-MARS/OMG-Data](https://huggingface.co/datasets/THU-MARS/OMG-Data)
LeRobotDataset v3 发布版本。训练、统计量计算、生成样本选择和基准测试都从该约定解析样本。

## 规范发布版本

请将数据集放在 `data/OMG-Data`，或设置：

```bash
export OMG_DATA_ROOT=/path/to/OMG-Data

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

预期目录结构如下：

```text
OMG-Data/
  data/chunk-*/file-*.parquet
  meta/info.json
  meta/stats.json
  meta/tasks.parquet
  meta/episodes/chunk-*/file-*.parquet
```

`observation.state` 存储 G1 `qpos_36`；文本由标准 LeRobot 任务表表示。
对齐的可选模态为 `omg.audio.feature` 和 `omg.humanref.motion`，
并带有显式的逐帧掩码。片段元数据还包含基准测试清单所使用的不可变源身份字段。

两个公开配置都固定到相同的数据集版本：

```text
configs/generation/data/omg_data_lerobot.yaml
configs/generation/data/omg_data_lerobot_omnimodal.yaml
```

第一个配置只启用文本。第二个配置启用文本、音频和人体参考条件。
加载器会同时验证完整的 40 字符 Hub 版本和 `meta/omg_manifest.json` 的 SHA-256；
如果 `OMG_DATA_ROOT` 指向更旧或不同的本地快照，则会在读取任何训练或基准测试样本前报错。

## G1 表示

源状态包含 36 个值：

- 根节点位置：3；
- `wxyz` 格式的根节点四元数：4；
- G1 关节位置：29。

默认模型表示由 `configs/generation/representation/125d.yaml` 定义。
它将 LeRobot 窗口转换为 125 维模型特征，并使用
`assets/stats/g1_125d_stats.json` 进行归一化。

使用以下命令直接从规范数据集计算统计量：

```bash
PYTHONPATH=src python -m omg.cli.generation.compute_stats \
  --data-config configs/generation/data/omg_data_lerobot.yaml \
  --representation-config configs/generation/representation/125d.yaml \
  --paths-config configs/generation/paths/default.yaml \
  --output assets/stats/g1_125d_stats.json
```

只要固定的数据集版本、表示、窗口长度或预处理发生变化，就应重新计算统计量。

## 可选片段缓存

对于大规模训练，OMG 可以从固定版本的 LeRobot 源派生帧级片段缓存。
该缓存是优化层，而不是另一种源数据集格式：其清单会记录 LeRobot 身份，
并且在使用前必须通过严格验证器。

```bash
export OMG_MATERIALIZED_ROOT=/path/to/OMG-Data/materialized
scripts/materialize_omg_data.sh --overwrite
PYTHONPATH=src python -m omg.cli.data.validate_episode_cache \
  "$OMG_MATERIALIZED_ROOT/omg_episode_cache_v2_rot6d_seq60_hist10_k1"
```

`omg_data_materialized` 和 `omg_data_materialized_omnimodal` 只读取此派生缓存。
删除缓存不会移除规范数据；可从固定版本的 LeRobot 发布资源重新创建。
旧版 v1 缓存没有可验证的源版本，因此不会被接受；请将其重建到 v2 路径。

## 基准测试样本身份

基准测试清单使用 `omg.benchmark.sample.v2`。每一行都会固定：

- `repo_id`、`revision` 和 `split`；
- `episode_index`、`window_start` 和 `num_frames`；
- 源数据集、源 ID、分段索引和源帧区间。

运行器会根据 LeRobot 元数据解析完整身份，只要任一字段不一致就会报错。
本地列表索引、私有文件系统路径以及已移除的 `.npz + labels + info.yaml` 布局
都不是有效的基准测试身份。已验证的发布集合位于
`assets/benchmarks/mixed_modalities_all_v2`；其摘要记录数据版本、清单哈希、
群组数量和帧级验证结果。

使用以下命令准备全部三类基准测试群组：

```bash
PYTHONPATH=src python -m omg.cli.evaluation.prepare_samples \
  --data omg_data_lerobot_omnimodal \
  --output_dir outputs/benchmark_samples/mixed_modalities_all_v2
```

条件资格遵循发布协议：文本需要非空任务，短片段使用动作有效掩码；
每个请求的音频帧或人体参考帧都必须设置相应的条件掩码。
采样在原始发布基准群组间保持均衡；四个特定语言的 BEAT2 源组共同构成一个人体参考群组，
从而使协议仍可与 `mixed_modalities_all_v1` 比较。

## 外部推理条件

独立生成可以接收显式的动作、音频或人体参考制品。它们是推理输入和输出，
而不是训练数据集。OMG 不会根据文件名推断同级文件，也不会静默修复缺失条件；
调用方必须显式传入每个制品。

对于新的训练数据，请将其发布为 LeRobotDataset v3 版本，并包含相同的必需状态、
任务、模态、掩码、数据划分和不可变身份字段。不要添加其他仓库专用的源加载器。
