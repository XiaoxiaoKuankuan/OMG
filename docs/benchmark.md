# 基准测试

基准测试评估生成的参考动作以及跟踪器执行后的动作。

## 主入口

```bash
PYTHONPATH=src python -m omg.cli.generation.benchmark --help
```

基准测试运行器位于：

```text
src/omg/benchmarks/
```

它们支持文本、音频、人体参考、基于制品以及跟踪器执行后的评估路径。

## 评估器检查点

基于评估器的分布和检索指标需要已发布的 OMG 评估器检查点：

```text
https://huggingface.co/THU-MARS/OMG/blob/main/evaluator/step_004000.pt
```

推荐的本地路径：

```text
models/evaluator/pretrained.ckpt
```

文本检索指标还会使用 [T5-3B 文本编码器](https://huggingface.co/google-t5/t5-3b)。
默认情况下，基准测试加载 `t5-3b`，也可以从 `OMG_T5_3B_MODEL` 读取本地路径。

离线运行时：

```bash
hf download google-t5/t5-3b --local-dir models/t5-3b-local
export OMG_T5_3B_MODEL=models/t5-3b-local
```

## 样本准备

已验证的发布清单提交在：

```text
assets/benchmarks/mixed_modalities_all_v2/
```

论文/发布版本对比请使用这些文件。如果确实要定义新的基准测试版本，
请从固定版本的公开 LeRobotDataset v3 源重新生成候选样本：

```bash
PYTHONPATH=src python -m omg.cli.evaluation.prepare_samples \
  --data omg_data_lerobot_omnimodal \
  --output_dir outputs/benchmark_samples/mixed_modalities_all_v2
```

生成的 `omg.benchmark.sample.v2` 行包含仓库版本、数据划分、片段、精确窗口和源身份。
运行器会根据 LeRobot 元数据解析每个字段，并拒绝过期或不匹配的清单。因此，
检查点评估和制品评估可以共享相同的固定行，而无需依赖计算机本地索引或私有源路径。

准备命令会保留发布群组协议（12 个文本群组、5 个音频群组和 11 个人体参考群组），
同时将每个选中行解析到其规范的 LeRobot 源数据集。

请将已提交的清单用作基准测试运行器的输入，例如使用
`--samples_path assets/benchmarks/mixed_modalities_all_v2/text_test_1024.jsonl`。
通过 `--datasets` 传入的数据集名称必须是 `omg/dataset` 片段列中的精确值。
外部基线复现脚本位于 `repro/baselines` 分支；`main` 仅保留基准测试制品接口。

## 物理指标

对动作制品运行物理指标：

```bash
PYTHONPATH=src python -m omg.cli.generation.physical_benchmark \
  --motion outputs_pipeline/run/reference_motion.npz
```

代表性的物理指标包括：

- `contact_sliding_speed`：脚部接触地面时的平均水平速度。
- `body_jerk_mean`：身体位置三阶有限差分幅值的平均值。
- `foot_ground_error`：最低鞋底代理点到地平面的有符号距离绝对值的平均值。

默认统计量路径为：

```text
assets/stats/g1_125d_stats.json
```

运行需要加载动作表示的基准测试前，请使用 `omg.cli.generation.compute_stats`
生成此文件。

## 跟踪器执行评估

跟踪器执行指标用于判断下游跟踪器能否跟随生成的参考动作。使用各运行器提供的
跟踪器参数，在基准测试运行器中启用跟踪器执行：

```text
--tracker_executed
--tracker_holomotion_onnx ...
```

运行器会将跟踪器执行后的制品和指标写入基准测试输出旁。

## 输出文件

常见的基准测试输出：

- `benchmark.json`
- `metrics.json`
- 各指标的 JSON 文件，例如 `physical_metrics.json`
- 启用时的跟踪器执行过程制品

请将 JSON 输出作为表格的规范来源。Markdown 摘要应从 JSON 重新生成，
而不是手动编辑。
