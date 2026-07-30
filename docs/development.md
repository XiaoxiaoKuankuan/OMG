# 开发

## 包结构

```text
src/omg/
  benchmarks/    基准指标、运行器、报告和评估器推理。
  callbacks/      Lightning 回调。
  cli/            面向用户的命令行入口。
  core/           小型共享日志、路径和张量工具。
  data/           规范的 LeRobotDataset v3 读取器及派生片段缓存。
  generation/     训练数据、去噪器、扩散、损失和导出。
  motion/         G1 动作表示工具。
  pipeline/       离线扩散和跟踪器编排。
  realtime/       ZMQ 协议、规划器服务和实时缓冲区。
  render/         MuJoCo 渲染。
  robots/         G1 运动学和常量。
  runtime/        ONNX 提供程序设置等运行时辅助工具。
  tracking/       HoloMotion 跟踪器集成。
```

外部基线复现代码保存在 `repro/baselines` 分支。
发布用的 `main` 分支仅保留评估生成的 `qpos_36` 输出所需的制品基准接口。

## CLI 入口

生成：

```text
omg.cli.generation.train
omg.cli.generation.generate
omg.cli.generation.export_onnx
omg.cli.generation.benchmark
omg.cli.generation.physical_benchmark
```

流水线：

```text
omg.cli.pipeline.main
```

跟踪：

```text
omg.cli.tracking.holomotion
omg.cli.tracking.export_holomotion_clip
```

实时：

```text
omg.cli.realtime.planner_server
omg.cli.realtime.holomotion_real_bridge
omg.cli.realtime.holomotion_dry_run
omg.cli.realtime.policy_node_smoke_driver
```

## 测试

使用以下命令运行测试套件：

```bash
PYTHONPATH=src pytest
```

轻量级语法检查：

```bash
python3 -m compileall -q src/omg tests
```

提交前：

```bash
git diff --check
PYTHONPATH=src pytest
```
