# OMG G1→BUMI 数据与原生生成实现报告

## 实现范围

本次实现跨两个相邻仓库：

- `robot_retarget`：完整 episode 的内存 FK→语义关键点→顺序 warm-start IK、质量拒绝、事务 resume、原子 staging，以及两遍流式、分片写入的确定性 LeRobotDataset v3 finalizer。reader、G1/BUMI MuJoCo 模型、IK 和质量评估器按 worker 复用，但每个 episode 都完整 reset。
- `OMG`：通用 kinematics/feature/representation/dataset/cache/model/loss/generation/render 主路径，以及 BUMI 28D state、93D feature、配置和显式 G1→BUMI checkpoint adapter。

官方源数据只读；批量流程不使用 CSV/PKL，不拆分 episode 内帧，不静默删除坏帧，也不由多个 worker 并发写最终 Parquet。

## 已实际执行

### robot_retarget

```text
PYTHONPATH=. /home/weili/miniconda3/envs/robot_retargeter/bin/python -m pytest -q tests
30 passed in 11.68s
```

指定的 worker、流式分片、task remap 和官方 loader 专项测试也单独执行：

```text
pytest -q tests/test_worker_context_reuse.py tests/test_streaming_finalizer_shards.py \
  tests/test_task_index_remap.py tests/test_official_lerobot_loader.py
8 passed in 7.54s
```

还实际对 `/home/weili/OMG_dev_data/omg_bumi_raw_sample/OMG-Data` 的 episode 0、1 完成单 worker 同 batch 转换。两者共 390 帧，诊断中的 `worker_pid` 和 `worker_initialization_id` 相同；以 200 帧 data shard 上限 finalize 后得到 2 个 data shard、2 个 episode metadata shard。重复转换报告 `scheduled=0, batches=0`。自定义验证和官方 `lerobot==0.4.4` loader 均为 `valid=true`；官方报告为 2 episodes、390 frames、2 tasks、30 FPS、state/action shape `[28]`、task range `[0,1]`。

Phase A commit：`25f76eb1a83d1cf4ce16f3a1649b40fc9f33d419`。

### OMG

```text
PYTHONPATH=src:. .venv/bin/python -m pytest -q tests
225 passed, 1 warning in 5.30s
```

warning 来自单元测试直接调用未绑定 Trainer 的 Lightning `self.log()`；loss、反向梯度和生成均为 finite。

实际执行 1000 个合法随机 qpos 的 Generic/PyTorch FK 对 MuJoCo：

```json
{
  "max_position_error_m": 1.2783302971719479e-08,
  "max_rotation_matrix_error": 5.1506365172926394e-08,
  "samples": 1000,
  "valid": true
}
```

报告位于 `outputs/integration/fk_parity.json`。BUMI 专项测试还实际覆盖 LeRobot/HuggingFace Parquet round-trip、多模态逐帧对齐、93D encode/decode/FK、cache v3 materialize/read、checkpoint adapter、单 train step、`[B,T,28]` 生成和 MuJoCo 视频渲染。

## 未执行

当前机器没有设置完整的 `OMG_BUMI_DEV_BUNDLE`。现有本地真实样本只提供源数据第一个 data shard，不能代表完整 20/6/6 或 Dev32 多模态 bundle。因此没有宣称完成以下真实数据/GPU 集成：

- 真实 mini 的 1 episode 和 32 episode IK 转换。
- 真实 mini finalize/validate 产物。
- 真实数据 materialize 和真实 93D stats。
- 50 个正式 Trainer step。
- 32～128 样本小数据过拟合。
- 真实 checkpoint 的 5 个生成/渲染样本。

所以 `outputs/integration/retarget_report.json`、`dataset_validation.json`、`dataset_validation_official.json`、`train_smoke.log` 和 `generation_samples/` 尚未由完整真实 bundle 端到端产生。完整执行命令见 `docs/bumi_native_training.md`。

`tools/run_bumi_mini_integration.py` 已传递 worker batching 和 finalizer sharding 参数，并在自定义验证后增加官方 loader 验证。本机用真实样本目录执行了 `--dry-run`，确认命令编排和参数传递，但没有把 dry-run 当作实际集成结果。

## 已知风险

- 全量 IK 失败率、吞吐、峰值 RSS 和质量阈值仍只能在完整官方数据上校准；尚未执行固定 100～500 episode 的 workers=1/4/8/16/32 benchmark。
- 26MB BUMI STL 属于外部大资产，未提交 Git；渲染前要运行 exporter 从相邻 `robot_retarget` 复制。
- 仓库自带 BUMI stats 是测试用 identity 占位，正式训练前必须计算真实 stats 并设置 `OMG_BUMI_STATS_PATH`。
- BUMI translucent GT overlay 尚未实现；单视频和左右 comparison 可渲染。human-reference skeleton comparison 的机器人侧仍是原 G1 专用路径。
- 当前 `robot_retargeter` 环境已明确安装并实测 `lerobot==0.4.4`；合成测试同时覆盖官方 `LeRobotDatasetMetadata`、`LeRobotDataset` 和 HuggingFace `Dataset.from_parquet`。
