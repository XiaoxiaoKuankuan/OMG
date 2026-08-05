# OMG G1→BUMI 数据与原生生成实现报告

## 实现范围

本次实现跨两个相邻仓库：

- `robot_retarget`：完整 episode 的内存 FK→语义关键点→顺序 warm-start IK、质量拒绝、事务 resume、原子 staging 和确定性 LeRobotDataset v3 finalize。
- `OMG`：通用 kinematics/feature/representation/dataset/cache/model/loss/generation/render 主路径，以及 BUMI 28D state、93D feature、配置和显式 G1→BUMI checkpoint adapter。

官方源数据只读；批量流程不使用 CSV/PKL，不拆分 episode 内帧，不静默删除坏帧，也不由多个 worker 并发写最终 Parquet。

## 已实际执行

### robot_retarget

```text
PYTHONPATH=. .../robot_retargeter/bin/pytest -q tests
10 passed in 2.52s
```

还实际执行了合成 LeRobot v3 的 inspect→单 episode convert→重复 resume→finalize→validate。结果为 1 episode/5 frames、第二次 resume scheduled=0、SQLite attempts 保持 1、验证 `valid=true`，两次 conversion manifest SHA-256 相同。

Phase A commit：`25f76eb1a83d1cf4ce16f3a1649b40fc9f33d419`。

### OMG

```text
PYTHONPATH=src:. .venv/bin/pytest -q tests
224 passed, 1 warning in 5.26s
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

当前机器没有设置 `OMG_BUMI_DEV_BUNDLE`，也没有提供真实 `source_mini`。因此没有宣称完成以下真实数据/GPU 集成：

- 真实 mini 的 1 episode 和 32 episode IK 转换。
- 真实 mini finalize/validate 产物。
- 真实数据 materialize 和真实 93D stats。
- 50 个正式 Trainer step。
- 32～128 样本小数据过拟合。
- 真实 checkpoint 的 5 个生成/渲染样本。

所以 `outputs/integration/retarget_report.json`、`dataset_validation.json`、`train_smoke.log` 和 `generation_samples/` 尚未由真实 bundle 产生。完整执行命令见 `docs/bumi_native_training.md`。

`tools/run_bumi_mini_integration.py` 已实现可恢复的真实 mini 编排和规定产物路径，但由于 bundle 缺失，本机只验证了入口解析/单测，没有把 dry-run 当作实际集成结果。

## 已知风险

- 全量 IK 失败率和质量阈值只能在官方数据上校准；严格模式拒绝异常 episode 是设计行为。
- 26MB BUMI STL 属于外部大资产，未提交 Git；渲染前要运行 exporter 从相邻 `robot_retarget` 复制。
- 仓库自带 BUMI stats 是测试用 identity 占位，正式训练前必须计算真实 stats 并设置 `OMG_BUMI_STATS_PATH`。
- BUMI translucent GT overlay 尚未实现；单视频和左右 comparison 可渲染。human-reference skeleton comparison 的机器人侧仍是原 G1 专用路径。
- 真实官方 LeRobot Python 包未安装；已使用兼容 v3 reader和 HuggingFace `Dataset.from_parquet` 完成 round-trip。
