# G1 → 通用机器人/BUMI 迁移说明

## 保持兼容的 G1 接口

- `G1Kinematics` 及其资产路径保持不变。
- `G1MotionFeatureCodec`、`G1MotionRepresentation` 保留。
- `compose_qpos_36()`、`get_default_prev_qpos36()` 保留并委托给通用实现。
- `LeRobotG1MotionDataset` 继续返回 `qpos_36`、`prev_qpos_36`，同时新增通用 `qpos`、`prev_qpos`。
- G1 cache `omg.episode_cache.g1_motion.v2` 继续由 `EpisodeCachedG1MotionDataset` 读取。
- G1 生成结果继续额外保存 `qpos_36` 兼容字段和文件。
- realtime、tracking、planner、HoloMotion 和既有 G1 checkpoint 严格加载路径不切换到 BUMI。

## 新通用接口

- `MotionFeatureCodec`：维度来自 kinematics spec。
- `RobotMotionRepresentation.compose_qpos()` 与 `get_default_prev_qpos()`：不含机器人维度硬编码。
- `LeRobotMotionDataset`：通过 robot_name/state_dim/manifest/kinematics 参数化。
- `EpisodeCachedMotionDataset`：cache v3 使用通用 qpos、机器人和 kinematics 身份。
- 生成文件主字段为 `qpos`，并包含 robot_name、joint_names、fps 和 wxyz 约定。

## BUMI checkpoint 初始化

不要对 G1 checkpoint 和 BUMI 模型直接执行未检查的 `strict=False`。使用：

```bash
init_weights_only_ckpt=/path/g1.ckpt init_weights_adapter=g1_to_bumi
```

适配器提供完整 loaded/skipped/missing/unexpected 报告。机器人相关 motion projection、normalization 和 head 保持目标模型初始化；其余共享 tensor 必须同名且 shape 完全一致。

## Cache 与数据隔离

- G1 v2：`qpos_36.npy`、robot 隐含为 G1。
- BUMI v3：`qpos.npy`、robot_name=bumi、state_dim=28、feature_dim=93、kinematics_sha256、representation_name。
- BUMI loader 校验自定义 `omg_bumi_manifest.json` SHA-256，但不会与官方 G1 manifest 常量比较。
- 不允许将 G1 materialized root 指给 BUMI 配置；身份检查会拒绝。

切换到 BUMI 后必须重新生成数据集、93D stats 和 cache；不能复用 G1 125D stats/cache，也不能把 G1 motion input/output projection 当作已迁移参数。
