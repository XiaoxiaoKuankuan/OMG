# BUMI 原生数据、训练、生成与渲染

本文描述完整链路：官方 OMG-Data 的 G1 `qpos36` 经 `robot_retarget` 按完整 episode 离线重定向为 BUMI `qpos28`，写成新的 LeRobotDataset v3，再由 OMG 以 BUMI 原生 `93D` 运动特征训练和生成。

## 数据约定

- G1 state：`root xyz(3) + root quaternion wxyz(4) + 29 joints = 36`。
- BUMI state：`root xyz(3) + root quaternion wxyz(4) + 21 joints = 28`。
- 所有内存 API、staging、Parquet、cache、checkpoint 输出和生成文件统一使用 `wxyz`；只有 `robot_retarget` 旧 CSV CLI 的显式兼容函数使用 `xyzw`。
- BUMI feature：root local position 3 + root rot6d 6 + joint DOF 21 + 21 个驱动关节 child body position 63，共 `93D`。
- BUMI 数据中 `action[t] = state[min(t+1,T-1)]`。

BUMI 关节顺序从 MuJoCo actuator→joint 映射和 `jnt_qposadr` 自动导出：

```text
waist_yaw_joint
l_arm_pitch_joint  l_arm_roll_joint  l_arm_yaw_joint  l_elbow_pitch_joint
r_arm_pitch_joint  r_arm_roll_joint  r_arm_yaw_joint  r_elbow_pitch_joint
l_leg_pitch_joint  l_leg_roll_joint  l_leg_yaw_joint  l_knee_pitch_joint
l_ankle_pitch_joint  l_ankle_roll_joint
r_leg_pitch_joint  r_leg_roll_joint  r_leg_yaw_joint  r_knee_pitch_joint
r_ankle_pitch_joint  r_ankle_roll_joint
```

## 服务器环境

程序没有内置 `/data0` 路径。服务器显式设置：

```bash
export SOURCE_OMG_DATA=/data0/user/liwei/OMG_data/OMG-Data
export BUMI_STAGE_ROOT=/data0/user/liwei/OMG_data/OMG-BUMI-Stage
export OMG_BUMI_DATA_ROOT=/data0/user/liwei/OMG_data/OMG-BUMI-Data
export BUMI_DATA_ROOT="$OMG_BUMI_DATA_ROOT"
export OMG_BUMI_MATERIALIZED_ROOT=/data0/user/liwei/OMG_data/OMG-BUMI-Materialized
export OMG_BUMI_REPO_ID=local/OMG-BUMI-Data
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

`OMG_BUMI_DATA_REVISION` 必须是 40 位十六进制转换版本标识；manifest SHA 必须来自新 BUMI manifest：

```bash
export OMG_BUMI_DATA_REVISION="$(git -C /home/weili/robot_retarget rev-parse HEAD)"
export OMG_BUMI_MANIFEST_SHA256="$(sha256sum "$OMG_BUMI_DATA_ROOT/meta/omg_bumi_manifest.json" | awk '{print $1}')"
```

## 转换官方数据

完整的 inspect、20/6/6 子集、单 episode、全量转换、resume、失败重试和 finalize 命令见 `robot_retarget/docs/omg_to_bumi_dataset.md`。

```bash
cd /home/weili/robot_retarget
conda activate robot_retargeter

python scripts/convert_omg_lerobot_to_bumi.py \
  --source-root "$SOURCE_OMG_DATA" \
  --stage-root "$BUMI_STAGE_ROOT" \
  --source-config config/robot/g1.yaml \
  --target-config config/robot/noetix_bumi_v1_3.yaml \
  --quality-config config/quality/omg_bumi.yaml \
  --split train --split val --split test \
  --workers 16 \
  --episodes-per-task 32 \
  --mp-start-method spawn

python scripts/finalize_bumi_lerobot.py \
  --stage-root "$BUMI_STAGE_ROOT" \
  --output-root "$BUMI_DATA_ROOT" \
  --source-root "$SOURCE_OMG_DATA" \
  --source-repo-id THU-MARS/OMG-Data \
  --source-revision 6e0dfbc1c5298bff14d4e2b1459ad678af0a38e7 \
  --source-manifest-sha256 be443885018180dda0f88ed874efc15cb3776a91148148baf49001d56a5ec855 \
  --source-config config/robot/g1.yaml \
  --target-config config/robot/noetix_bumi_v1_3.yaml \
  --quality-config config/quality/omg_bumi.yaml \
  --max-frames-per-data-file 1000000 \
  --data-files-per-chunk 1000 \
  --episodes-per-meta-file 10000 \
  --row-group-size 65536 \
  --compression zstd

python scripts/validate_bumi_lerobot.py \
  --dataset-root "$BUMI_DATA_ROOT" \
  --output /home/weili/OMG/outputs/integration/dataset_validation.json

HF_HUB_OFFLINE=1 python scripts/validate_bumi_lerobot_official.py \
  --dataset-root "$BUMI_DATA_ROOT" \
  --repo-id local/OMG-BUMI-Data \
  --output /home/weili/OMG/outputs/integration/dataset_validation_official.json
```

源数据目录只读；worker 仅原子写独立 staging artifact，最终 Parquet 仅由 finalize 单进程写入。每个 worker 只初始化一次 reader、G1/BUMI MuJoCo 模型、IK 和质量评估器；episode 之间仍完整 reset IK。finalizer 两遍扫描 staging，逐 episode 写 data/meta shard，并以 batch Welford 计算统计量，不将全量 state/action 驻留内存。frame `task_index` 按任务文本的确定性首次出现顺序重新编号。

新 finalizer 要求 staging 中存在明确的源 `task_index→text` 映射；早期版本 artifact 需要用当前 converter 重新生成，不能直接混入新数据集。

## 导出并验证 BUMI FK/渲染资产

```bash
cd /home/weili/OMG
source .venv/bin/activate
export PYTHONPATH=/home/weili/OMG/src

python tools/export_mjcf_kinematics.py \
  --mjcf /home/weili/robot_retarget/asset/robot/noetix_bumi_v1_3/mjcf/bumi3.xml \
  --output-mjcf assets/robots/bumi/bumi3.xml \
  --output-spec assets/robots/bumi/bumi_kinematics.json \
  --robot-name bumi \
  --copy-meshes

python tools/validate_mjcf_kinematics.py \
  --mjcf assets/robots/bumi/bumi3.xml \
  --spec assets/robots/bumi/bumi_kinematics.json \
  --samples 1000 \
  --seed 2026 \
  --output outputs/integration/fk_parity.json
```

固定 body 的变换折叠到最近的上游驱动关节 body。验证阈值为位置 `<1e-5 m`、旋转矩阵元素最大误差 `<1e-4`。STL 不提交到 Git，渲染前必须用 exporter 从相邻仓库复制。

## 计算真实 93D stats

`assets/stats/bumi_93d_stats.json` 是无数据单测使用的 identity 占位文件，禁止直接用于正式训练：

```bash
python -m omg.cli.generation.compute_stats \
  --data-config configs/generation/data/omg_bumi_lerobot_omnimodal.yaml \
  --representation-config configs/generation/representation/bumi_rot6d.yaml \
  --paths-config configs/generation/paths/default.yaml \
  --split train \
  --batch-size 256 \
  --device cuda \
  --output "$OMG_BUMI_DATA_ROOT/meta/bumi_93d_stats.json"

export OMG_BUMI_STATS_PATH="$OMG_BUMI_DATA_ROOT/meta/bumi_93d_stats.json"
```

输出 mean/std 长度必须均为 93，default state 必须为 root 7 + joint 21。

## Materialize cache v3

文本 cache：

```bash
python -m omg.cli.data.materialize_episode_cache \
  --data-config configs/generation/data/omg_bumi_lerobot.yaml \
  --representation-config configs/generation/representation/bumi_rot6d.yaml \
  --paths-config configs/generation/paths/default.yaml \
  --output-root "$OMG_BUMI_MATERIALIZED_ROOT/omg_bumi_episode_cache_v3_rot6d_seq60_hist10_k1" \
  --splits train val test \
  --device cuda
```

全模态 cache：

```bash
python -m omg.cli.data.materialize_episode_cache \
  --data-config configs/generation/data/omg_bumi_lerobot_omnimodal.yaml \
  --representation-config configs/generation/representation/bumi_rot6d.yaml \
  --paths-config configs/generation/paths/default.yaml \
  --output-root "$OMG_BUMI_MATERIALIZED_ROOT/omg_bumi_episode_cache_v3_omnimodal_rot6d_seq60_hist10_k1" \
  --splits train val test \
  --device cuda
```

BUMI 只读取 `omg.episode_cache.v3`，并校验 robot/state/feature/kinematics/representation 身份；不会误读 G1 v2 cache。G1 原有 v2 cache 仍保持可读。

## 训练与 checkpoint 迁移

从头训练 50M：

```bash
export OMG_MODELS_ROOT=/home/weili/OMG_models

CUDA_VISIBLE_DEVICES=0 python -m omg.cli.generation.train \
  exp=50m_bumi \
  data=omg_bumi_materialized_omnimodal \
  representation=bumi_rot6d \
  model.text_encoder.model_name="$OMG_MODELS_ROOT/t5-base-local"
```

100M、300M 将 `exp` 改为 `100m_bumi`、`300m_bumi`。从 G1 checkpoint 显式迁移：

```bash
CUDA_VISIBLE_DEVICES=0 python -m omg.cli.generation.train \
  exp=50m_bumi \
  data=omg_bumi_materialized_omnimodal \
  representation=bumi_rot6d \
  init_weights_only_ckpt=/absolute/path/to/g1.ckpt \
  init_weights_adapter=g1_to_bumi \
  model.text_encoder.model_name="$OMG_MODELS_ROOT/t5-base-local"
```

适配器只加载同名且 shape 完全相同的条件 encoder、time embedding、Transformer block 和 attention；显式跳过 motion input/output projection、history feature normalization/input projection、representation buffer 和机器人 head。完整报告写到 `${output_dir}/checkpoint_adaptation_report.json`。未分类 shape mismatch 会报错，不会裁剪 tensor 或静默继续。

## 生成和渲染

```bash
CUDA_VISIBLE_DEVICES=0 python -m omg.cli.generation.generate \
  --ckpt_path /absolute/path/to/bumi.ckpt \
  --exp 50m_bumi \
  --output_root outputs/integration/generation_samples \
  --output_tag sample_0 \
  --num_frames 120 \
  --text "walk forward" \
  --render_video \
  --fps 30 \
  --width 1280 \
  --height 720 \
  model.text_encoder.model_name="$OMG_MODELS_ROOT/t5-base-local"
```

生成目录包含通用 `qpos.npy`/`qpos` 及 robot_name、joint_names、fps、`quaternion_convention=wxyz`。只有 G1 额外保留 `qpos_36` 兼容字段；BUMI 输出 `[T,28]`。

## 真实 mini 集成顺序

本地 bundle 必须由环境变量提供：

```bash
export OMG_BUMI_DEV_BUNDLE=/path/to/omg_bumi_dev_bundle
test -e "$OMG_BUMI_DEV_BUNDLE"
```

可用集成执行器按规定顺序生成全部报告（`ROBOT_RETARGET_PYTHON` 必须指向安装了 robot_retarget 依赖的解释器）：

```bash
cd /home/weili/OMG
source .venv/bin/activate

export ROBOT_RETARGET_PYTHON=/absolute/path/to/robot_retargeter/bin/python
export OMG_T5_MODEL=/home/weili/OMG_models/t5-base-local

python tools/run_bumi_mini_integration.py \
  --bundle "$OMG_BUMI_DEV_BUNDLE" \
  --robot-retarget-root /home/weili/robot_retarget \
  --retarget-python "$ROBOT_RETARGET_PYTHON" \
  --t5-model "$OMG_T5_MODEL" \
  --workers 1 \
  --episodes-per-task 32 \
  --max-frames-per-data-file 1000000 \
  --data-files-per-chunk 1000 \
  --episodes-per-meta-file 10000 \
  --row-group-size 65536 \
  --device cuda
```

执行器支持目录或单个 zip/tar bundle，校验 archive SHA 后解压；重复运行会复用 episode staging。`--dry-run` 只打印命令，不表示集成已通过；`--skip-train`、`--skip-overfit`、`--skip-generate` 仅用于分段诊断。

在独立临时目录依次执行：解压 `source_mini`；单 worker `--max-episodes 1`；同一 staging resume 到 `--max-episodes 32`；finalize；validate；OMG loader；materialize；compute_stats；50-step smoke；限制 32～128 个样本过拟合；以 5 个 seed/history index 生成并渲染。保留：

```text
outputs/integration/retarget_report.json
outputs/integration/dataset_validation.json
outputs/integration/dataset_validation_official.json
outputs/integration/fk_parity.json
outputs/integration/train_smoke.log
outputs/integration/generation_samples/
```

50-step smoke：

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python -m omg.cli.generation.train \
  exp=50m_bumi \
  data=omg_bumi_materialized_omnimodal \
  representation=bumi_rot6d \
  trainer.max_steps=50 \
  trainer.val_check_interval=50 \
  model.text_encoder.model_name="$OMG_MODELS_ROOT/t5-base-local" \
  2>&1 | tee outputs/integration/train_smoke.log
```

失败 episode 查询 `conversion.sqlite3`；临时错误用相同输入身份加 `--retry-failed`。配置、代码或源 release 变化时使用新的 staging 根目录。

## 生产并行 benchmark

不要直接使用 `workers=100`。先固定同一批 100～500 个完整 episode 和同一套配置，分别以 `workers=1,4,8,16,32` 写入互不相同的 staging 根目录：

```bash
for workers in 1 4 8 16 32; do
  /usr/bin/time -v "$ROBOT_RETARGET_PYTHON" \
    /home/weili/robot_retarget/scripts/convert_omg_lerobot_to_bumi.py \
    --source-root "$SOURCE_OMG_DATA" \
    --stage-root "/path/to/benchmark/w${workers}" \
    --episode-list /path/to/fixed_100_500_episodes.txt \
    --workers "$workers" \
    --episodes-per-task 32 \
    --mp-start-method spawn \
    2>&1 | tee "/path/to/benchmark/w${workers}.log"
done
```

每档记录 frames/s、episodes/hour、峰值 RSS、源盘读取吞吐、失败率、不同 `worker_initialization_id` 的数量和模型加载次数。先根据源盘吞吐与峰值内存选择并行度，再运行全量转换；上述 100～500 episode benchmark 尚未在当前开发机执行。
