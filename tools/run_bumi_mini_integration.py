#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


OFFICIAL_REVISION = "6e0dfbc1c5298bff14d4e2b1459ad678af0a38e7"
OFFICIAL_MANIFEST_SHA256 = "be443885018180dda0f88ed874efc15cb3776a91148148baf49001d56a5ec855"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_roots(root: Path) -> list[Path]:
    roots = []
    if (root / "meta/info.json").is_file():
        roots.append(root)
    if root.is_dir():
        roots.extend(path.parent.parent for path in root.rglob("meta/info.json"))
    return sorted(set(path.resolve() for path in roots))


def resolve_source_mini(bundle: Path, extraction_root: Path) -> Path:
    bundle = bundle.expanduser().resolve()
    if not bundle.exists():
        raise FileNotFoundError(f"OMG_BUMI_DEV_BUNDLE does not exist: {bundle}")
    roots = _dataset_roots(bundle) if bundle.is_dir() else []
    if len(roots) == 1:
        return roots[0]
    if len(roots) > 1:
        named = [path for path in roots if path.name == "source_mini"]
        if len(named) == 1:
            return named[0]
        raise ValueError(f"Bundle contains multiple LeRobot datasets; expected one source_mini: {roots}")

    archive = bundle if bundle.is_file() else None
    if archive is None:
        candidates = sorted(
            path
            for path in bundle.iterdir()
            if path.is_file()
            and (path.suffix.lower() == ".zip" or ".tar" in "".join(path.suffixes).lower())
        )
        if len(candidates) != 1:
            raise ValueError(f"Cannot identify one source_mini archive under {bundle}: {candidates}")
        archive = candidates[0]
    extraction_root.mkdir(parents=True, exist_ok=True)
    marker = extraction_root / ".archive_sha256"
    digest = _sha256(archive)
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() != digest:
        raise ValueError(
            f"Extraction directory belongs to a different archive: {extraction_root}; use a new --work-root"
        )
    if not marker.is_file():
        if any(extraction_root.iterdir()):
            raise ValueError(f"Refusing to unpack into non-empty unpinned directory: {extraction_root}")
        shutil.unpack_archive(str(archive), str(extraction_root))
        marker.write_text(digest + "\n", encoding="utf-8")
    roots = _dataset_roots(extraction_root)
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one LeRobot source_mini after extraction, got {roots}")
    return roots[0]


class Runner:
    def __init__(self, *, dry_run: bool, env: dict[str, str]):
        self.dry_run = bool(dry_run)
        self.env = dict(env)
        self.commands: list[list[str]] = []

    def run(self, command: list[str], *, cwd: Path, log_path: Path | None = None) -> None:
        command = [str(value) for value in command]
        self.commands.append(command)
        print("+", " ".join(command), flush=True)
        if self.dry_run:
            return
        if log_path is None:
            subprocess.run(command, cwd=cwd, env=self.env, check=True)
            return
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
            return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run the real OMG→BUMI mini integration pipeline.")
    parser.add_argument("--bundle", type=Path, default=os.environ.get("OMG_BUMI_DEV_BUNDLE"))
    parser.add_argument("--robot-retarget-root", type=Path, default=repo_root.parent / "robot_retarget")
    parser.add_argument("--retarget-python", default=os.environ.get("ROBOT_RETARGET_PYTHON"))
    parser.add_argument("--omg-python", default=sys.executable)
    parser.add_argument("--work-root", type=Path, default=repo_root / "outputs/integration/mini_work")
    parser.add_argument("--integration-root", type=Path, default=repo_root / "outputs/integration")
    parser.add_argument("--t5-model", type=Path, default=os.environ.get("OMG_T5_MODEL"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--episodes-per-task", type=int, default=32)
    parser.add_argument("--max-frames-per-data-file", type=int, default=1_000_000)
    parser.add_argument("--data-files-per-chunk", type=int, default=1000)
    parser.add_argument("--episodes-per-meta-file", type=int, default=10_000)
    parser.add_argument("--row-group-size", type=int, default=65_536)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-steps", type=int, default=50)
    parser.add_argument("--overfit-samples", type=int, default=64)
    parser.add_argument("--overfit-steps", type=int, default=500)
    parser.add_argument("--generation-count", type=int, default=5)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-overfit", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.bundle is None:
        raise SystemExit("--bundle or OMG_BUMI_DEV_BUNDLE is required")
    if not args.retarget_python:
        raise SystemExit("--retarget-python or ROBOT_RETARGET_PYTHON is required")
    positive = (
        "workers",
        "generation_count",
        "episodes_per_task",
        "max_frames_per_data_file",
        "data_files_per_chunk",
        "episodes_per_meta_file",
        "row_group_size",
    )
    if any(getattr(args, name) <= 0 for name in positive):
        raise SystemExit("Worker, batching, sharding, and generation values must be positive")
    if (not args.skip_train or not args.skip_overfit or not args.skip_generate) and args.t5_model is None:
        raise SystemExit("--t5-model or OMG_T5_MODEL is required for train/generate stages")

    repo_root = Path(__file__).resolve().parents[1]
    retarget_root = args.robot_retarget_root.expanduser().resolve()
    work_root = args.work_root.expanduser().resolve()
    integration_root = args.integration_root.expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    integration_root.mkdir(parents=True, exist_ok=True)
    source_root = resolve_source_mini(args.bundle, work_root / "source_unpacked")
    stage_root = work_root / "stage"
    dataset_root = work_root / "bumi_lerobot"
    cache_parent = work_root / "materialized"
    cache_root = cache_parent / "omg_bumi_episode_cache_v3_omnimodal_rot6d_seq60_hist10_k1"
    stats_path = work_root / "bumi_93d_stats.json"
    train_root = work_root / "train"
    overfit_root = work_root / "overfit"

    retarget_commit = subprocess.check_output(
        ["git", "-C", str(retarget_root), "rev-parse", "HEAD"], text=True
    ).strip()
    env = dict(os.environ)
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env.setdefault(variable, "1")
    env.update(
        {
            "PYTHONPATH": str(repo_root / "src"),
            "OMG_BUMI_DATA_ROOT": str(dataset_root),
            "BUMI_DATA_ROOT": str(dataset_root),
            "BUMI_STAGE_ROOT": str(stage_root),
            "SOURCE_OMG_DATA": str(source_root),
            "OMG_BUMI_MATERIALIZED_ROOT": str(cache_parent),
            "OMG_BUMI_REPO_ID": "local/OMG-BUMI-Mini",
            "OMG_BUMI_DATA_REVISION": retarget_commit,
            "OMG_BUMI_STATS_PATH": str(stats_path),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    runner = Runner(dry_run=args.dry_run, env=env)
    convert = retarget_root / "scripts/convert_omg_lerobot_to_bumi.py"
    common_convert = [
        args.retarget_python,
        convert,
        "--source-root", source_root,
        "--stage-root", stage_root,
        "--source-config", retarget_root / "config/robot/g1.yaml",
        "--target-config", retarget_root / "config/robot/noetix_bumi_v1_3.yaml",
        "--quality-config", retarget_root / "config/quality/omg_bumi.yaml",
        "--episodes-per-task", str(args.episodes_per_task),
    ]
    runner.run([*common_convert, "--max-episodes", "1", "--workers", "1"], cwd=retarget_root)
    runner.run(
        [*common_convert, "--max-episodes", "32", "--workers", str(args.workers)],
        cwd=retarget_root,
    )
    if not args.dry_run:
        shutil.copy2(stage_root / "conversion_manifest.json", integration_root / "retarget_report.json")

    runner.run(
        [
            args.retarget_python,
            retarget_root / "scripts/finalize_bumi_lerobot.py",
            "--stage-root", stage_root,
            "--output-root", dataset_root,
            "--source-root", source_root,
            "--source-repo-id", "THU-MARS/OMG-Data",
            "--source-revision", OFFICIAL_REVISION,
            "--source-manifest-sha256", OFFICIAL_MANIFEST_SHA256,
            "--source-config", retarget_root / "config/robot/g1.yaml",
            "--target-config", retarget_root / "config/robot/noetix_bumi_v1_3.yaml",
            "--quality-config", retarget_root / "config/quality/omg_bumi.yaml",
            "--max-frames-per-data-file", str(args.max_frames_per_data_file),
            "--data-files-per-chunk", str(args.data_files_per_chunk),
            "--episodes-per-meta-file", str(args.episodes_per_meta_file),
            "--row-group-size", str(args.row_group_size),
            "--overwrite",
        ],
        cwd=retarget_root,
    )
    runner.run(
        [
            args.retarget_python,
            retarget_root / "scripts/validate_bumi_lerobot.py",
            "--dataset-root", dataset_root,
            "--output", integration_root / "dataset_validation.json",
        ],
        cwd=retarget_root,
    )
    runner.run(
        [
            args.retarget_python,
            retarget_root / "scripts/validate_bumi_lerobot_official.py",
            "--dataset-root", dataset_root,
            "--repo-id", "local/OMG-BUMI-Mini",
            "--output", integration_root / "dataset_validation_official.json",
        ],
        cwd=retarget_root,
    )
    if not args.dry_run:
        env["OMG_BUMI_MANIFEST_SHA256"] = _sha256(dataset_root / "meta/omg_bumi_manifest.json")
        runner.env = dict(env)

    runner.run(
        [
            args.omg_python,
            repo_root / "tools/export_mjcf_kinematics.py",
            "--mjcf", retarget_root / "asset/robot/noetix_bumi_v1_3/mjcf/bumi3.xml",
            "--output-mjcf", repo_root / "assets/robots/bumi/bumi3.xml",
            "--output-spec", repo_root / "assets/robots/bumi/bumi_kinematics.json",
            "--robot-name", "bumi",
            "--copy-meshes",
        ],
        cwd=repo_root,
    )
    if not args.dry_run:
        env["OMG_BUMI_KINEMATICS_SHA256"] = _sha256(
            repo_root / "assets/robots/bumi/bumi_kinematics.json"
        )
        runner.env = dict(env)
    runner.run(
        [
            args.omg_python,
            repo_root / "tools/validate_mjcf_kinematics.py",
            "--mjcf", repo_root / "assets/robots/bumi/bumi3.xml",
            "--spec", repo_root / "assets/robots/bumi/bumi_kinematics.json",
            "--samples", "1000",
            "--seed", "2026",
            "--output", integration_root / "fk_parity.json",
        ],
        cwd=repo_root,
    )
    runner.run(
        [
            args.omg_python,
            "-m", "omg.cli.data.materialize_episode_cache",
            "--data-config", "configs/generation/data/omg_bumi_lerobot_omnimodal.yaml",
            "--representation-config", "configs/generation/representation/bumi_rot6d.yaml",
            "--paths-config", "configs/generation/paths/default.yaml",
            "--output-root", cache_root,
            "--splits", "train", "val", "test",
            "--device", args.device,
            "--overwrite",
        ],
        cwd=repo_root,
    )
    runner.run(
        [
            args.omg_python,
            "-m", "omg.cli.generation.compute_stats",
            "--data-config", "configs/generation/data/omg_bumi_materialized_omnimodal.yaml",
            "--representation-config", "configs/generation/representation/bumi_rot6d.yaml",
            "--paths-config", "configs/generation/paths/default.yaml",
            "--split", "train",
            "--batch-size", "64",
            "--device", args.device,
            "--output", stats_path,
        ],
        cwd=repo_root,
    )

    common_train = [
        args.omg_python,
        "-m", "omg.cli.generation.train",
        "exp=50m_bumi",
        "data=omg_bumi_materialized_omnimodal",
        "representation=bumi_rot6d",
        f"model.text_encoder.model_name={args.t5_model}",
        "data.loader_opts.train.batch_size=4",
        "data.loader_opts.train.num_workers=0",
        "data.loader_opts.train.prefetch_factor=null",
        "trainer.devices=1",
    ]
    if args.device == "cpu":
        common_train.extend(("trainer.accelerator=cpu", "trainer.precision=32-true"))
    if not args.skip_train:
        runner.run(
            [
                *common_train,
                f"paths.output_root={train_root}",
                f"trainer.max_steps={args.train_steps}",
                f"trainer.val_check_interval={args.train_steps}",
                f"callbacks.checkpoint.every_n_train_steps={args.train_steps}",
            ],
            cwd=repo_root,
            log_path=integration_root / "train_smoke.log",
        )
    if not args.skip_overfit:
        runner.run(
            [
                *common_train,
                f"paths.output_root={overfit_root}",
                f"data.limit_each_trainset={args.overfit_samples}",
                "trainer.overfit_batches=1.0",
                f"trainer.max_steps={args.overfit_steps}",
                f"trainer.val_check_interval={args.overfit_steps}",
            ],
            cwd=repo_root,
            log_path=integration_root / "overfit.log",
        )
    if not args.skip_generate:
        checkpoint = train_root / "bumi_transformer_guided_50m_rot6d_omnimodal/checkpoints/last.ckpt"
        if not args.dry_run and not checkpoint.is_file():
            raise FileNotFoundError(f"50-step checkpoint not found: {checkpoint}")
        for index in range(args.generation_count):
            runner.run(
                [
                    args.omg_python,
                    "-m", "omg.cli.generation.generate",
                    "--ckpt_path", checkpoint,
                    "--exp", "50m_bumi",
                    "--output_root", integration_root / "generation_samples",
                    "--output_tag", f"sample_{index}",
                    "--history_val_index", str(index),
                    "--seed", str(index),
                    "--num_frames", "60",
                    "--text", "walk forward",
                    "--render_video",
                    f"model.text_encoder.model_name={args.t5_model}",
                ],
                cwd=repo_root,
            )
    summary = {
        "source_root": str(source_root),
        "stage_root": str(stage_root),
        "dataset_root": str(dataset_root),
        "cache_root": str(cache_root),
        "stats_path": str(stats_path),
        "dry_run": bool(args.dry_run),
        "commands": runner.commands,
    }
    (integration_root / "mini_integration_commands.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
