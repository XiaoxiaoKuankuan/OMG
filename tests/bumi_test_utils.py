from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from omg.motion.representation import BumiMotionRepresentation


TEST_REPO_ID = "local/OMG-BUMI-Test"
TEST_REVISION = "a" * 40


def standing_qpos(frames: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    representation = BumiMotionRepresentation(num_prev_states=2, canonical_frame_idx=1)
    qpos = representation.get_default_prev_qpos(1, torch.device("cpu"), dtype)[0, :1]
    return qpos.expand(int(frames), -1).clone()


def make_bumi_lerobot_dataset(root: Path, *, frames_per_split: int = 6) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(root)
    data_dir = root / "data" / "chunk-000"
    episode_dir = root / "meta" / "episodes" / "chunk-000"
    data_dir.mkdir(parents=True)
    episode_dir.mkdir(parents=True)
    splits = ("train", "val", "test")
    states_per_episode = []
    episode_rows = []
    cursor = 0
    for episode_index, split in enumerate(splits):
        states = standing_qpos(frames_per_split).numpy().astype(np.float32)
        states[:, 0] += np.linspace(0.0, 0.02 * episode_index, frames_per_split, dtype=np.float32)
        states_per_episode.append(states)
        episode_rows.append(
            {
                "episode_index": episode_index,
                "length": frames_per_split,
                "dataset_from_index": cursor,
                "dataset_to_index": cursor + frames_per_split,
                "tasks": [f"test motion {episode_index}"],
                "omg/split": split,
                "omg/source_id": f"episode-{episode_index}",
                "omg/dataset": "synthetic-test",
                "omg/segment_index": 0,
                "omg/source_start_frame": 0,
                "omg/source_end_frame": frames_per_split,
                "omg/has_text": True,
                "omg/has_audio": True,
                "omg/has_humanref": True,
            }
        )
        cursor += frames_per_split
    states = np.concatenate(states_per_episode, axis=0)
    actions = np.concatenate(
        [np.concatenate((item[1:], item[-1:]), axis=0) for item in states_per_episode], axis=0
    )
    frame_indices = np.tile(np.arange(frames_per_split, dtype=np.int64), len(splits))
    episode_indices = np.repeat(np.arange(len(splits), dtype=np.int64), frames_per_split)
    frame_table = pa.table(
        {
            "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32(), 28)),
            "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), 28)),
            "index": pa.array(np.arange(states.shape[0], dtype=np.int64)),
            "episode_index": pa.array(episode_indices),
            "frame_index": pa.array(frame_indices),
            "timestamp": pa.array(frame_indices.astype(np.float32) / 30.0),
            "task_index": pa.array(episode_indices),
            "omg.audio.feature": pa.array(
                np.zeros((states.shape[0], 35), dtype=np.float32).tolist(),
                type=pa.list_(pa.float32(), 35),
            ),
            "omg.condition.has_audio": pa.array(np.ones(states.shape[0], dtype=np.bool_)),
            "omg.humanref.motion": pa.array(
                np.zeros((states.shape[0], 66), dtype=np.float32).tolist(),
                type=pa.list_(pa.float32(), 66),
            ),
            "omg.condition.has_humanref": pa.array(np.ones(states.shape[0], dtype=np.bool_)),
        }
    )
    pq.write_table(frame_table, data_dir / "file-000.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(episode_rows), episode_dir / "file-000.parquet", compression="zstd")
    pq.write_table(
        pa.table({"task_index": pa.array([0, 1, 2]), "task": pa.array([row["tasks"][0] for row in episode_rows])}),
        root / "meta" / "tasks.parquet",
        compression="zstd",
    )
    joint_names = list(BumiMotionRepresentation().joint_names)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "bumi",
        "fps": 30.0,
        "total_episodes": 3,
        "total_frames": int(states.shape[0]),
        "total_tasks": 3,
        "chunks_size": 3,
        "splits": {name: f"{index}:{index + 1}" for index, name in enumerate(splits)},
        "features": {
            "observation.state": {"dtype": "float32", "shape": [28]},
            "action": {"dtype": "float32", "shape": [28]},
            "omg.audio.feature": {"dtype": "float32", "shape": [35]},
            "omg.condition.has_audio": {"dtype": "bool", "shape": [1]},
            "omg.humanref.motion": {"dtype": "float32", "shape": [66]},
            "omg.condition.has_humanref": {"dtype": "bool", "shape": [1]},
        },
        "state_dim": 28,
        "action_dim": 28,
        "quaternion_convention": "wxyz",
        "joint_names": joint_names,
    }
    manifest = {
        "format": "LeRobotDataset-v3.0",
        "repo_id": TEST_REPO_ID,
        "robot_name": "bumi",
        "state_dim": 28,
        "action_dim": 28,
        "quaternion_convention": "wxyz",
        "joint_names": joint_names,
        "source": {"repo_id": "THU-MARS/OMG-Data", "revision": "source-test"},
        "conversion": {"test": True},
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, sort_keys=True) + "\n", encoding="utf-8")
    (root / "meta" / "stats.json").write_text("{}\n", encoding="utf-8")
    manifest_path = root / "meta" / "omg_bumi_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "root": root,
        "repo_id": TEST_REPO_ID,
        "revision": TEST_REVISION,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "states": states,
        "actions": actions,
        "joint_names": joint_names,
    }


def open_bumi_dataset(bundle: dict, *, split: str = "train", omnimodal: bool = True):
    from omg.data.lerobot_dataset import LeRobotBumiMotionDataset

    return LeRobotBumiMotionDataset(
        dataset_root=bundle["root"],
        split=split,
        repo_id=bundle["repo_id"],
        revision=bundle["revision"],
        manifest_sha256=bundle["manifest_sha256"],
        sequence_duration=4.0 / 30.0,
        fps=30.0,
        num_prev_states=2,
        canonical_frame_idx=1,
        use_audio=omnimodal,
        use_human_motion=omnimodal,
        rotation_representation="rot6d",
        train_window_policy="exhaustive",
    )
