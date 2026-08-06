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


def make_bumi_lerobot_dataset(
    root: Path,
    *,
    frames_per_split: int = 6,
    episodes_per_split: int = 1,
    shard_by_episode: bool = False,
    distinct_modalities: bool = False,
) -> dict:
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
    split_episode_ranges = {}
    cursor = 0
    episode_index = 0
    for split in splits:
        split_episode_start = episode_index
        for _ in range(int(episodes_per_split)):
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
            episode_index += 1
        split_episode_ranges[split] = (split_episode_start, episode_index)
    states = np.concatenate(states_per_episode, axis=0)
    actions = np.concatenate(
        [np.concatenate((item[1:], item[-1:]), axis=0) for item in states_per_episode], axis=0
    )
    total_episodes = len(episode_rows)
    frame_indices = np.tile(np.arange(frames_per_split, dtype=np.int64), total_episodes)
    episode_indices = np.repeat(np.arange(total_episodes, dtype=np.int64), frames_per_split)
    audio_features = np.zeros((states.shape[0], 35), dtype=np.float32)
    human_motion = np.zeros((states.shape[0], 66), dtype=np.float32)
    has_audio = np.ones(states.shape[0], dtype=np.bool_)
    has_humanref = np.ones(states.shape[0], dtype=np.bool_)
    if distinct_modalities:
        global_indices = np.arange(states.shape[0], dtype=np.float32)
        audio_features[:, 0] = global_indices
        human_motion[:, 0] = global_indices + 1000.0
        has_audio = (np.arange(states.shape[0]) % 3) != 0
        has_humanref = (np.arange(states.shape[0]) % 4) != 0
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
                audio_features.tolist(),
                type=pa.list_(pa.float32(), 35),
            ),
            "omg.condition.has_audio": pa.array(has_audio),
            "omg.humanref.motion": pa.array(
                human_motion.tolist(),
                type=pa.list_(pa.float32(), 66),
            ),
            "omg.condition.has_humanref": pa.array(has_humanref),
        }
    )
    episode_table = pa.Table.from_pylist(episode_rows)
    if shard_by_episode:
        for current_episode, row in enumerate(episode_rows):
            pq.write_table(
                frame_table.slice(int(row["dataset_from_index"]), int(row["length"])),
                data_dir / f"file-{current_episode:03d}.parquet",
                compression="zstd",
            )
            pq.write_table(
                episode_table.slice(current_episode, 1),
                episode_dir / f"file-{current_episode:03d}.parquet",
                compression="zstd",
            )
    else:
        pq.write_table(frame_table, data_dir / "file-000.parquet", compression="zstd")
        pq.write_table(episode_table, episode_dir / "file-000.parquet", compression="zstd")
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array(np.arange(total_episodes, dtype=np.int64)),
                "task": pa.array([row["tasks"][0] for row in episode_rows]),
            }
        ),
        root / "meta" / "tasks.parquet",
        compression="zstd",
    )
    joint_names = list(BumiMotionRepresentation().joint_names)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "bumi",
        "fps": 30.0,
        "total_episodes": total_episodes,
        "total_frames": int(states.shape[0]),
        "total_tasks": total_episodes,
        "chunks_size": total_episodes,
        "splits": {
            name: f"{split_episode_ranges[name][0]}:{split_episode_ranges[name][1]}"
            for name in splits
        },
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
        "audio_features": audio_features,
        "human_motion": human_motion,
        "has_audio": has_audio,
        "has_humanref": has_humanref,
        "episode_rows": episode_rows,
        "split_episode_ranges": split_episode_ranges,
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
