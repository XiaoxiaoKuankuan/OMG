from __future__ import annotations

import numpy as np
import pytest

from omg.cli.data.materialize_episode_cache import write_episode_cache
from omg.data.episode_cache_inspect import inspect_episode_cache
from tests.bumi_test_utils import make_bumi_lerobot_dataset, open_bumi_dataset


def _split_frame_interval(bundle: dict, split: str) -> tuple[int, int]:
    episode_start, episode_end = bundle["split_episode_ranges"][split]
    rows = bundle["episode_rows"][episode_start:episode_end]
    return int(rows[0]["dataset_from_index"]), int(rows[-1]["dataset_to_index"])


def _make_sharded_dataset(tmp_path):
    return make_bumi_lerobot_dataset(
        tmp_path / "dataset",
        frames_per_split=5,
        episodes_per_split=2,
        shard_by_episode=True,
        distinct_modalities=True,
    )


@pytest.mark.parametrize("split", ["val", "test"])
def test_grouped_kinematics_uses_split_local_frame_indices(tmp_path, split: str) -> None:
    bundle = _make_sharded_dataset(tmp_path)
    dataset = open_bumi_dataset(bundle, split=split)
    global_start, global_end = _split_frame_interval(bundle, split)

    assert dataset.frame_dataset_offset == global_start
    assert dataset.frame_dataset_offset > 0
    groups = list(dataset.iter_episode_kinematics_groups(max_frames=100, device="cpu"))

    assert len(groups) == 1
    group = groups[0]
    expected_frames = global_end - global_start
    assert [episode["episode_index"] for episode in group["episodes"]] == list(
        range(*bundle["split_episode_ranges"][split])
    )
    assert [
        int(episode["data_end_row"]) - int(episode["data_start_row"])
        for episode in group["episodes"]
    ] == [5, 5]
    assert group["qpos"].shape == (expected_frames, 28)
    assert group["qpos"].numel() > 0
    assert group["qpos"].isfinite().all()
    assert group["body_pos_w"].shape[0] == expected_frames
    assert group["body_quat_w"].shape[0] == expected_frames
    assert group["audio_features"].shape == (expected_frames, 35)
    assert group["human_motion"].shape == (expected_frames, 66)
    assert group["has_audio"].shape == (expected_frames,)
    assert group["has_human_motion"].shape == (expected_frames,)

    np.testing.assert_allclose(group["qpos"].cpu().numpy(), bundle["states"][global_start:global_end])
    np.testing.assert_allclose(
        group["audio_features"].cpu().numpy(),
        bundle["audio_features"][global_start:global_end],
    )
    np.testing.assert_allclose(
        group["human_motion"].cpu().numpy(),
        bundle["human_motion"][global_start:global_end],
    )
    np.testing.assert_array_equal(
        group["has_audio"].cpu().numpy(),
        bundle["has_audio"][global_start:global_end],
    )
    np.testing.assert_array_equal(
        group["has_human_motion"].cpu().numpy(),
        bundle["has_humanref"][global_start:global_end],
    )


def test_grouped_kinematics_reports_split_interval_context(tmp_path) -> None:
    bundle = _make_sharded_dataset(tmp_path)
    dataset = open_bumi_dataset(bundle, split="val")
    dataset.frame_dataset_offset += 1

    with pytest.raises(IndexError) as error:
        next(dataset.iter_episode_kinematics_groups(max_frames=100, device="cpu"))

    message = str(error.value)
    for context in (
        "split='val'",
        "global_interval=10:20",
        "local_interval=-1:9",
        "frame_dataset_offset=11",
        "loaded_frame_count=10",
        "group_episode_indices=[2, 3]",
    ):
        assert context in message


def test_materialize_and_validate_all_nonzero_offset_splits(tmp_path) -> None:
    bundle = _make_sharded_dataset(tmp_path)
    cache_root = tmp_path / "cache"

    for split in ("train", "val", "test"):
        dataset = open_bumi_dataset(bundle, split=split)
        global_start, global_end = _split_frame_interval(bundle, split)
        summary = write_episode_cache(
            dataset,
            output_root=cache_root,
            split=split,
            max_frames_per_shard=100,
            device="cpu",
            overwrite=False,
        )

        assert (cache_root / split / "summary.json").is_file()
        assert summary["episodes"] == 2
        assert summary["frames"] == global_end - global_start
        report = inspect_episode_cache(cache_root, split)
        assert report["valid"] is True, report["errors"]

        cached_qpos = np.concatenate(
            [
                np.load(path)
                for path in sorted((cache_root / split / "shards").glob("shard_*/qpos.npy"))
            ],
            axis=0,
        )
        np.testing.assert_allclose(cached_qpos, bundle["states"][global_start:global_end])
