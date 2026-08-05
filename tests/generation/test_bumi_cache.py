import hashlib

from omg.cli.data.materialize_episode_cache import write_episode_cache
from omg.data.episode_cache import EpisodeCachedBumiMotionDataset, EpisodeCachedMotionDataset
from tests.bumi_test_utils import make_bumi_lerobot_dataset, open_bumi_dataset


def test_bumi_cache_v3_materialize_validate_and_read(tmp_path) -> None:
    bundle = make_bumi_lerobot_dataset(tmp_path / "dataset")
    source = open_bumi_dataset(bundle)
    cache_root = tmp_path / "cache"
    summary = write_episode_cache(
        source,
        output_root=cache_root,
        split="train",
        max_frames_per_shard=64,
        device="cpu",
        overwrite=False,
    )
    assert summary["format"] == EpisodeCachedMotionDataset.FORMAT
    assert summary["robot_name"] == "bumi"
    assert summary["state_dim"] == 28
    assert summary["feature_dim"] == 93
    digest = hashlib.sha256(open(source.kinematics.kinematics_path, "rb").read()).hexdigest()
    cached = EpisodeCachedBumiMotionDataset(
        root=cache_root,
        split="train",
        source_repo_id=bundle["repo_id"],
        source_revision=bundle["revision"],
        kinematics_sha256=digest,
        representation_name="bumi_rot6d_93d",
    )
    sample = cached[0]
    assert sample["qpos"].shape == (4, 28)
    assert sample["prev_qpos"].shape == (2, 28)
    assert sample["motion_features"].shape == (4, 93)
    assert "qpos_36" not in sample
