import argparse
from pathlib import Path

from omg.cli.generation import compute_stats as module
from tests.bumi_test_utils import make_bumi_lerobot_dataset, open_bumi_dataset


def test_bumi_stats_have_93_features_and_28d_defaults(tmp_path, monkeypatch) -> None:
    bundle = make_bumi_lerobot_dataset(tmp_path / "dataset")
    dataset = open_bumi_dataset(bundle, omnimodal=False)
    configs = {
        "data.yaml": {"dataset_opts": {"train": {"bumi": {"_target_": "unused"}}}},
        "representation.yaml": {
            "feat_dim": 93,
            "robot_name": "bumi",
            "state_dim": 28,
            "rotation_representation": "rot6d",
            "num_prev_states": 2,
            "canonical_frame_idx": 1,
            "sequence_length": 4,
        },
        "paths.yaml": {},
    }
    monkeypatch.setattr(module, "_load_yaml", lambda path: configs[Path(path).name])
    monkeypatch.setattr(module, "instantiate", lambda cfg: dataset)
    args = argparse.Namespace(
        data_config=Path("data.yaml"),
        representation_config=Path("representation.yaml"),
        paths_config=Path("paths.yaml"),
        split="train",
        batch_size=2,
        max_samples=None,
        device="cpu",
        episode_batch_frames=64,
        rank=0,
        world_size=1,
        std_min=1e-6,
    )
    stats = module.compute_stats(args)
    assert stats["robot_name"] == "bumi"
    assert stats["state_dim"] == 28
    assert stats["feature_dim"] == 93
    assert len(stats["mean"]) == 93
    assert len(stats["std"]) == 93
    assert len(stats["default_joint_dof"]) == 21
    assert stats["quaternion_convention"] == "wxyz"
