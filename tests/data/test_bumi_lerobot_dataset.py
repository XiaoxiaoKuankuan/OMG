import hashlib

import pytest

from tests.bumi_test_utils import make_bumi_lerobot_dataset, open_bumi_dataset


def test_bumi_lerobot_dataset_loads_native_qpos_and_modalities(tmp_path) -> None:
    bundle = make_bumi_lerobot_dataset(tmp_path / "dataset")
    dataset = open_bumi_dataset(bundle)
    sample = dataset[0]
    assert sample["qpos"].shape == (4, 28)
    assert sample["prev_qpos"].shape == (2, 28)
    assert sample["motion_features"].shape == (4, 93)
    assert sample["audio_features"].shape == (4, 35)
    assert sample["human_motion"].shape == (4, 66)
    assert sample["mask"]["has_audio"].all()
    assert sample["mask"]["has_human_motion"].all()
    assert "qpos_36" not in sample

    from datasets import Dataset

    roundtrip = Dataset.from_parquet(str(bundle["root"] / "data/chunk-000/file-000.parquet"))
    assert len(roundtrip) == 18
    assert len(roundtrip[0]["observation.state"]) == 28


def test_bumi_manifest_digest_is_enforced(tmp_path) -> None:
    bundle = make_bumi_lerobot_dataset(tmp_path / "dataset")
    bundle["manifest_sha256"] = hashlib.sha256(b"wrong").hexdigest()
    with pytest.raises(ValueError, match="manifest identity mismatch"):
        open_bumi_dataset(bundle)
