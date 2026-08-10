from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from omg.pipeline.reference import load_motion_reference
from omg.tracking.holomotion.reference import resample_qpos


def _bumi_qpos(frames: int) -> np.ndarray:
    qpos = np.zeros((frames, 28), dtype=np.float32)
    qpos[:, 0] = np.arange(frames, dtype=np.float32)
    qpos[:, 2] = 0.55
    qpos[:, 3] = 2.0
    return qpos


def test_load_bumi_reference_and_resample(tmp_path: Path) -> None:
    path = tmp_path / "bumi.npz"
    np.savez_compressed(
        path,
        qpos=_bumi_qpos(4),
        fps=np.asarray([30.0], dtype=np.float32),
        robot_name=np.asarray(["bumi"]),
        state_dim=np.asarray([28], dtype=np.int32),
    )
    reference = load_motion_reference(path, expected_state_dim=28)
    assert reference.qpos.shape == (4, 28)
    np.testing.assert_allclose(np.linalg.norm(reference.qpos[:, 3:7], axis=1), 1.0, atol=1e-6)

    resampled = resample_qpos(reference.qpos, source_fps=30.0, target_fps=50.0)
    assert resampled.shape[1] == 28
    assert np.isfinite(resampled).all()
    np.testing.assert_allclose(np.linalg.norm(resampled[:, 3:7], axis=1), 1.0, atol=1e-6)


def test_load_bumi_reference_rejects_state_dim_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "bumi.npy"
    np.save(path, _bumi_qpos(4))
    with pytest.raises(ValueError, match="T,36"):
        load_motion_reference(path, expected_state_dim=36, fps=30.0)

