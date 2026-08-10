from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from omg.motion.representation import BumiMotionRepresentation
from omg.pipeline.planner import (
    MotionPlan,
    _build_runtime_representation,
    _canonical_robot_name,
    _coerce_seed_qpos,
    save_motion_plan,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bumi_metadata() -> dict:
    stats = REPO_ROOT / "assets/stats/bumi_93d_stats.json"
    kinematics = REPO_ROOT / "assets/robots/bumi/bumi_kinematics.json"
    spec = json.loads(kinematics.read_text(encoding="utf-8"))
    return {
        "robot_name": "bumi",
        "state_dim": 28,
        "joint_names": spec["joint_order"],
        "representation_name": "bumi_rot6d_93d",
        "quaternion_convention": "wxyz",
        "stats_path": str(stats),
        "stats_sha256": _sha256(stats),
        "kinematics_path": str(kinematics),
        "kinematics_sha256": _sha256(kinematics),
        "num_prev_states": 10,
        "canonical_frame_idx": 9,
        "feat_dim": 93,
        "sequence_length": 60,
        "rotation_representation": "rot6d",
        "rot6d_gradient_mode": "vanilla",
    }


def test_bumi_runtime_representation_from_onnx_metadata() -> None:
    representation = _build_runtime_representation(_bumi_metadata(), device=torch.device("cpu"))
    assert isinstance(representation, BumiMotionRepresentation)
    assert representation.robot_name == "bumi"
    assert representation.state_dim == 28
    assert representation.feat_dim == 93
    assert representation.num_prev_states == 10

    qpos = representation.get_default_prev_qpos(1, torch.device("cpu"), torch.float32)
    assert qpos.shape == (1, 10, 28)
    assert torch.isfinite(qpos).all()
    torch.testing.assert_close(qpos[:, 1:], qpos[:, :-1])


def test_runtime_robot_name_normalizes_legacy_g1_alias() -> None:
    assert _canonical_robot_name("g1") == "g1"
    assert _canonical_robot_name("g1_29dof") == "g1"
    assert _canonical_robot_name("BUMI") == "bumi"


def test_bumi_runtime_rejects_wrong_asset_hash() -> None:
    metadata = _bumi_metadata()
    metadata["stats_sha256"] = "0" * 64
    try:
        _build_runtime_representation(metadata, device=torch.device("cpu"))
    except ValueError as exc:
        assert "SHA256 mismatch" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected the mismatched stats hash to be rejected")


def test_bumi_runtime_asset_override_keeps_hash_identity(tmp_path: Path) -> None:
    metadata = _bumi_metadata()
    original_stats = Path(metadata["stats_path"])
    copied_stats = tmp_path / "relocated_stats.json"
    copied_stats.write_bytes(original_stats.read_bytes())
    metadata["stats_path"] = str(tmp_path / "missing_original_location.json")

    representation = _build_runtime_representation(
        metadata,
        device=torch.device("cpu"),
        stats_path_override=copied_stats,
    )
    assert Path(representation.stats_path) == copied_stats

    copied_stats.write_text("{}\n", encoding="utf-8")
    try:
        _build_runtime_representation(
            metadata,
            device=torch.device("cpu"),
            stats_path_override=copied_stats,
        )
    except ValueError as exc:
        assert "SHA256 mismatch" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected a modified relocated stats file to be rejected")


def test_bumi_seed_and_motion_plan_save_use_generic_qpos(tmp_path: Path) -> None:
    qpos = np.zeros((10, 28), dtype=np.float32)
    qpos[:, 2] = 0.55
    qpos[:, 3] = 1.0
    coerced = _coerce_seed_qpos(qpos, state_dim=28)
    assert coerced.shape == (10, 28)

    plan = MotionPlan(
        qpos_36=qpos[:3],
        motion_features=np.zeros((3, 93), dtype=np.float32),
        fps=30.0,
        metadata={"robot_name": "bumi", "state_dim": 28, "joint_names": []},
    )
    assert plan.qpos.shape == (3, 28)
    out = save_motion_plan(plan, tmp_path / "plan")
    assert (out / "qpos.npy").exists()
    assert not (out / "qpos_36.npy").exists()
    with np.load(out / "reference_motion.npz") as payload:
        assert payload["qpos"].shape == (3, 28)
        assert "qpos_36" not in payload.files
