from pathlib import Path

from tools.validate_mjcf_kinematics import validate


def test_bumi_generic_fk_matches_mujoco_for_1000_qpos() -> None:
    report = validate(
        Path("assets/robots/bumi/bumi3.xml").resolve(),
        Path("assets/robots/bumi/bumi_kinematics.json").resolve(),
        samples=1000,
        seed=2026,
    )
    assert report["max_position_error_m"] < 1.0e-5
    assert report["max_rotation_matrix_error"] < 1.0e-4
    assert report["valid"] is True
