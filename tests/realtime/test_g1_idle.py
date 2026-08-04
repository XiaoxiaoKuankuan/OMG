from __future__ import annotations

import math

import numpy as np
import pytest

from omg.realtime.g1_idle import (
    G1_JOINT_QPOS_INDEX,
    G1_MODEL_ROOT_HEIGHT_METERS,
    neutral_g1_idle_qpos,
)


def test_neutral_idle_has_zero_legs_and_waist() -> None:
    qpos = neutral_g1_idle_qpos()

    assert qpos.shape == (36,)
    assert qpos.dtype == np.float32
    assert np.isfinite(qpos).all()
    np.testing.assert_allclose(qpos[:3], [0.0, 0.0, G1_MODEL_ROOT_HEIGHT_METERS])
    np.testing.assert_array_equal(qpos[3:7], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(qpos[7:22], np.zeros(15, dtype=np.float32))


def test_neutral_idle_opens_shoulders_and_hangs_forearms() -> None:
    qpos = neutral_g1_idle_qpos()

    assert qpos[G1_JOINT_QPOS_INDEX["left_shoulder_roll_joint"]] == pytest.approx(
        0.12
    )
    assert qpos[G1_JOINT_QPOS_INDEX["right_shoulder_roll_joint"]] == pytest.approx(
        -0.12
    )
    assert qpos[G1_JOINT_QPOS_INDEX["left_elbow_joint"]] == pytest.approx(
        math.pi / 2.0
    )
    assert qpos[G1_JOINT_QPOS_INDEX["right_elbow_joint"]] == pytest.approx(
        math.pi / 2.0
    )


def test_neutral_idle_rejects_unsafe_opening_values() -> None:
    with pytest.raises(ValueError, match=r"\[0, 0.5\]"):
        neutral_g1_idle_qpos(shoulder_open_radians=-0.01)
    with pytest.raises(ValueError, match="finite"):
        neutral_g1_idle_qpos(shoulder_open_radians=float("nan"))


def test_neutral_idle_accepts_counter_clockwise_initial_yaw() -> None:
    qpos = neutral_g1_idle_qpos(yaw_degrees=90.0)

    np.testing.assert_allclose(
        qpos[3:7],
        np.asarray([2.0**-0.5, 0.0, 0.0, 2.0**-0.5], dtype=np.float32),
        atol=1e-6,
    )


def test_neutral_idle_rejects_non_finite_initial_yaw() -> None:
    with pytest.raises(ValueError, match="yaw_degrees"):
        neutral_g1_idle_qpos(yaw_degrees=float("nan"))
