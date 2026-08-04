from __future__ import annotations

import numpy as np
import pytest

from omg.realtime.g1_motion_mux import (
    G1MotionMuxConfig,
    G1MotionSourceMux,
    G1MotionState,
    TrackerExecutionHistory,
    coerce_g1_qpos_frame,
    interpolate_g1_qpos,
    yaw_radians_from_wxyz,
)


def _idle() -> np.ndarray:
    frame = np.zeros(36, dtype=np.float32)
    frame[2] = 0.8
    frame[3] = 1.0
    frame[7:] = np.linspace(-0.2, 0.2, 29, dtype=np.float32)
    return frame


def _plan(frames: int = 8) -> np.ndarray:
    motion = np.repeat(_idle()[None], frames, axis=0)
    motion[:, 0] = np.linspace(1.0, 2.0, frames, dtype=np.float32)
    motion[:, 7] = np.linspace(0.2, 0.8, frames, dtype=np.float32)
    return motion


def test_idle_outputs_the_exact_same_fixed_pose_forever() -> None:
    idle = _idle()
    mux = G1MotionSourceMux(idle)

    ticks = [mux.tick(index) for index in range(20)]

    assert all(tick.state == G1MotionState.IDLE.value for tick in ticks)
    assert all(tick.source == "idle" for tick in ticks)
    np.testing.assert_array_equal(np.stack([tick.qpos_36 for tick in ticks]), idle[None].repeat(20, axis=0))
    assert mux.plan_remaining_frames == 0


def test_waiting_for_first_plan_continues_fixed_idle() -> None:
    mux = G1MotionSourceMux(_idle())
    mux.mark_waiting_for_plan()

    tick = mux.tick(0)

    assert tick.state == G1MotionState.WAITING_PLAN.value
    assert tick.source == "waiting_plan"
    np.testing.assert_array_equal(tick.qpos_36, _idle())


def test_generated_plan_is_resampled_and_consumed() -> None:
    mux = G1MotionSourceMux(
        _idle(),
        G1MotionMuxConfig(tracker_fps=10.0, blend_seconds=0.0, return_seconds=0.2),
    )

    accepted = mux.accept_plan(_plan(4), source_fps=10.0, plan_id=7, command_revision=2)
    outputs = [mux.tick(index) for index in range(4)]

    assert accepted
    assert all(tick.source == "generated" for tick in outputs)
    np.testing.assert_allclose(np.stack([tick.qpos_36 for tick in outputs]), _plan(4), atol=1e-6)
    assert mux.plan_remaining_frames == 0
    assert mux.status()["active_plan_id"] == 7


def test_new_plan_blends_from_last_executed_frame() -> None:
    mux = G1MotionSourceMux(
        _idle(),
        G1MotionMuxConfig(tracker_fps=10.0, blend_seconds=0.2, return_seconds=0.2),
    )
    mux.accept_plan(_plan(8), source_fps=10.0)

    first = mux.tick(0)
    second = mux.tick(1)

    assert first.source == "generated_blend"
    assert 0.0 < float(first.qpos_36[0]) < float(_plan(8)[0, 0])
    assert second.state == G1MotionState.GENERATED.value
    assert np.isclose(np.linalg.norm(first.qpos_36[3:7]), 1.0)


def test_return_to_idle_preserves_xy_and_uses_canonical_pose() -> None:
    mux = G1MotionSourceMux(
        _idle(),
        G1MotionMuxConfig(tracker_fps=10.0, blend_seconds=0.0, return_seconds=0.3),
    )
    mux.accept_plan(_plan(8), source_fps=10.0)
    moving = mux.tick(0).qpos_36

    mux.return_to_idle(reason="stand")
    returned = [mux.tick(index) for index in range(1, 4)][-1]

    expected = _idle()
    expected[:2] = moving[:2]
    assert returned.state == G1MotionState.IDLE.value
    np.testing.assert_allclose(returned.qpos_36, expected, atol=1e-6)


def test_return_to_idle_keeps_heading_and_removes_root_tilt() -> None:
    idle = _idle()
    idle[3:7] = np.asarray(
        [np.cos(np.deg2rad(45.0)), 0.0, 0.0, np.sin(np.deg2rad(45.0))],
        dtype=np.float32,
    )
    mux = G1MotionSourceMux(
        idle,
        G1MotionMuxConfig(
            tracker_fps=10.0,
            blend_seconds=0.0,
            return_seconds=0.2,
            preserve_idle_heading=True,
        ),
    )
    yaw = np.deg2rad(140.0)
    roll = np.deg2rad(20.0)
    moving = idle.copy()
    moving[3:7] = np.asarray(
        [
            np.cos(yaw / 2.0) * np.cos(roll / 2.0),
            np.cos(yaw / 2.0) * np.sin(roll / 2.0),
            np.sin(yaw / 2.0) * np.sin(roll / 2.0),
            np.sin(yaw / 2.0) * np.cos(roll / 2.0),
        ],
        dtype=np.float32,
    )
    mux.accept_plan(moving[None, :], source_fps=10.0)
    mux.tick(0)

    mux.return_to_idle(reason="stand")
    mux.tick(1)
    returned = mux.tick(2).qpos_36

    assert np.isclose(
        yaw_radians_from_wxyz(returned[3:7]),
        yaw_radians_from_wxyz(moving[3:7]),
        atol=1e-5,
    )
    np.testing.assert_allclose(returned[4:6], 0.0, atol=1e-5)


def test_joint_speed_limit_can_extend_smooth_return_duration() -> None:
    mux = G1MotionSourceMux(
        _idle(),
        G1MotionMuxConfig(
            tracker_fps=10.0,
            blend_seconds=0.0,
            return_seconds=0.1,
            return_max_joint_speed_radians=1.0,
        ),
    )
    moving = _idle()
    moving[7] += 2.0
    mux.accept_plan(moving[None, :], source_fps=10.0)
    mux.tick(0)

    mux.return_to_idle(reason="stand")

    assert mux.status()["return_transition_frames"] == 30
    first = mux.tick(1)
    assert first.source == "return_to_idle"
    assert float(_idle()[7]) < float(first.qpos_36[7]) < float(moving[7])
    returned = [moving[7], first.qpos_36[7]]
    returned.extend(mux.tick(index).qpos_36[7] for index in range(2, 31))
    assert float(np.max(np.abs(np.diff(returned)))) <= 0.1 + 1e-5


def test_plan_underflow_returns_safely_instead_of_raising() -> None:
    mux = G1MotionSourceMux(
        _idle(),
        G1MotionMuxConfig(tracker_fps=10.0, blend_seconds=0.0, return_seconds=0.1),
    )
    mux.accept_plan(_plan(1), source_fps=10.0)

    mux.tick(0)
    fallback = mux.tick(1)

    assert fallback.underflow_count == 1
    assert fallback.source == "error_return"
    assert fallback.state == G1MotionState.IDLE.value
    assert np.isfinite(fallback.qpos_36).all()


def test_expired_plan_does_not_replace_current_source() -> None:
    mux = G1MotionSourceMux(_idle(), G1MotionMuxConfig(tracker_fps=10.0))

    accepted = mux.accept_plan(_plan(2), source_fps=10.0, skip_tracker_frames=2)

    assert not accepted
    assert mux.state == G1MotionState.IDLE
    assert mux.status()["expired_plan_count"] == 1


def test_interpolation_handles_antipodal_quaternions() -> None:
    first = _idle()
    second = _idle()
    second[3:7] = -first[3:7]

    halfway = interpolate_g1_qpos(first, second, 0.5)

    np.testing.assert_allclose(halfway[3:7], first[3:7], atol=1e-6)
    assert np.isclose(np.linalg.norm(halfway[3:7]), 1.0)


def test_invalid_frames_are_rejected() -> None:
    with pytest.raises(ValueError, match="36"):
        coerce_g1_qpos_frame(np.zeros(35, dtype=np.float32))
    bad = _idle()
    bad[3:7] = 0.0
    with pytest.raises(ValueError, match="norm"):
        coerce_g1_qpos_frame(bad)


def test_tracker_history_has_requested_planner_length_and_latest_pose() -> None:
    initial = np.repeat(_idle()[None], 3, axis=0)
    history = TrackerExecutionHistory(
        initial,
        tracker_fps=50.0,
        history_fps=30.0,
        history_frames=10,
    )
    for index in range(20):
        frame = _idle()
        frame[0] = float(index)
        history.append(frame)

    planner_history = history.planner_history()

    assert planner_history.shape == (10, 36)
    assert np.isclose(planner_history[-1, 0], 19.0)
    assert np.all(np.diff(planner_history[:, 0]) >= 0.0)
