from __future__ import annotations

import numpy as np

from omg.realtime.bumi_motion_mux import (
    BumiMotionMuxConfig,
    BumiMotionSourceMux,
    BumiTrackerExecutionHistory,
    resample_bumi_plan,
)


def _idle() -> np.ndarray:
    qpos = np.zeros((28,), dtype=np.float32)
    qpos[2] = 0.48
    qpos[3] = 1.0
    qpos[8] = 0.3
    qpos[12] = -0.3
    return qpos


def test_sixty_frames_at_30hz_resample_to_one_hundred_at_50hz() -> None:
    motion = np.repeat(_idle()[None], 60, axis=0)
    motion[:, 0] = np.arange(60, dtype=np.float32) / 30.0
    result = resample_bumi_plan(motion, source_fps=30.0, target_fps=50.0)
    assert result.shape == (100, 28)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(np.linalg.norm(result[:, 3:7], axis=1), 1.0, atol=1e-6)


def test_fixed_idle_never_underflows_and_waiting_keeps_same_pose() -> None:
    idle = _idle()
    mux = BumiMotionSourceMux(idle)
    for frame in range(200):
        np.testing.assert_array_equal(mux.tick(frame).qpos, idle)
    mux.mark_waiting_for_plan()
    for frame in range(200, 220):
        tick = mux.tick(frame)
        assert tick.source == "waiting_plan"
        np.testing.assert_array_equal(tick.qpos, idle)
    assert mux.status()["underflow_count"] == 0


def test_return_to_idle_preserves_xy_and_heading_and_is_smooth() -> None:
    idle = _idle()
    mux = BumiMotionSourceMux(
        idle,
        BumiMotionMuxConfig(
            tracker_fps=50.0,
            blend_seconds=0.0,
            return_seconds=1.0,
            return_max_joint_speed_radians=1.5,
        ),
    )
    plan = np.repeat(idle[None], 60, axis=0)
    plan[:, :2] = np.array([1.2, -0.7], dtype=np.float32)
    yaw = np.deg2rad(35.0)
    plan[:, 3:7] = np.array(
        [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float32
    )
    plan[:, 7:] += 0.5
    assert mux.accept_plan(plan, source_fps=30.0, command_revision=1)
    current = mux.tick(0).qpos
    mux.return_to_idle()
    outputs = [current]
    while mux.state.value != "FIXED_IDLE":
        outputs.append(mux.tick(len(outputs)).qpos)
    output = np.stack(outputs)
    np.testing.assert_allclose(output[-1, :2], [1.2, -0.7], atol=1e-6)
    np.testing.assert_allclose(output[-1, 3:7], plan[0, 3:7], atol=1e-6)
    max_joint_speed = np.max(np.abs(np.diff(output[:, 7:], axis=0))) * 50.0
    assert max_joint_speed <= 1.5 + 1e-4


def test_planner_history_is_actual_executed_motion_not_repeated_current() -> None:
    initial = np.repeat(_idle()[None], 10, axis=0)
    history = BumiTrackerExecutionHistory(
        initial, tracker_fps=50.0, history_fps=30.0, history_frames=10
    )
    for frame in range(30):
        qpos = _idle()
        qpos[0] = frame / 50.0
        history.append(qpos)
    result = history.planner_history()
    assert result.shape == (10, 28)
    assert np.all(np.diff(result[:, 0]) > 0.0)
    assert np.unique(result[:, 0]).size == 10
