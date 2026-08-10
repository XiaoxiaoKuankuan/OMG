from __future__ import annotations

import numpy as np
import pytest

from omg.realtime.protocol import MotionPlanChunk, RobotStateRequest, decode_message


def _qpos(frames: int) -> np.ndarray:
    out = np.zeros((frames, 36), dtype=np.float32)
    out[:, 2] = 0.75
    out[:, 3] = 1.0
    return out


def test_robot_state_request_round_trip() -> None:
    request = RobotStateRequest(
        qpos_36_history=_qpos(10),
        history_fps=30.0,
        tracker_frame=123,
        buffer_remaining_frames=40,
        prompt="walk forward",
        metadata={"source": "unit-test"},
    )
    header, arrays = decode_message(*request.to_message())
    restored = RobotStateRequest.from_message(header, arrays)

    assert restored.request_id == request.request_id
    assert restored.tracker_frame == 123
    assert restored.buffer_remaining_frames == 40
    assert restored.prompt == "walk forward"
    np.testing.assert_allclose(restored.qpos_36_history, request.qpos_36_history)


def test_motion_plan_chunk_round_trip_with_features() -> None:
    plan = MotionPlanChunk(
        qpos_36=_qpos(60),
        motion_features=np.ones((60, 8), dtype=np.float32),
        fps=30.0,
        request_id="req-1",
        plan_id=2,
        request_tracker_frame=50,
        planning_latency_seconds=0.07,
        prompt="walk forward",
        metadata={"timing_ms": {"total_ms": 70.0}},
    )
    header, arrays = decode_message(*plan.to_message())
    restored = MotionPlanChunk.from_message(header, arrays)

    assert restored.request_id == "req-1"
    assert restored.plan_id == 2
    assert restored.request_tracker_frame == 50
    assert restored.planning_latency_seconds == pytest.approx(0.07)
    np.testing.assert_allclose(restored.qpos_36, plan.qpos_36)
    np.testing.assert_allclose(restored.motion_features, plan.motion_features)


def test_rejects_invalid_qpos_shape() -> None:
    with pytest.raises(ValueError, match="qpos_36_history"):
        RobotStateRequest(qpos_36_history=np.zeros((10, 35), dtype=np.float32), history_fps=30.0, tracker_frame=0)


def test_bumi_request_and_plan_round_trip_use_generic_wire_keys() -> None:
    qpos = np.zeros((10, 28), dtype=np.float32)
    qpos[:, 2] = 0.55
    qpos[:, 3] = 1.0
    identity = {"robot_name": "bumi", "state_dim": 28}
    request = RobotStateRequest(
        qpos_36_history=qpos,
        history_fps=30.0,
        tracker_frame=0,
        metadata=identity,
    )
    header_bytes, payload_bytes = request.to_message()
    header, arrays = decode_message(header_bytes, payload_bytes)
    assert "qpos_history" in arrays
    assert "qpos_36_history" not in arrays
    restored_request = RobotStateRequest.from_message(header, arrays)
    assert restored_request.robot_name == "bumi"
    assert restored_request.state_dim == 28
    np.testing.assert_allclose(restored_request.qpos_history, qpos)

    plan = MotionPlanChunk(
        qpos_36=np.repeat(qpos[:1], 60, axis=0),
        motion_features=np.zeros((60, 93), dtype=np.float32),
        fps=30.0,
        request_id=request.request_id,
        plan_id=0,
        request_tracker_frame=0,
        metadata=identity,
    )
    header, arrays = decode_message(*plan.to_message())
    assert "qpos" in arrays
    assert "qpos_36" not in arrays
    restored_plan = MotionPlanChunk.from_message(header, arrays)
    assert restored_plan.qpos.shape == (60, 28)
    assert restored_plan.robot_name == "bumi"


def test_protocol_normalizes_legacy_g1_robot_name() -> None:
    qpos = np.zeros((10, 36), dtype=np.float32)
    qpos[:, 3] = 1.0
    request = RobotStateRequest(
        qpos_36_history=qpos,
        history_fps=30.0,
        tracker_frame=0,
        metadata={"robot_name": "g1_29dof", "state_dim": 36},
    )
    assert request.robot_name == "g1"
