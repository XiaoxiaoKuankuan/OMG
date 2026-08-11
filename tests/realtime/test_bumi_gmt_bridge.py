from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from omg.cli.realtime.omg_bumi_gmt_bridge import (
    BumiGmtRuntime,
    BumiGmtRuntimeConfig,
)
from omg.realtime.bumi_motion_mux import (
    BumiMotionMuxConfig,
    BumiMotionSourceMux,
    BumiTrackerExecutionHistory,
    BumiTrajectoryHistory,
)
from omg.realtime.dynamic_condition import DynamicConditionController
from omg.realtime.gmt_trajectory import (
    GmtLowStateFeedback,
    GmtTrajectoryAck,
    GmtTrajectoryPacket,
    joint_order_sha256,
)
from omg.realtime.protocol import MotionPlanChunk


def _idle() -> np.ndarray:
    value = np.zeros((28,), dtype=np.float32)
    value[2] = 0.48
    value[3] = 1.0
    return value


class _Planner:
    def __init__(self) -> None:
        self.requests = []
        self.responses = []
        self.closed = False

    def begin_request(self, request) -> None:
        self.requests.append(request)

    def poll_plan(self, timeout_ms=0):
        del timeout_ms
        return self.responses.pop(0) if self.responses else None

    def close(self) -> None:
        self.closed = True


class _Publisher:
    def __init__(self) -> None:
        self.packets = []

    def publish(self, packet) -> bool:
        self.packets.append(packet)
        return True

    def status(self):
        return {"published_packets": len(self.packets)}


class _AudioPlayer:
    def __init__(self) -> None:
        self.enabled = True
        self.command_id = None
        self.starts = []
        self.stops = []

    def start(self, path, *, command_id, tracker_frame):
        self.command_id = command_id
        self.starts.append((str(path), command_id, tracker_frame))
        return True

    def stop(self, *, reason):
        self.stops.append(reason)
        self.command_id = None
        return True

    def status(self):
        return {"enabled": True, "playing": self.command_id is not None}

    def close(self):
        self.stop(reason="closed")


class _AckReader:
    def __init__(self) -> None:
        self.ack = None

    def latest_ack(self):
        return self.ack

    def status(self):
        return {
            "latest_sequence": None if self.ack is None else self.ack.sequence,
        }


class _LowStateReader:
    def __init__(self, order_hash: bytes) -> None:
        self.order_hash = order_hash
        self.available = True
        self.sequence = 0
        self.samples: list[GmtLowStateFeedback] = []

    def latest_feedback(self, *, max_age_seconds=None):
        del max_age_seconds
        if not self.available:
            return None
        self.sequence += 1
        sample = GmtLowStateFeedback(
            sequence=self.sequence,
            captured_unix_ns=self.sequence * 1_000_000,
            joint_order_hash=self.order_hash,
            root_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            joint_pos=(
                np.arange(21, dtype=np.float32) + np.float32(self.sequence)
            ),
        )
        self.samples.append(sample)
        return sample

    def status(self):
        return {"latest_sequence": self.sequence, "available": self.available}


def _runtime(
    controller: DynamicConditionController,
    *,
    planner: _Planner | None = None,
    ack_reader=None,
    history_source: str = "reference",
    lowstate_reader=None,
    native_to_gmt: np.ndarray | None = None,
):
    idle = _idle()
    planner = planner or _Planner()
    publisher = _Publisher()
    player = _AudioPlayer()
    order_hash = joint_order_sha256([f"j{i}" for i in range(21)])
    runtime = BumiGmtRuntime(
        config=BumiGmtRuntimeConfig(
            replan_remaining_frames=60,
            status_interval_seconds=1000.0,
            history_source=history_source,
        ),
        controller=controller,
        planner_client=planner,
        mux=BumiMotionSourceMux(
            idle,
            BumiMotionMuxConfig(
                tracker_fps=50.0, blend_seconds=0.0, return_seconds=1.0
            ),
        ),
        planner_history=BumiTrackerExecutionHistory(
            np.repeat(idle[None], 10, axis=0),
            tracker_fps=50.0,
            history_fps=30.0,
            history_frames=10,
        ),
        trajectory_history=BumiTrajectoryHistory(idle),
        publisher=publisher,
        native_to_gmt=(
            np.arange(21) if native_to_gmt is None else native_to_gmt
        ),
        joint_order_hash=order_hash,
        audio_player=player,  # type: ignore[arg-type]
        ack_reader=ack_reader,
        lowstate_history=(
            BumiTrackerExecutionHistory(
                np.repeat(idle[None], 10, axis=0),
                tracker_fps=50.0,
                history_fps=30.0,
                history_frames=10,
            )
            if history_source == "lowstate"
            else None
        ),
        lowstate_reader=lowstate_reader,
        clock=lambda: runtime.cursor / 50.0 if "runtime" in locals() else 0.0,
        stream_id=10,
    )
    return runtime, planner, publisher, player


class _FailingPlanner(_Planner):
    def __init__(self) -> None:
        super().__init__()
        self.request_attempts = 0

    def begin_request(self, request) -> None:
        self.request_attempts += 1
        raise RuntimeError("planner unavailable")


def _response(request, *, x_end: float = 1.0) -> MotionPlanChunk:
    plan = np.repeat(_idle()[None], 60, axis=0)
    plan[:, 0] = np.linspace(0.0, x_end, 60, dtype=np.float32)
    return MotionPlanChunk(
        qpos_36=plan,
        fps=30.0,
        request_id=request.request_id,
        plan_id=len(request.metadata.get("test_plan_ids", [])),
        request_tracker_frame=request.tracker_frame,
        metadata={"robot_name": "bumi", "state_dim": 28},
    )


def test_startup_and_stand_publish_fixed_pose_without_planner_requests() -> None:
    controller = DynamicConditionController(tracker_fps=50.0)
    runtime, planner, publisher, _player = _runtime(controller)
    for _ in range(20):
        tick = runtime.step()
        assert tick.source == "fixed_idle"
    assert planner.requests == []
    controller.accept_stand()
    for _ in range(5):
        runtime.step()
    assert planner.requests == []
    assert len(publisher.packets) == 25
    decoded = GmtTrajectoryPacket.decode(publisher.packets[-1].encode())
    assert decoded.frames.shape == (110, 55)
    runtime.close()


def test_text_requests_plan_and_pending_stand_never_requests_stand_text() -> None:
    controller = DynamicConditionController(tracker_fps=50.0)
    runtime, planner, _publisher, _player = _runtime(controller)
    controller.accept_text("walk forward")
    runtime.step()
    assert len(planner.requests) == 1
    assert planner.requests[0].metadata["condition_sequence"] == "text: walk forward"
    controller.accept_stand()
    for _ in range(4):
        runtime.step()
    assert len(planner.requests) == 1
    assert all(
        request.metadata["condition_sequence"] != "text: stand still"
        for request in planner.requests
    )
    runtime.close()


def test_lowstate_history_waits_for_ten_real_samples_and_fuses_reference_xyz() -> None:
    controller = DynamicConditionController(tracker_fps=50.0)
    order_hash = joint_order_sha256([f"j{i}" for i in range(21)])
    reader = _LowStateReader(order_hash)
    permutation = np.arange(20, -1, -1, dtype=np.int64)
    runtime, planner, _publisher, _player = _runtime(
        controller,
        history_source="lowstate",
        lowstate_reader=reader,
        native_to_gmt=permutation,
    )
    controller.accept_text("walk forward")

    # 10 frames at 30 Hz span 16 tracker samples at 50 Hz.  Until that
    # measured interval exists the bridge stays in fixed waiting pose.
    for _ in range(16):
        runtime.step()
    assert planner.requests == []
    runtime.step()
    assert len(planner.requests) == 1

    request = planner.requests[0]
    history = request.qpos_36_history
    assert history.shape == (10, 28)
    assert request.metadata["history_source"] == "lowstate"
    assert request.metadata["history_root_xyz_source"] == "reference"
    np.testing.assert_allclose(
        history[:, :3], np.repeat(_idle()[None, :3], 10, axis=0), atol=1e-6
    )
    expected_native = np.empty((21,), dtype=np.float32)
    expected_native[permutation] = reader.samples[-2].joint_pos
    np.testing.assert_allclose(history[-1, 7:], expected_native)
    assert runtime.status()["planner_history"]["ready"] is True

    # Losing fresh LowState never silently falls back to reference history.
    planner.responses.append(_response(request))
    runtime.step()
    reader.available = False
    runtime.step()
    controller.accept_text("turn left")
    runtime.step()
    assert len(planner.requests) == 1
    assert runtime.status()["planner_history"]["ready"] is False
    runtime.close()


def test_audio_clock_and_playback_start_on_first_generated_tick_then_end_idle(
    tmp_path: Path,
) -> None:
    path = tmp_path / "short.wav"
    wavfile.write(path, 100, np.zeros((4,), dtype=np.int16))
    controller = DynamicConditionController(tracker_fps=50.0)
    runtime, planner, _publisher, player = _runtime(controller)
    controller.accept_audio(str(path))
    runtime.step()
    assert len(planner.requests) == 1
    assert controller.snapshot().audio_start_tracker_frame is None
    assert player.starts == []
    planner.responses.append(_response(planner.requests[0]))
    first_motion = runtime.step()
    assert first_motion.source == "generated"
    start_frame = controller.snapshot().audio_start_tracker_frame
    assert start_frame == first_motion.frame_index
    assert player.starts[0][2] == first_motion.frame_index
    planner_count = len(planner.requests)
    runtime.step()
    runtime.step()
    assert controller.snapshot().command_type == "stand"
    assert len(planner.requests) == planner_count
    assert all(
        request.metadata["condition_sequence"] != "text: stand still"
        for request in planner.requests
    )
    assert "audio_ended" in player.stops
    runtime.close()


def test_effective_audio_end_discards_generated_tail_and_starts_smooth_idle_return(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trim-tail.wav"
    waveform = np.concatenate(
        (
            np.full((4,), 12000, dtype=np.int16),
            np.zeros((8,), dtype=np.int16),
        )
    )
    wavfile.write(path, 100, waveform)
    controller = DynamicConditionController(
        tracker_fps=50.0,
        audio_tail_silence_min_seconds=0.05,
        audio_tail_analysis_window_seconds=0.01,
    )
    runtime, planner, _publisher, player = _runtime(controller)
    controller.accept_audio(str(path))
    runtime.step()
    planner.responses.append(_response(planner.requests[0]))

    first_motion = runtime.step()
    assert first_motion.source == "generated"
    snapshot = controller.snapshot()
    assert snapshot.audio_source_duration_seconds == pytest.approx(0.12)
    assert snapshot.audio_effective_duration_seconds == pytest.approx(0.04)
    assert snapshot.audio_trailing_silence_seconds == pytest.approx(0.08)
    assert snapshot.audio_end_tracker_frame == first_motion.frame_index + 2

    runtime.step()
    first_return = runtime.step()
    assert controller.snapshot().command_type == "stand"
    assert first_return.source == "return_to_idle"
    assert first_return.plan_remaining_frames == 0
    assert "audio_ended" in player.stops
    runtime.close()


def test_audio_packet_is_published_before_matching_gmt_ack_starts_playback(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ack.wav"
    wavfile.write(path, 100, np.zeros((100,), dtype=np.int16))
    controller = DynamicConditionController(tracker_fps=50.0)
    ack_reader = _AckReader()
    runtime, planner, publisher, player = _runtime(
        controller, ack_reader=ack_reader
    )
    controller.accept_audio(str(path))
    runtime.step()
    planner.responses.append(_response(planner.requests[0]))

    first_motion = runtime.step()
    assert first_motion.source == "generated"
    assert len(publisher.packets) == 2
    first_audio_packet = publisher.packets[-1]
    assert player.starts == []
    assert controller.snapshot().audio_start_tracker_frame is None
    assert runtime.status()["audio_ack_pending"] is True

    ack_reader.ack = GmtTrajectoryAck(
        stream_id=999,
        sequence=first_audio_packet.sequence,
        command_revision=controller.current_revision,
        plan_id=first_audio_packet.plan_id,
        received_unix_ns=1,
    )
    runtime.step()
    assert player.starts == []

    ack_reader.ack = GmtTrajectoryAck(
        stream_id=10,
        sequence=first_audio_packet.sequence,
        command_revision=controller.current_revision,
        plan_id=first_audio_packet.plan_id,
        received_unix_ns=2,
    )
    ack_tick = runtime.cursor
    runtime.step()
    assert player.starts[0][2] == ack_tick
    assert controller.snapshot().audio_start_tracker_frame == ack_tick
    assert runtime.status()["audio_ack_pending"] is False
    runtime.close()


def test_planner_failure_latches_error_idle_until_a_new_command() -> None:
    controller = DynamicConditionController(tracker_fps=50.0)
    failing = _FailingPlanner()
    runtime, planner, _publisher, _player = _runtime(
        controller, planner=failing
    )

    controller.accept_text("walk forward")
    for _ in range(10):
        runtime.step()
    assert planner.request_attempts == 1
    assert runtime.status()["failed_command_revision"] == controller.current_revision

    controller.accept_text("turn left")
    runtime.step()
    assert planner.request_attempts == 2

    controller.accept_stand()
    for _ in range(5):
        runtime.step()
    assert planner.request_attempts == 2
    runtime.close()
