from __future__ import annotations

from pathlib import Path

import numpy as np
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
from omg.realtime.gmt_trajectory import GmtTrajectoryPacket, joint_order_sha256
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


def _runtime(
    controller: DynamicConditionController,
    *,
    planner: _Planner | None = None,
):
    idle = _idle()
    planner = planner or _Planner()
    publisher = _Publisher()
    player = _AudioPlayer()
    runtime = BumiGmtRuntime(
        config=BumiGmtRuntimeConfig(
            replan_remaining_frames=60,
            status_interval_seconds=1000.0,
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
        native_to_gmt=np.arange(21),
        joint_order_hash=joint_order_sha256([f"j{i}" for i in range(21)]),
        audio_player=player,  # type: ignore[arg-type]
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
