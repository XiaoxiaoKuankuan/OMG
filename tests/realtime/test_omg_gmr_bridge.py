from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile

from omg.cli.realtime.omg_gmr_bridge import (
    OmgGmrRuntime,
    OmgGmrRuntimeConfig,
    _parse_args,
)
from omg.realtime.dynamic_condition import DynamicConditionController
from omg.realtime.g1_motion_mux import (
    G1MotionMuxConfig,
    G1MotionSourceMux,
    TrackerExecutionHistory,
)
from omg.realtime.protocol import MotionPlanChunk, RobotStateRequest


def _idle() -> np.ndarray:
    frame = np.zeros(36, dtype=np.float32)
    frame[2] = 0.8
    frame[3] = 1.0
    return frame


def _motion(frames: int = 8, *, x_start: float = 1.0) -> np.ndarray:
    motion = np.repeat(_idle()[None], frames, axis=0)
    motion[:, 0] = np.linspace(x_start, x_start + 1.0, frames, dtype=np.float32)
    motion[:, 7] = np.linspace(0.1, 0.5, frames, dtype=np.float32)
    return motion


class _FakePlanner:
    def __init__(self) -> None:
        self.requests: list[RobotStateRequest] = []
        self.pending: RobotStateRequest | None = None
        self.response: MotionPlanChunk | None = None
        self.closed = False

    def begin_request(self, request: RobotStateRequest) -> None:
        assert self.pending is None
        self.pending = request
        self.requests.append(request)

    def complete(self, *, plan_id: int = 0, frames: int = 8, prompt: str = "move") -> None:
        assert self.pending is not None
        request = self.pending
        self.response = MotionPlanChunk(
            qpos_36=_motion(frames, x_start=float(plan_id + 1)),
            fps=10.0,
            request_id=str(request.request_id),
            plan_id=plan_id,
            request_tracker_frame=int(request.tracker_frame),
            prompt=prompt,
        )

    def poll_plan(self, *, timeout_ms: int = 0) -> MotionPlanChunk | None:
        del timeout_ms
        if self.response is None:
            return None
        response = self.response
        self.response = None
        self.pending = None
        return response

    def close(self) -> None:
        self.closed = True


class _FakePublisher:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def publish(self, qpos_36: np.ndarray, *, timestamp: float) -> bool:
        del timestamp
        self.frames.append(np.asarray(qpos_36, dtype=np.float32).copy())
        return True

    def status(self) -> dict[str, Any]:
        return {"published_frames": len(self.frames)}


def _runtime(
    *,
    initial: str = "text: stand still",
    replan_remaining_frames: int = 2,
    request_timeout_ms: int = 120000,
    clock: Any | None = None,
    planner_client_factory: Any | None = None,
) -> tuple[
    OmgGmrRuntime,
    DynamicConditionController,
    _FakePlanner,
    _FakePublisher,
    list[dict[str, Any]],
]:
    controller = DynamicConditionController(
        tracker_fps=10.0,
        initial_condition_sequence=initial,
    )
    planner = _FakePlanner()
    publisher = _FakePublisher()
    events: list[dict[str, Any]] = []
    idle = _idle()
    mux = G1MotionSourceMux(
        idle,
        G1MotionMuxConfig(
            tracker_fps=10.0,
            blend_seconds=0.0,
            return_seconds=0.2,
        ),
    )
    history = TrackerExecutionHistory(
        np.repeat(idle[None], 10, axis=0),
        tracker_fps=10.0,
        history_fps=10.0,
        history_frames=4,
    )
    runtime = OmgGmrRuntime(
        config=OmgGmrRuntimeConfig(
            tracker_fps=10.0,
            history_fps=10.0,
            history_frames=4,
            planner_frames=8,
            replan_remaining_frames=replan_remaining_frames,
            audio_fps=10.0,
            condition_audio_step_frames=6,
            request_timeout_ms=request_timeout_ms,
        ),
        controller=controller,
        planner_client=planner,
        mux=mux,
        history=history,
        publisher=publisher,
        status_callback=events.append,
        planner_client_factory=planner_client_factory,
        **({"clock": clock} if clock is not None else {}),
    )
    return runtime, controller, planner, publisher, events


def test_cli_defaults_to_neutral_idle_preset(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["omg_gmr_bridge", "--seed-motion", "inputs/seed_motion.npz"],
    )

    args = _parse_args()

    assert args.idle_preset == "neutral"
    assert args.idle_motion is None
    assert args.initial_yaw_degrees == 90.0
    assert args.return_seconds == 1.0
    assert args.return_max_joint_speed == 1.5
    assert not args.reset_idle_heading


def test_default_stand_publishes_fixed_pose_without_planner_requests() -> None:
    runtime, _controller, planner, publisher, _events = _runtime()

    for _ in range(12):
        runtime.step()

    assert planner.requests == []
    assert len(publisher.frames) == 12
    np.testing.assert_array_equal(
        np.stack(publisher.frames),
        np.repeat(_idle()[None], 12, axis=0),
    )
    assert runtime.status()["command_type"] == "stand"


def test_text_command_requests_plan_and_switches_to_generated_output() -> None:
    runtime, controller, planner, publisher, events = _runtime()
    command = controller.accept_text("walk forward")

    waiting = runtime.step()
    assert waiting.source == "waiting_plan"
    assert len(planner.requests) == 1
    request = planner.requests[0]
    assert request.metadata["condition_sequence"] == "text: walk forward"
    assert request.metadata["command_revision"] == command.revision
    assert request.qpos_36_history.shape == (4, 36)

    planner.complete(plan_id=3, prompt="walk forward")
    generated = runtime.step()

    assert generated.source == "generated"
    assert generated.qpos_36[0] > 0.0
    assert runtime.active_plan_revision == command.revision
    assert any(event["kind"] == "replan_completed" and event["accepted"] for event in events)
    assert len(publisher.frames) == 2


def test_stand_during_pending_request_discards_response_and_never_plans_stand() -> None:
    runtime, controller, planner, publisher, events = _runtime()
    controller.accept_text("turn left")
    runtime.step()
    assert len(planner.requests) == 1
    planner.complete(plan_id=1, prompt="turn left")

    stand = controller.accept_stand()
    tick = runtime.step()

    assert controller.snapshot().command_type == "stand"
    assert len(planner.requests) == 1
    assert tick.source == "return_to_idle"
    assert any(
        event["kind"] == "replan_completed"
        and event["stale_command"]
        and not event["accepted"]
        for event in events
    )
    runtime.step()
    np.testing.assert_allclose(publisher.frames[-1], _idle(), atol=1e-6)
    assert runtime.status()["command_revision"] == stand.revision


def test_new_text_during_pending_request_is_planned_immediately_after_old_reply() -> None:
    runtime, controller, planner, _publisher, events = _runtime()
    first = controller.accept_text("walk forward")
    runtime.step()
    planner.complete(plan_id=1, prompt="walk forward")
    second = controller.accept_text("wave both arms")

    runtime.step()

    assert len(planner.requests) == 2
    latest = planner.requests[-1]
    assert latest.metadata["condition_sequence"] == "text: wave both arms"
    assert latest.metadata["command_revision"] == second.revision
    assert latest.metadata["condition_session_id"] != planner.requests[0].metadata["condition_session_id"]
    assert first.revision != second.revision
    assert any(event.get("stale_command") for event in events if event["kind"] == "replan_completed")


def test_audio_end_switches_to_fixed_idle_without_a_stand_plan(tmp_path: Path) -> None:
    wav = tmp_path / "short.wav"
    wavfile.write(wav, 10, np.zeros(2, dtype=np.int16))  # 0.2 seconds at 10 Hz.
    runtime, controller, planner, _publisher, events = _runtime()
    audio = controller.accept_audio(str(wav))

    runtime.step()  # Audio starts at tracker frame 0 and creates one request.
    assert len(planner.requests) == 1
    assert controller.snapshot().audio_start_tracker_frame == 0
    assert controller.snapshot().audio_end_tracker_frame == 2
    runtime.step()
    runtime.step()  # frame 2 reaches audio end before output selection.

    assert controller.snapshot().command_type == "stand"
    assert controller.snapshot().revision == audio.revision + 1
    assert len(planner.requests) == 1
    assert runtime.mux.state.value in {"RETURNING", "IDLE"}
    assert any(event["kind"] == "audio_end" for event in events)


def test_plan_underflow_keeps_publishing_and_does_not_raise() -> None:
    runtime, controller, planner, publisher, _events = _runtime(
        replan_remaining_frames=0
    )
    controller.accept_text("move")
    runtime.step()
    planner.complete(plan_id=0, frames=2)
    runtime.step()

    fallback = runtime.step()

    assert fallback.underflow_count == 1
    assert fallback.source == "error_return"
    assert len(publisher.frames) == 3
    assert np.isfinite(np.stack(publisher.frames)).all()


def test_close_releases_planner_client() -> None:
    runtime, _controller, planner, _publisher, _events = _runtime()

    runtime.close()

    assert planner.closed


def test_planner_timeout_recreates_transport_without_stopping_output() -> None:
    now = [0.0]
    replacements: list[_FakePlanner] = []

    def factory() -> _FakePlanner:
        client = _FakePlanner()
        replacements.append(client)
        return client

    runtime, controller, first, publisher, events = _runtime(
        request_timeout_ms=100,
        clock=lambda: now[0],
        planner_client_factory=factory,
    )
    controller.accept_text("walk")
    runtime.step()
    assert first.pending is not None

    now[0] = 0.2
    tick = runtime.step()

    assert first.closed
    assert len(replacements) == 1
    assert runtime.planner_client is replacements[0]
    assert len(replacements[0].requests) == 1
    assert len(publisher.frames) == 2
    assert tick.source == "waiting_plan"
    assert any(event["kind"] == "replan_timeout" for event in events)
