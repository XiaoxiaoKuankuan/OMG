from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from omg.realtime.command_server import CommandServerConfig, DynamicCommandServer
from omg.realtime.dynamic_condition import (
    ConditionSnapshot,
    DynamicConditionController,
    compute_condition_audio_step_frames,
)
from omg.realtime.g1_idle import neutral_g1_idle_qpos
from omg.realtime.g1_motion_mux import (
    G1MotionMuxConfig,
    G1MotionSourceMux,
    G1MotionTick,
    TrackerExecutionHistory,
    coerce_g1_qpos_motion,
)
from omg.realtime.protocol import MotionPlanChunk, RobotStateRequest
from omg.realtime.redis_qpos import (
    AsyncRedisQposPublisher,
    RedisQposPublisher,
    RedisQposPublisherConfig,
)
from omg.realtime.status_log import append_jsonl
from omg.realtime.transport import ZmqPlanClient


def load_qpos_motion(
    path: str | Path,
    *,
    fps: float | None,
) -> tuple[np.ndarray, float]:
    motion_path = Path(path).expanduser()
    if not motion_path.exists():
        raise FileNotFoundError(f"G1 motion not found: {motion_path}")
    loaded_fps: float | None = None
    if motion_path.suffix.lower() == ".npy":
        qpos = np.load(motion_path, allow_pickle=False)
    elif motion_path.suffix.lower() == ".npz":
        with np.load(motion_path, allow_pickle=False) as data:
            for key in (
                "qpos_36",
                "pred_qpos_36",
                "executed_qpos_36",
                "qpos",
            ):
                if key in data:
                    qpos = np.asarray(data[key])
                    break
            else:
                raise KeyError(f"No qpos36 array found in {motion_path}")
            if "fps" in data:
                loaded_fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    else:
        raise ValueError(f"G1 motion must be .npy or .npz, got {motion_path}")
    resolved_fps = float(fps) if fps is not None else loaded_fps
    if resolved_fps is None:
        raise ValueError(
            f"An explicit FPS is required because {motion_path} has no fps field"
        )
    if not np.isfinite(resolved_fps) or resolved_fps <= 0.0:
        raise ValueError(f"Motion FPS must be positive and finite, got {resolved_fps}")
    motion = np.asarray(qpos, dtype=np.float32)
    if motion.ndim == 3 and motion.shape[0] == 1:
        motion = motion[0]
    return coerce_g1_qpos_motion(motion), resolved_fps


@dataclass(frozen=True)
class OmgGmrRuntimeConfig:
    tracker_fps: float = 50.0
    history_fps: float = 30.0
    history_frames: int = 10
    planner_frames: int = 60
    replan_remaining_frames: int = 40
    audio_fps: float = 30.0
    audio_type: str = "audio"
    audio_feature_type: str = "current35"
    condition_audio_step_frames: int = 36
    request_timeout_ms: int = 120000
    status_interval_seconds: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("tracker_fps", self.tracker_fps),
            ("history_fps", self.history_fps),
            ("audio_fps", self.audio_fps),
            ("status_interval_seconds", self.status_interval_seconds),
        ):
            number = float(value)
            if not np.isfinite(number) or number <= 0.0:
                raise ValueError(f"{name} must be positive and finite, got {value}")
        if int(self.history_frames) <= 0:
            raise ValueError("history_frames must be positive")
        if int(self.planner_frames) <= 0:
            raise ValueError("planner_frames must be positive")
        if int(self.replan_remaining_frames) < 0:
            raise ValueError("replan_remaining_frames must be non-negative")
        if int(self.condition_audio_step_frames) <= 0:
            raise ValueError("condition_audio_step_frames must be positive")
        if int(self.request_timeout_ms) <= 0:
            raise ValueError("request_timeout_ms must be positive")
        if str(self.audio_type) not in {"audio", "feature"}:
            raise ValueError(f"Unsupported audio_type: {self.audio_type}")
        if str(self.audio_feature_type) != "current35":
            raise ValueError(
                f"Unsupported audio_feature_type: {self.audio_feature_type}"
            )


@dataclass(frozen=True)
class PendingReplan:
    request: RobotStateRequest
    snapshot: ConditionSnapshot
    started_perf_time: float
    condition_index_committed: bool


class OmgGmrRuntime:
    """One deterministic tracker tick of the live OMG → GMR bridge.

    The planner is asynchronous.  Every call to :meth:`step` still selects and
    publishes one G1 frame, including while a diffusion request is pending.
    """

    def __init__(
        self,
        *,
        config: OmgGmrRuntimeConfig,
        controller: DynamicConditionController,
        planner_client: Any,
        mux: G1MotionSourceMux,
        history: TrackerExecutionHistory,
        publisher: Any,
        sim_stream: Any | None = None,
        status_callback: Callable[[dict[str, Any]], None] | None = None,
        planner_client_factory: Callable[[], Any] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config
        self.controller = controller
        self.planner_client = planner_client
        self.mux = mux
        self.history = history
        self.publisher = publisher
        self.sim_stream = sim_stream
        self.status_callback = status_callback
        self.planner_client_factory = planner_client_factory
        self.clock = clock
        self.cursor = 0
        self.pending: PendingReplan | None = None
        self.active_plan_revision: int | None = None
        self.last_plan_id: int | None = None
        self._observed_revision = int(controller.current_revision)
        self._last_status_time = float("-inf")
        self._closed = False
        self._stale_plan_count = 0
        self._discarded_plan_count = 0
        self._accepted_plan_count = 0
        self._pending_timeout_reported = False

    def _emit(self, event: dict[str, Any]) -> None:
        if self.status_callback is not None:
            self.status_callback(dict(event))

    def _condition_metadata(self, snapshot: ConditionSnapshot) -> dict[str, Any]:
        metadata = snapshot.metadata(
            audio_fps=float(self.config.audio_fps),
            tracker_fps=float(self.config.tracker_fps),
            audio_type=(
                "audio"
                if snapshot.command_type == "audio"
                else str(self.config.audio_type)
            ),
            audio_feature_type=str(self.config.audio_feature_type),
            condition_audio_step_frames=int(
                self.config.condition_audio_step_frames
            ),
        )
        metadata["bridge"] = "omg_gmr"
        metadata["fixed_idle_for_stand"] = True
        return metadata

    def _begin_replan(self) -> None:
        if self.pending is not None:
            return
        snapshot = self.controller.snapshot_for_replan(
            current_tracker_frame=int(self.cursor)
        )
        if snapshot.command_type == "stand":
            return
        request = RobotStateRequest(
            qpos_36_history=self.history.planner_history(),
            history_fps=float(self.config.history_fps),
            tracker_frame=int(self.cursor),
            buffer_remaining_frames=int(self.mux.plan_remaining_frames),
            last_plan_id=self.last_plan_id,
            metadata=self._condition_metadata(snapshot),
        )
        started = float(self.clock())
        self.planner_client.begin_request(request)
        committed = self.controller.mark_replan_submitted(snapshot)
        self.pending = PendingReplan(
            request=request,
            snapshot=snapshot,
            started_perf_time=started,
            condition_index_committed=bool(committed),
        )
        self._pending_timeout_reported = False
        self.mux.mark_waiting_for_plan()
        self._emit(
            {
                "kind": "replan_requested",
                "request_id": request.request_id,
                "request_tracker_frame": int(request.tracker_frame),
                "command_id": snapshot.command_id,
                "command_type": snapshot.command_type,
                "command_revision": int(snapshot.revision),
                "condition_session_id": snapshot.condition_session_id,
                "condition_index": int(snapshot.condition_index),
                "condition_index_committed": bool(committed),
                "buffer_remaining_frames": int(request.buffer_remaining_frames),
            }
        )
        print(
            f"[OMG→GMR replan request] frame={self.cursor} "
            f"command_id={snapshot.command_id} type={snapshot.command_type} "
            f"revision={snapshot.revision} condition_index={snapshot.condition_index} "
            f"prompt={snapshot.condition_sequence!r}",
            flush=True,
        )

    def _handle_response(
        self,
        response: MotionPlanChunk,
        pending: PendingReplan,
    ) -> None:
        active = self.controller.snapshot()
        stale = bool(
            active.revision != pending.snapshot.revision
            or active.command_id != pending.snapshot.command_id
            or active.condition_session_id != pending.snapshot.condition_session_id
            or active.command_type == "stand"
        )
        accepted = False
        expired = False
        skip_frames = max(
            0,
            int(self.cursor) - int(response.request_tracker_frame),
        )
        self.last_plan_id = int(response.plan_id)
        if stale:
            self._stale_plan_count += 1
            self._discarded_plan_count += 1
        else:
            accepted = self.mux.accept_plan(
                response.qpos_36,
                source_fps=float(response.fps),
                skip_tracker_frames=skip_frames,
                plan_id=int(response.plan_id),
                command_id=pending.snapshot.command_id,
                command_revision=int(pending.snapshot.revision),
            )
            expired = not accepted
            if accepted:
                self.active_plan_revision = int(pending.snapshot.revision)
                self._accepted_plan_count += 1
            else:
                self._discarded_plan_count += 1
        latency = max(0.0, float(self.clock()) - pending.started_perf_time)
        event = {
            "kind": "replan_completed",
            "request_id": pending.request.request_id,
            "plan_id": int(response.plan_id),
            "request_tracker_frame": int(response.request_tracker_frame),
            "response_tracker_frame": int(self.cursor),
            "skip_tracker_frames": int(skip_frames),
            "latency_seconds": float(latency),
            "planner_latency_seconds": float(response.planning_latency_seconds),
            "prompt": response.prompt,
            "command_id": pending.snapshot.command_id,
            "command_type": pending.snapshot.command_type,
            "command_revision": int(pending.snapshot.revision),
            "condition_session_id": pending.snapshot.condition_session_id,
            "condition_index": int(pending.snapshot.condition_index),
            "stale_command": bool(stale),
            "accepted": bool(accepted),
            "expired": bool(expired),
            "active_command_id": active.command_id,
            "active_command_type": active.command_type,
            "active_command_revision": int(active.revision),
            "plan_remaining_frames": int(self.mux.plan_remaining_frames),
            "response_condition": response.metadata.get("realtime_condition"),
            "transport_timing": response.metadata.get("realtime_transport", {}),
        }
        self._emit(event)
        print(
            f"[OMG→GMR replan {response.plan_id:04d}] frame={self.cursor} "
            f"command_id={pending.snapshot.command_id} "
            f"type={pending.snapshot.command_type} revision={pending.snapshot.revision} "
            f"condition_index={pending.snapshot.condition_index} prompt={response.prompt!r} "
            f"skip={skip_frames} accepted={accepted} stale_command={stale} "
            f"latency={latency * 1000.0:.3f}ms",
            flush=True,
        )
        if stale:
            print(
                "[OMG→GMR] stale plan consumed and discarded; scheduling current command",
                flush=True,
            )
        elif expired:
            print(
                "[OMG→GMR] plan horizon expired during inference; scheduling a fresh plan",
                flush=True,
            )

    def _poll_response(self) -> None:
        if self.pending is None:
            return
        response = self.planner_client.poll_plan(timeout_ms=0)
        if response is None:
            elapsed_ms = (
                float(self.clock()) - self.pending.started_perf_time
            ) * 1000.0
            if elapsed_ms < float(self.config.request_timeout_ms):
                return
            if not self._pending_timeout_reported:
                self._pending_timeout_reported = True
                timed_out = self.pending
                self._emit(
                    {
                        "kind": "replan_timeout",
                        "request_id": timed_out.request.request_id,
                        "request_tracker_frame": int(
                            timed_out.request.tracker_frame
                        ),
                        "tracker_frame": int(self.cursor),
                        "elapsed_ms": float(elapsed_ms),
                        "command_id": timed_out.snapshot.command_id,
                        "command_type": timed_out.snapshot.command_type,
                        "command_revision": int(timed_out.snapshot.revision),
                        "transport_recreated": self.planner_client_factory
                        is not None,
                    }
                )
                print(
                    f"[OMG→GMR] planner request timed out after {elapsed_ms:.1f}ms; "
                    + (
                        "recreating REQ transport"
                        if self.planner_client_factory is not None
                        else "waiting for the pending REP response"
                    ),
                    flush=True,
                )
            if self.planner_client_factory is not None:
                self.planner_client.close()
                self.planner_client = self.planner_client_factory()
                self.pending = None
                self._discarded_plan_count += 1
                self._pending_timeout_reported = False
            return
        pending = self.pending
        self.pending = None
        self._pending_timeout_reported = False
        self._handle_response(response, pending)

    def _handle_command_transition(self) -> ConditionSnapshot:
        snapshot = self.controller.snapshot()
        if int(snapshot.revision) == self._observed_revision:
            return snapshot
        self._observed_revision = int(snapshot.revision)
        if snapshot.command_type == "stand":
            self.mux.return_to_idle(reason="stand", error=False)
            self.active_plan_revision = None
            print(
                f"[OMG→GMR fixed-idle] command_id={snapshot.command_id} "
                f"revision={snapshot.revision}; returning to fixed G1 pose",
                flush=True,
            )
        elif self.mux.plan_remaining_frames <= 0:
            self.mux.mark_waiting_for_plan()
        self._emit(
            {
                "kind": "output_command_transition",
                "tracker_frame": int(self.cursor),
                "command_id": snapshot.command_id,
                "command_type": snapshot.command_type,
                "command_revision": int(snapshot.revision),
                "condition_session_id": snapshot.condition_session_id,
                "output_policy": (
                    "fixed_idle" if snapshot.command_type == "stand" else "planner"
                ),
            }
        )
        return snapshot

    def _should_replan(self, snapshot: ConditionSnapshot) -> bool:
        if self.pending is not None or snapshot.command_type == "stand":
            return False
        if self.active_plan_revision != int(snapshot.revision):
            return True
        return (
            int(self.mux.plan_remaining_frames)
            <= int(self.config.replan_remaining_frames)
        )

    def _periodic_status(self, tick: G1MotionTick, redis_queued: bool) -> None:
        now = float(self.clock())
        if now - self._last_status_time < float(
            self.config.status_interval_seconds
        ):
            return
        self._last_status_time = now
        snapshot = self.controller.snapshot()
        publisher_status = (
            self.publisher.status() if hasattr(self.publisher, "status") else {}
        )
        self._emit(
            {
                "kind": "bridge_status",
                "tracker_frame": int(self.cursor),
                "command_id": snapshot.command_id,
                "command_type": snapshot.command_type,
                "command_revision": int(snapshot.revision),
                "condition_session_id": snapshot.condition_session_id,
                "condition_index": int(snapshot.condition_index),
                "audio_duration_seconds": snapshot.audio_duration_seconds,
                "audio_start_tracker_frame": snapshot.audio_start_tracker_frame,
                "audio_end_tracker_frame": snapshot.audio_end_tracker_frame,
                "pending_replan": self.pending is not None,
                "active_plan_revision": self.active_plan_revision,
                "stale_plan_count": int(self._stale_plan_count),
                "discarded_plan_count": int(self._discarded_plan_count),
                "accepted_plan_count": int(self._accepted_plan_count),
                "output_state": tick.state,
                "output_source": tick.source,
                "plan_remaining_frames": int(tick.plan_remaining_frames),
                "underflow_count": int(tick.underflow_count),
                "redis_queued": bool(redis_queued),
                "redis": publisher_status,
            }
        )

    def step(self) -> G1MotionTick:
        if self._closed:
            raise RuntimeError("OMG→GMR runtime is closed")
        audio_end_event = self.controller.update_for_tracker_frame(int(self.cursor))
        if audio_end_event is not None:
            self._emit(audio_end_event)
        snapshot = self._handle_command_transition()
        self._poll_response()
        snapshot = self.controller.snapshot()
        if self._should_replan(snapshot):
            self._begin_replan()
        tick = self.mux.tick(int(self.cursor))
        redis_queued = bool(
            self.publisher.publish(
                tick.qpos_36,
                timestamp=time.monotonic(),
            )
        )
        self.history.append(tick.qpos_36)
        if self.sim_stream is not None:
            active = self.controller.snapshot()
            self.sim_stream.update(
                tick.qpos_36,
                frame_index=int(self.cursor),
                overlay_lines=[
                    f"command: {active.command_type}",
                    f"condition: {active.condition_sequence}",
                    f"output: {tick.source} ({tick.state})",
                    f"tracker frame: {self.cursor}",
                    f"plan remaining: {tick.plan_remaining_frames}",
                    f"GMR Redis: {'queued' if redis_queued else 'not connected'}",
                ],
            )
        self._periodic_status(tick, redis_queued)
        self.cursor += 1
        return tick

    def status(self) -> dict[str, Any]:
        snapshot = self.controller.snapshot()
        return {
            "tracker_frame": int(self.cursor),
            "command_id": snapshot.command_id,
            "command_type": snapshot.command_type,
            "command_revision": int(snapshot.revision),
            "pending_replan": self.pending is not None,
            "active_plan_revision": self.active_plan_revision,
            "last_plan_id": self.last_plan_id,
            "mux": self.mux.status(),
            "publisher": (
                self.publisher.status() if hasattr(self.publisher, "status") else {}
            ),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.planner_client.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously publish OMG G1 qpos to the existing GMR G1→BUMI3 "
            "pipeline with GEM-style fixed idle arbitration."
        )
    )
    parser.add_argument("--connect", default="tcp://127.0.0.1:5571")
    parser.add_argument("--command-bind", default="tcp://127.0.0.1:5581")
    parser.add_argument(
        "--initial-condition-sequence",
        default="text: stand still",
        help="Initial dynamic condition. The default is handled as fixed idle without planning.",
    )
    parser.add_argument("--seed-motion", required=True)
    parser.add_argument("--seed-fps", type=float, default=None)
    parser.add_argument(
        "--idle-motion",
        default=None,
        help=(
            "Optional verified G1 qpos motion; its final frame overrides "
            "--idle-preset and becomes the fixed idle pose."
        ),
    )
    parser.add_argument("--idle-fps", type=float, default=None)
    parser.add_argument(
        "--idle-preset",
        choices=["neutral", "seed-last"],
        default="neutral",
        help=(
            "Fixed-idle source when --idle-motion is absent. 'neutral' uses "
            "zero legs, slightly open shoulders and hanging forearms; "
            "'seed-last' preserves the previous training-mean seed behavior."
        ),
    )
    parser.add_argument(
        "--initial-yaw-degrees",
        type=float,
        default=90.0,
        help=(
            "World yaw of the built-in neutral idle. Positive is "
            "counter-clockwise; the default applies the requested 90-degree "
            "initial rotation. Ignored for --idle-motion/seed-last."
        ),
    )
    parser.add_argument("--tracker-fps", type=float, default=50.0)
    parser.add_argument("--history-fps", type=float, default=30.0)
    parser.add_argument("--history-frames", type=int, default=10)
    parser.add_argument("--planner-frames", type=int, default=60)
    parser.add_argument("--replan-remaining-frames", type=int, default=40)
    parser.add_argument("--blend-seconds", type=float, default=0.20)
    parser.add_argument("--return-seconds", type=float, default=1.0)
    parser.add_argument(
        "--return-max-joint-speed",
        type=float,
        default=1.5,
        help=(
            "Maximum joint-space return rate in rad/s. It can extend "
            "--return-seconds for large pose changes."
        ),
    )
    parser.add_argument("--no-preserve-idle-xy", action="store_true")
    parser.add_argument(
        "--reset-idle-heading",
        action="store_true",
        help="Return to the canonical idle yaw instead of keeping the current yaw.",
    )
    parser.add_argument("--audio-fps", type=float, default=30.0)
    parser.add_argument("--audio-type", choices=["audio", "feature"], default="audio")
    parser.add_argument(
        "--audio-feature-type", choices=["current35"], default="current35"
    )
    parser.add_argument("--condition-audio-step-frames", type=int, default=None)
    parser.add_argument("--timeout-ms", type=int, default=120000)
    parser.add_argument("--redis-host", default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--redis-key", default="omg_online_frame_g1")
    parser.add_argument("--redis-ttl-ms", type=int, default=250)
    parser.add_argument("--redis-connect-timeout", type=float, default=0.5)
    parser.add_argument("--redis-socket-timeout", type=float, default=0.5)
    parser.add_argument("--redis-reconnect-interval", type=float, default=0.25)
    parser.add_argument("--sim-stream-bind", default=None)
    parser.add_argument("--sim-stream-fps", type=float, default=20.0)
    parser.add_argument("--sim-stream-width", type=int, default=1280)
    parser.add_argument("--sim-stream-height", type=int, default=720)
    parser.add_argument(
        "--sim-camera-view", choices=["back", "side", "iso", "front"], default="iso"
    )
    parser.add_argument(
        "--sim-follow-mode",
        choices=["none", "xy", "xyz", "heading"],
        default="xy",
    )
    parser.add_argument("--sim-camera-distance", type=float, default=4.5)
    parser.add_argument("--sim-camera-elevation", type=float, default=-18.0)
    parser.add_argument("--status-jsonl", default=None)
    parser.add_argument("--status-interval-seconds", type=float, default=1.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--num-frames", type=int, default=0)
    return parser.parse_args()


def _save_output(
    path: str | Path | None,
    frames: list[np.ndarray],
    *,
    fps: float,
) -> None:
    if path is None:
        return
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    qpos = (
        np.stack(frames, axis=0).astype(np.float32, copy=False)
        if frames
        else np.zeros((0, 36), dtype=np.float32)
    )
    np.savez_compressed(output, qpos_36=qpos, executed_qpos_36=qpos, fps=np.float32(fps))
    print(f"[OMG→GMR] saved {qpos.shape[0]} G1 frames to {output}", flush=True)


def main() -> None:
    args = _parse_args()
    if not args.continuous and int(args.num_frames) <= 0:
        raise ValueError("Use --continuous or provide a positive --num-frames")
    if int(args.timeout_ms) <= 0:
        raise ValueError("--timeout-ms must be positive")
    seed_motion, _seed_fps = load_qpos_motion(args.seed_motion, fps=args.seed_fps)
    idle_source = str(args.idle_preset)
    if args.idle_motion is not None:
        idle_motion, _idle_fps = load_qpos_motion(
            args.idle_motion,
            fps=args.idle_fps,
        )
        idle_source = f"file:{Path(args.idle_motion).expanduser()}"
    elif args.idle_preset == "neutral":
        idle_motion = neutral_g1_idle_qpos(
            yaw_degrees=float(args.initial_yaw_degrees)
        )[None, :]
    else:
        idle_motion = seed_motion

    # The planner history must describe frames that the mux actually emitted.
    # A separate idle pose must not be mixed with the training-mean seed during
    # the first request.
    initial_tracker_motion = np.repeat(
        idle_motion[-1][None, :],
        max(2, int(args.history_frames)),
        axis=0,
    ).astype(np.float32, copy=False)
    if args.condition_audio_step_frames is None:
        audio_step = compute_condition_audio_step_frames(
            planner_frames=int(args.planner_frames),
            history_fps=float(args.history_fps),
            tracker_fps=float(args.tracker_fps),
            replan_remaining_frames=int(args.replan_remaining_frames),
            audio_fps=float(args.audio_fps),
        )
        print(f"[OMG→GMR] computed audio step frames={audio_step}", flush=True)
    else:
        audio_step = int(args.condition_audio_step_frames)
        if audio_step <= 0:
            raise ValueError("--condition-audio-step-frames must be positive")
    runtime_config = OmgGmrRuntimeConfig(
        tracker_fps=float(args.tracker_fps),
        history_fps=float(args.history_fps),
        history_frames=int(args.history_frames),
        planner_frames=int(args.planner_frames),
        replan_remaining_frames=int(args.replan_remaining_frames),
        audio_fps=float(args.audio_fps),
        audio_type=str(args.audio_type),
        audio_feature_type=str(args.audio_feature_type),
        condition_audio_step_frames=int(audio_step),
        request_timeout_ms=int(args.timeout_ms),
        status_interval_seconds=float(args.status_interval_seconds),
    )
    controller = DynamicConditionController(
        tracker_fps=float(args.tracker_fps),
        initial_condition_sequence=str(args.initial_condition_sequence),
    )
    mux = G1MotionSourceMux(
        idle_motion[-1],
        G1MotionMuxConfig(
            tracker_fps=float(args.tracker_fps),
            blend_seconds=float(args.blend_seconds),
            return_seconds=float(args.return_seconds),
            preserve_idle_xy=not bool(args.no_preserve_idle_xy),
            preserve_idle_heading=not bool(args.reset_idle_heading),
            return_max_joint_speed_radians=float(args.return_max_joint_speed),
        ),
    )
    history = TrackerExecutionHistory(
        initial_tracker_motion,
        tracker_fps=float(args.tracker_fps),
        history_fps=float(args.history_fps),
        history_frames=int(args.history_frames),
    )
    planner = ZmqPlanClient(str(args.connect))
    sync_redis = RedisQposPublisher(
        RedisQposPublisherConfig(
            host=str(args.redis_host),
            port=int(args.redis_port),
            db=int(args.redis_db),
            key=str(args.redis_key),
            ttl_ms=int(args.redis_ttl_ms),
            connect_timeout_seconds=float(args.redis_connect_timeout),
            socket_timeout_seconds=float(args.redis_socket_timeout),
            reconnect_interval_seconds=float(args.redis_reconnect_interval),
        )
    )
    redis_publisher = AsyncRedisQposPublisher(sync_redis)
    sim_stream = None
    status_callback = lambda event: append_jsonl(args.status_jsonl, event)
    runtime: OmgGmrRuntime | None = None
    command_server: DynamicCommandServer | None = None
    executed: list[np.ndarray] = []
    interrupted = False
    try:
        redis_publisher.start()
        if args.sim_stream_bind is not None:
            from omg.realtime.sim_stream import SimStreamConfig, SimStreamServer

            sim_stream = SimStreamServer(
                SimStreamConfig(
                    bind=str(args.sim_stream_bind),
                    fps=float(args.sim_stream_fps),
                    width=int(args.sim_stream_width),
                    height=int(args.sim_stream_height),
                    camera_view=str(args.sim_camera_view),
                    follow_mode=str(args.sim_follow_mode),
                    camera_distance=float(args.sim_camera_distance),
                    camera_elevation=float(args.sim_camera_elevation),
                )
            )
            sim_stream.start()
        runtime = OmgGmrRuntime(
            config=runtime_config,
            controller=controller,
            planner_client=planner,
            mux=mux,
            history=history,
            publisher=redis_publisher,
            sim_stream=sim_stream,
            status_callback=status_callback,
            planner_client_factory=lambda: ZmqPlanClient(str(args.connect)),
        )
        command_server = DynamicCommandServer(
            CommandServerConfig(bind=str(args.command_bind)),
            controller,
            status_callback=status_callback,
        )
        command_server.start()
        print(
            f"[OMG→GMR] fixed-rate G1 output={args.tracker_fps:g}Hz "
            f"Redis={args.redis_host}:{args.redis_port}/{args.redis_db} "
            f"key={args.redis_key}",
            flush=True,
        )
        print(f"[OMG→GMR] fixed idle source={idle_source}", flush=True)
        if args.idle_motion is None and args.idle_preset == "neutral":
            print(
                "[OMG→GMR] neutral idle: legs/waist=0, shoulder_roll="
                "+/-0.12rad, elbows=pi/2, initial_yaw="
                f"{float(args.initial_yaw_degrees):g}deg; use a verified "
                "--idle-motion for hardware",
                flush=True,
            )
        else:
            print(
                "[OMG→GMR] --initial-yaw-degrees is ignored unless the "
                "built-in neutral idle is active",
                flush=True,
            )
        print(
            "[OMG→GMR] stand return: smooth minimum="
            f"{float(args.return_seconds):g}s max_joint_speed="
            f"{float(args.return_max_joint_speed):g}rad/s preserve_xy="
            f"{not bool(args.no_preserve_idle_xy)} preserve_heading="
            f"{not bool(args.reset_idle_heading)}",
            flush=True,
        )
        print(
            "[OMG→GMR] stand/no command uses a fixed pose; no stand prompt is sent to Planner",
            flush=True,
        )
        period = 1.0 / float(args.tracker_fps)
        next_deadline = time.perf_counter()
        assert runtime is not None
        while args.continuous or runtime.cursor < int(args.num_frames):
            tick = runtime.step()
            if args.output is not None:
                executed.append(tick.qpos_36.copy())
            next_deadline += period
            remaining = next_deadline - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
            elif remaining < -period:
                next_deadline = time.perf_counter()
    except KeyboardInterrupt:
        interrupted = True
        print("[OMG→GMR] interrupted", flush=True)
    finally:
        close_errors: list[BaseException] = []
        for close in (
            command_server.close if command_server is not None else None,
            runtime.close if runtime is not None else planner.close,
            redis_publisher.close,
            (lambda: sim_stream.close()) if sim_stream is not None else None,
        ):
            if close is None:
                continue
            try:
                close()
            except BaseException as exc:
                close_errors.append(exc)
        _save_output(
            args.output,
            executed,
            fps=float(args.tracker_fps),
        )
        append_jsonl(
            args.status_jsonl,
            {
                "kind": "bridge_stopped",
                "tracker_frame": 0 if runtime is None else int(runtime.cursor),
                "interrupted": bool(interrupted),
                "close_errors": [f"{type(exc).__name__}: {exc}" for exc in close_errors],
            },
        )
        if close_errors:
            raise RuntimeError(
                "Errors while closing OMG→GMR bridge: "
                + "; ".join(str(exc) for exc in close_errors)
            )


if __name__ == "__main__":
    main()
