from __future__ import annotations

import argparse
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from omg.realtime.audio_playback import SynchronizedAudioPlayer
from omg.realtime.bumi_motion_mux import (
    BUMI_QPOS_DIM,
    BumiMotionMuxConfig,
    BumiMotionSourceMux,
    BumiMotionTick,
    BumiTrackerExecutionHistory,
    BumiTrajectoryHistory,
    blend_bumi_history_toward_measurement,
    root_tilt_angle_wxyz,
    root_tilt_error_angle_wxyz,
)
from omg.realtime.command_server import CommandServerConfig, DynamicCommandServer
from omg.realtime.dynamic_condition import (
    ConditionSnapshot,
    DynamicConditionController,
    compute_condition_audio_step_frames,
)
from omg.realtime.gmt_trajectory import (
    FLAG_AUDIO,
    FLAG_ERROR,
    FLAG_FIXED_IDLE,
    FLAG_TEXT,
    FLAG_TRANSITION,
    TRAJECTORY_FRAME_COUNT,
    GmtLowStateFeedback,
    GmtTrajectoryAck,
    GmtPolicyContract,
    build_policy_default_idle_qpos,
    make_trajectory_packet,
)
from omg.realtime.protocol import MotionPlanChunk, RobotStateRequest
from omg.realtime.redis_trajectory import (
    AsyncRedisTrajectoryPublisher,
    RedisLowStateFeedbackReader,
    RedisLowStateFeedbackReaderConfig,
    RedisTrajectoryAckReader,
    RedisTrajectoryAckReaderConfig,
    RedisTrajectoryPublisher,
    RedisTrajectoryPublisherConfig,
)
from omg.realtime.status_log import append_jsonl
from omg.realtime.transport import ZmqPlanClient


DEFAULT_HYBRID_HISTORY_BETA = (
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    1.00,
)


@dataclass(frozen=True)
class BumiGmtRuntimeConfig:
    tracker_fps: float = 50.0
    history_fps: float = 30.0
    history_frames: int = 10
    planner_frames: int = 60
    replan_remaining_frames: int = 60
    audio_fps: float = 30.0
    audio_type: str = "audio"
    audio_feature_type: str = "current35"
    condition_audio_step_frames: int = 24
    request_timeout_ms: int = 120000
    audio_ack_timeout_seconds: float = 2.0
    status_interval_seconds: float = 1.0
    history_source: str = "reference"
    lowstate_max_age_seconds: float = 0.2
    hybrid_history_beta: tuple[float, ...] = DEFAULT_HYBRID_HISTORY_BETA
    hybrid_max_rotation_error_degrees: float = 25.0
    hybrid_max_tilt_error_degrees: float = 20.0
    hybrid_hard_root_tilt_degrees: float = 45.0

    def __post_init__(self) -> None:
        for name, value in (
            ("tracker_fps", self.tracker_fps),
            ("history_fps", self.history_fps),
            ("audio_fps", self.audio_fps),
            ("status_interval_seconds", self.status_interval_seconds),
            ("audio_ack_timeout_seconds", self.audio_ack_timeout_seconds),
            ("lowstate_max_age_seconds", self.lowstate_max_age_seconds),
            (
                "hybrid_max_rotation_error_degrees",
                self.hybrid_max_rotation_error_degrees,
            ),
            ("hybrid_max_tilt_error_degrees", self.hybrid_max_tilt_error_degrees),
            ("hybrid_hard_root_tilt_degrees", self.hybrid_hard_root_tilt_degrees),
        ):
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if int(self.history_frames) <= 0 or int(self.planner_frames) <= 0:
            raise ValueError("history_frames and planner_frames must be positive")
        if int(self.replan_remaining_frames) < 0:
            raise ValueError("replan_remaining_frames must be non-negative")
        if int(self.condition_audio_step_frames) <= 0:
            raise ValueError("condition_audio_step_frames must be positive")
        if int(self.request_timeout_ms) <= 0:
            raise ValueError("request_timeout_ms must be positive")
        if self.history_source not in {"reference", "lowstate", "hybrid"}:
            raise ValueError(
                "history_source must be 'reference', 'lowstate', or 'hybrid'"
            )
        if self.history_source == "hybrid":
            beta = np.asarray(self.hybrid_history_beta, dtype=np.float64)
            if beta.shape != (int(self.history_frames),):
                raise ValueError(
                    "hybrid_history_beta must contain exactly history_frames values"
                )
            if (
                not np.isfinite(beta).all()
                or np.any(beta < 0.0)
                or np.any(beta > 1.0)
                or np.any(np.diff(beta) < 0.0)
            ):
                raise ValueError(
                    "hybrid_history_beta must be finite, non-decreasing, and within "
                    "[0,1]"
                )
            if not np.isclose(beta[-1], 1.0):
                raise ValueError("hybrid_history_beta must end at 1.0")


@dataclass(frozen=True)
class PendingReplan:
    request: RobotStateRequest
    snapshot: ConditionSnapshot
    started_perf_time: float
    condition_index_committed: bool


@dataclass(frozen=True)
class PendingAudioAck:
    command_id: str
    command_revision: int
    packet_sequence: int
    first_motion_tracker_frame: int
    armed_perf_time: float


class BumiGmtRuntime:
    """Native BUMI OMG planner loop and trajectory_v1 publisher."""

    def __init__(
        self,
        *,
        config: BumiGmtRuntimeConfig,
        controller: DynamicConditionController,
        planner_client: Any,
        mux: BumiMotionSourceMux,
        planner_history: BumiTrackerExecutionHistory,
        trajectory_history: BumiTrajectoryHistory,
        publisher: Any,
        native_to_gmt: np.ndarray,
        joint_order_hash: bytes,
        audio_player: SynchronizedAudioPlayer,
        ack_reader: Any | None = None,
        lowstate_history: BumiTrackerExecutionHistory | None = None,
        lowstate_reader: Any | None = None,
        sim_stream: Any | None = None,
        status_callback: Callable[[dict[str, Any]], None] | None = None,
        planner_client_factory: Callable[[], Any] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        stream_id: int | None = None,
    ) -> None:
        self.config = config
        self.controller = controller
        self.planner_client = planner_client
        self.mux = mux
        self.planner_history = planner_history
        self.trajectory_history = trajectory_history
        self.publisher = publisher
        self.native_to_gmt = np.asarray(native_to_gmt, dtype=np.int64)
        self.joint_order_hash = bytes(joint_order_hash)
        self.audio_player = audio_player
        self.ack_reader = ack_reader
        self.lowstate_history = lowstate_history
        self.lowstate_reader = lowstate_reader
        if self.config.history_source in {"lowstate", "hybrid"} and (
            self.lowstate_history is None or self.lowstate_reader is None
        ):
            raise ValueError(
                f"history_source={self.config.history_source} requires a LowState "
                "reader and history buffer"
            )
        self.sim_stream = sim_stream
        self.status_callback = status_callback
        self.planner_client_factory = planner_client_factory
        self.clock = clock
        self.stream_id = secrets.randbits(64) if stream_id is None else int(stream_id)
        self.cursor = 0
        self.packet_sequence = 0
        self.pending: PendingReplan | None = None
        self.active_plan_revision: int | None = None
        self.last_plan_id: int | None = None
        initial = controller.snapshot()
        self._observed_revision = int(initial.revision)
        self._observed_command_id = initial.command_id
        self._last_status_time = float("-inf")
        self._closed = False
        self._stale_plan_count = 0
        self._discarded_plan_count = 0
        self._accepted_plan_count = 0
        self._planner_error_count = 0
        self._failed_command_revision: int | None = None
        self._pending_audio_ack: PendingAudioAck | None = None
        self._lowstate_stream_fresh = False
        self._lowstate_valid_tracker_frames = 0
        self._lowstate_unique_samples = 0
        self._last_lowstate_identity: tuple[int, int] | None = None
        self._lowstate_rejection_reason: str | None = None
        self._lowstate_ready_announced = False
        self._latest_measured_root_tilt_degrees: float | None = None
        self._latest_root_tilt_error_degrees: float | None = None

    def _emit(self, event: dict[str, Any]) -> None:
        if self.status_callback is not None:
            self.status_callback(dict(event))

    def _lowstate_history_ready(self) -> bool:
        if self.config.history_source == "reference":
            return True
        assert self.lowstate_history is not None
        return bool(
            self._lowstate_stream_fresh
            and self._lowstate_valid_tracker_frames
            >= int(self.lowstate_history.required_tracker_frames)
            and self._lowstate_unique_samples >= int(self.config.history_frames)
        )

    def _planner_history_for_replan(self) -> np.ndarray | None:
        if self.config.history_source == "reference":
            return self.planner_history.planner_history()
        if not self._lowstate_history_ready():
            return None
        assert self.lowstate_history is not None
        measured = self.lowstate_history.planner_history()
        if self.config.history_source == "lowstate":
            return measured
        reference = self.planner_history.planner_history()
        return blend_bumi_history_toward_measurement(
            reference,
            measured,
            beta=self.config.hybrid_history_beta,
            max_rotation_error_radians=np.deg2rad(
                self.config.hybrid_max_rotation_error_degrees
            ),
        )

    def _invalidate_lowstate_history(
        self, *, reason: str, reference_qpos: np.ndarray
    ) -> None:
        was_fresh = self._lowstate_stream_fresh
        previous_reason = self._lowstate_rejection_reason
        self._lowstate_stream_fresh = False
        self._lowstate_valid_tracker_frames = 0
        self._lowstate_unique_samples = 0
        self._last_lowstate_identity = None
        self._lowstate_rejection_reason = reason
        self._lowstate_ready_announced = False
        if self.lowstate_history is not None:
            self.lowstate_history.reset(reference_qpos)
        if was_fresh or reason != previous_reason:
            print(
                f"[BUMI history] LowState unavailable ({reason}); "
                f"pausing new replans until {self.config.history_frames} measured "
                "history frames are ready",
                flush=True,
            )
            self._emit(
                {
                    "kind": "lowstate_history_unavailable",
                    "tracker_frame": self.cursor,
                    "reason": reason,
                }
            )

    def _append_history_sample(self, reference_qpos: np.ndarray) -> None:
        """Append reference history and, when selected, a fused LowState pose."""

        self.planner_history.append(reference_qpos)
        if self.config.history_source == "reference":
            return
        assert self.lowstate_reader is not None
        assert self.lowstate_history is not None
        sample: GmtLowStateFeedback | None = self.lowstate_reader.latest_feedback(
            max_age_seconds=float(self.config.lowstate_max_age_seconds)
        )
        if sample is None:
            self._invalidate_lowstate_history(
                reason="missing_or_stale", reference_qpos=reference_qpos
            )
            return
        if bytes(sample.joint_order_hash) != self.joint_order_hash:
            self._invalidate_lowstate_history(
                reason="joint_order_hash_mismatch", reference_qpos=reference_qpos
            )
            return

        measured_tilt_degrees = float(
            np.rad2deg(root_tilt_angle_wxyz(sample.root_quat_wxyz))
        )
        tilt_error_degrees = float(
            np.rad2deg(
                root_tilt_error_angle_wxyz(
                    reference_qpos[3:7], sample.root_quat_wxyz
                )
            )
        )
        self._latest_measured_root_tilt_degrees = measured_tilt_degrees
        self._latest_root_tilt_error_degrees = tilt_error_degrees
        if self.config.history_source == "hybrid":
            if measured_tilt_degrees > float(
                self.config.hybrid_hard_root_tilt_degrees
            ):
                self._invalidate_lowstate_history(
                    reason=(
                        "unsafe_measured_root_tilt:"
                        f"{measured_tilt_degrees:.2f}deg"
                    ),
                    reference_qpos=reference_qpos,
                )
                return
            if tilt_error_degrees > float(
                self.config.hybrid_max_tilt_error_degrees
            ):
                self._invalidate_lowstate_history(
                    reason=(
                        "unsafe_root_tilt_error:"
                        f"{tilt_error_degrees:.2f}deg"
                    ),
                    reference_qpos=reference_qpos,
                )
                return

        fused = np.asarray(reference_qpos, dtype=np.float32).copy()
        # root xyz stays exactly on the current reference trajectory.  LowState
        # supplies the observable root orientation and the measured joints.
        fused[3:7] = np.asarray(sample.root_quat_wxyz, dtype=np.float32)
        measured_native = np.empty((21,), dtype=np.float32)
        measured_native[self.native_to_gmt] = np.asarray(
            sample.joint_pos, dtype=np.float32
        )
        fused[7:] = measured_native

        if not self._lowstate_stream_fresh:
            self.lowstate_history.reset(fused)
        identity = (int(sample.sequence), int(sample.captured_unix_ns))
        if identity != self._last_lowstate_identity:
            self._lowstate_unique_samples += 1
            self._last_lowstate_identity = identity
        self.lowstate_history.append(fused)
        self._lowstate_valid_tracker_frames += 1
        self._lowstate_stream_fresh = True
        self._lowstate_rejection_reason = None
        if self._lowstate_history_ready() and not self._lowstate_ready_announced:
            self._lowstate_ready_announced = True
            print(
                f"[BUMI history] {self.config.history_source} history ready: "
                f"{self.config.history_frames} frames @ {self.config.history_fps:g} Hz; "
                "root xyz=reference, root quat/joints use measured feedback",
                flush=True,
            )
            self._emit(
                {
                    "kind": "lowstate_history_ready",
                    "tracker_frame": self.cursor,
                    "valid_tracker_frames": self._lowstate_valid_tracker_frames,
                    "unique_samples": self._lowstate_unique_samples,
                }
            )

    def _history_status(self) -> dict[str, Any]:
        reader_status = (
            self.lowstate_reader.status()
            if self.lowstate_reader is not None
            and hasattr(self.lowstate_reader, "status")
            else None
        )
        return {
            "source": self.config.history_source,
            "ready": self._lowstate_history_ready(),
            "root_xyz_source": "reference",
            "root_quaternion_source": (
                "reference_to_lowstate_geodesic"
                if self.config.history_source == "hybrid"
                else self.config.history_source
            ),
            "joint_position_source": (
                "reference_to_lowstate_beta"
                if self.config.history_source == "hybrid"
                else self.config.history_source
            ),
            "valid_tracker_frames": int(self._lowstate_valid_tracker_frames),
            "unique_lowstate_samples": int(self._lowstate_unique_samples),
            "required_tracker_frames": (
                None
                if self.lowstate_history is None
                else int(self.lowstate_history.required_tracker_frames)
            ),
            "rejection_reason": self._lowstate_rejection_reason,
            "hybrid_beta": (
                list(self.config.hybrid_history_beta)
                if self.config.history_source == "hybrid"
                else None
            ),
            "hybrid_max_rotation_error_degrees": float(
                self.config.hybrid_max_rotation_error_degrees
            ),
            "hybrid_max_tilt_error_degrees": float(
                self.config.hybrid_max_tilt_error_degrees
            ),
            "hybrid_hard_root_tilt_degrees": float(
                self.config.hybrid_hard_root_tilt_degrees
            ),
            "measured_root_tilt_degrees": self._latest_measured_root_tilt_degrees,
            "root_tilt_error_degrees": self._latest_root_tilt_error_degrees,
            "reader": reader_status,
        }

    def _condition_metadata(self, snapshot: ConditionSnapshot) -> dict[str, Any]:
        metadata = snapshot.metadata(
            audio_fps=float(self.config.audio_fps),
            tracker_fps=float(self.config.tracker_fps),
            audio_type=(
                "audio" if snapshot.command_type == "audio" else self.config.audio_type
            ),
            audio_feature_type=self.config.audio_feature_type,
            condition_audio_step_frames=int(self.config.condition_audio_step_frames),
        )
        metadata.update(
            {
                "bridge": "omg_bumi_gmt",
                "robot_name": "bumi",
                "state_dim": BUMI_QPOS_DIM,
                "fixed_idle_for_stand": True,
                "history_source": self.config.history_source,
                "history_root_xyz_source": "reference",
                "hybrid_history_beta": list(self.config.hybrid_history_beta),
                "hybrid_max_rotation_error_degrees": float(
                    self.config.hybrid_max_rotation_error_degrees
                ),
                "condition_elapsed_tracker_frames": int(
                    self.controller.audio_elapsed_tracker_frames(self.cursor)
                ),
            }
        )
        return metadata

    def _planner_failed(
        self,
        *,
        operation: str,
        error: BaseException,
        command_revision: int | None = None,
    ) -> None:
        self._planner_error_count += 1
        if command_revision is None and self.pending is not None:
            command_revision = self.pending.snapshot.revision
        self._failed_command_revision = (
            None if command_revision is None else int(command_revision)
        )
        self._discarded_plan_count += int(self.pending is not None)
        self.pending = None
        self.active_plan_revision = None
        self.mux.return_to_idle(reason=f"planner_{operation}_error", error=True)
        self._emit(
            {
                "kind": "planner_error",
                "operation": operation,
                "tracker_frame": self.cursor,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        print(
            f"[OMG BUMI→GMT] planner {operation} failed: "
            f"{type(error).__name__}: {error}; returning to fixed idle",
            flush=True,
        )
        if self.planner_client_factory is not None:
            try:
                self.planner_client.close()
            finally:
                self.planner_client = self.planner_client_factory()

    def _begin_replan(self) -> None:
        if self.pending is not None:
            return
        # Deliberately do not start the audio timer here.  It is armed on the
        # first generated output tick, after diffusion has completed.
        snapshot = self.controller.snapshot_for_replan(
            current_tracker_frame=self.cursor, start_audio=False
        )
        if snapshot.command_type == "stand":
            return
        planner_history = self._planner_history_for_replan()
        if planner_history is None:
            return
        request = RobotStateRequest(
            qpos_36_history=planner_history,
            history_fps=float(self.config.history_fps),
            tracker_frame=self.cursor,
            buffer_remaining_frames=self.mux.plan_remaining_frames,
            last_plan_id=self.last_plan_id,
            metadata=self._condition_metadata(snapshot),
        )
        started = float(self.clock())
        try:
            self.planner_client.begin_request(request)
        except BaseException as exc:
            self._planner_failed(
                operation="request",
                error=exc,
                command_revision=snapshot.revision,
            )
            return
        committed = self.controller.mark_replan_submitted(snapshot)
        self.pending = PendingReplan(request, snapshot, started, bool(committed))
        self.mux.mark_waiting_for_plan()
        self._emit(
            {
                "kind": "replan_requested",
                "tracker_frame": self.cursor,
                "request_id": request.request_id,
                "command_id": snapshot.command_id,
                "command_type": snapshot.command_type,
                "command_revision": snapshot.revision,
                "condition_index": snapshot.condition_index,
                "condition_index_committed": bool(committed),
                "history_source": self.config.history_source,
            }
        )
        print(
            f"[BUMI→GMT replan request] frame={self.cursor} "
            f"command_id={snapshot.command_id} type={snapshot.command_type} "
            f"revision={snapshot.revision} condition_index={snapshot.condition_index} "
            f"history={self.config.history_source} prompt={snapshot.condition_sequence!r}",
            flush=True,
        )

    def _handle_response(self, response: MotionPlanChunk, pending: PendingReplan) -> None:
        if response.robot_name != "bumi" or response.state_dim != BUMI_QPOS_DIM:
            raise ValueError(
                f"Planner returned {response.robot_name}/{response.state_dim}; expected bumi/28"
            )
        if response.qpos.shape != (
            int(self.config.planner_frames),
            BUMI_QPOS_DIM,
        ):
            raise ValueError(
                "Planner returned an unexpected BUMI plan shape: "
                f"expected ({self.config.planner_frames},{BUMI_QPOS_DIM}), "
                f"got {response.qpos.shape}"
            )
        active = self.controller.snapshot()
        stale = bool(
            active.revision != pending.snapshot.revision
            or active.command_id != pending.snapshot.command_id
            or active.condition_session_id != pending.snapshot.condition_session_id
            or active.command_type == "stand"
        )
        self.last_plan_id = int(response.plan_id)
        accepted = False
        if stale:
            self._stale_plan_count += 1
            self._discarded_plan_count += 1
        else:
            # Diffusion latency must not consume the beginning of a newly
            # accepted command.  Once that command is already rolling, skip
            # the elapsed tracker frames so replans stay on execution time.
            first_plan_for_command = bool(
                self.active_plan_revision != pending.snapshot.revision
            )
            skip_frames = (
                0
                if first_plan_for_command
                else max(0, self.cursor - int(response.request_tracker_frame))
            )
            accepted = self.mux.accept_plan(
                response.qpos,
                source_fps=float(response.fps),
                skip_tracker_frames=skip_frames,
                plan_id=response.plan_id,
                command_id=pending.snapshot.command_id,
                command_revision=pending.snapshot.revision,
            )
            if accepted:
                self.active_plan_revision = pending.snapshot.revision
                self._accepted_plan_count += 1
            else:
                self._discarded_plan_count += 1
        latency = max(0.0, float(self.clock()) - pending.started_perf_time)
        self._emit(
            {
                "kind": "replan_completed",
                "tracker_frame": self.cursor,
                "plan_id": response.plan_id,
                "command_id": pending.snapshot.command_id,
                "command_type": pending.snapshot.command_type,
                "command_revision": pending.snapshot.revision,
                "stale_command": stale,
                "accepted": accepted,
                "latency_seconds": latency,
            }
        )
        print(
            f"[BUMI→GMT replan {response.plan_id:04d}] frame={self.cursor} "
            f"type={pending.snapshot.command_type} revision={pending.snapshot.revision} "
            f"accepted={accepted} stale_command={stale} latency={latency * 1000:.1f}ms",
            flush=True,
        )

    def _poll_response(self) -> None:
        if self.pending is None:
            return
        try:
            response = self.planner_client.poll_plan(timeout_ms=0)
        except BaseException as exc:
            self._planner_failed(operation="response", error=exc)
            return
        if response is None:
            elapsed_ms = (float(self.clock()) - self.pending.started_perf_time) * 1000.0
            if elapsed_ms >= float(self.config.request_timeout_ms):
                self._planner_failed(
                    operation="timeout",
                    error=TimeoutError(f"request exceeded {elapsed_ms:.1f}ms"),
                )
            return
        pending, self.pending = self.pending, None
        try:
            self._handle_response(response, pending)
        except BaseException as exc:
            self._planner_failed(
                operation="validation",
                error=exc,
                command_revision=pending.snapshot.revision,
            )

    def _handle_command_transition(self) -> ConditionSnapshot:
        snapshot = self.controller.snapshot()
        changed = bool(
            snapshot.revision != self._observed_revision
            or snapshot.command_id != self._observed_command_id
        )
        if not changed:
            return snapshot
        previous_command_id = self._observed_command_id
        self._observed_revision = snapshot.revision
        self._observed_command_id = snapshot.command_id
        self._failed_command_revision = None
        if self.audio_player.command_id == previous_command_id:
            self.audio_player.stop(reason="command_changed")
        if self._pending_audio_ack is not None:
            self._pending_audio_ack = None
        if snapshot.command_type == "stand":
            self.mux.return_to_idle(reason="stand", error=False)
            self.active_plan_revision = None
            print(
                f"[BUMI→GMT fixed-idle] command_id={snapshot.command_id} "
                f"revision={snapshot.revision}; planner request skipped",
                flush=True,
            )
        elif self.mux.plan_remaining_frames <= 0:
            self.mux.mark_waiting_for_plan()
        self._emit(
            {
                "kind": "output_command_transition",
                "tracker_frame": self.cursor,
                "command_id": snapshot.command_id,
                "command_type": snapshot.command_type,
                "command_revision": snapshot.revision,
                "output_policy": (
                    "fixed_idle" if snapshot.command_type == "stand" else "planner"
                ),
            }
        )
        return snapshot

    def _should_replan(self, snapshot: ConditionSnapshot) -> bool:
        if self.pending is not None or snapshot.command_type == "stand":
            return False
        if self._failed_command_revision == snapshot.revision:
            return False
        if not self._lowstate_history_ready():
            return False
        return bool(
            self.active_plan_revision != snapshot.revision
            or self.mux.plan_remaining_frames <= self.config.replan_remaining_frames
        )

    @staticmethod
    def _packet_flags(tick: BumiMotionTick, snapshot: ConditionSnapshot) -> int:
        flags = 0
        if tick.source in {"fixed_idle", "waiting_plan"}:
            flags |= FLAG_FIXED_IDLE
        if tick.source in {"generated_blend", "return_to_idle", "error_return"}:
            flags |= FLAG_TRANSITION
        if tick.source == "error_return":
            flags |= FLAG_ERROR
        if snapshot.command_type == "text":
            flags |= FLAG_TEXT
        elif snapshot.command_type == "audio":
            flags |= FLAG_AUDIO
        return flags

    def _audio_requires_gmt_ack(self) -> bool:
        return bool(
            self.ack_reader is not None
            and bool(getattr(self.audio_player, "enabled", True))
        )

    def _start_audio_without_ack_if_needed(
        self, tick: BumiMotionTick, snapshot: ConditionSnapshot
    ) -> None:
        if (
            snapshot.command_type != "audio"
            or tick.command_revision != snapshot.revision
            or tick.source not in {"generated_blend", "generated"}
        ):
            return
        if self._audio_requires_gmt_ack():
            return
        started = self.controller.mark_audio_execution_started(
            snapshot, current_tracker_frame=self.cursor
        )
        if started:
            active = self.controller.active_status()
            audio_path = active.get("audio_path")
            if audio_path is not None:
                self.audio_player.start(
                    audio_path,
                    command_id=snapshot.command_id,
                    tracker_frame=self.cursor,
                )

    def _arm_audio_ack_if_needed(
        self,
        tick: BumiMotionTick,
        snapshot: ConditionSnapshot,
        *,
        packet_sequence: int,
        redis_queued: bool,
    ) -> None:
        if (
            not self._audio_requires_gmt_ack()
            or not redis_queued
            or self._pending_audio_ack is not None
            or snapshot.command_type != "audio"
            or snapshot.audio_start_tracker_frame is not None
            or tick.command_revision != snapshot.revision
            or tick.source not in {"generated_blend", "generated"}
        ):
            return
        self._pending_audio_ack = PendingAudioAck(
            command_id=snapshot.command_id,
            command_revision=int(snapshot.revision),
            packet_sequence=int(packet_sequence),
            first_motion_tracker_frame=int(self.cursor),
            armed_perf_time=float(self.clock()),
        )
        self._emit(
            {
                "kind": "audio_waiting_for_gmt_ack",
                "tracker_frame": self.cursor,
                "command_id": snapshot.command_id,
                "command_revision": snapshot.revision,
                "stream_id": self.stream_id,
                "packet_sequence": int(packet_sequence),
            }
        )
        print(
            f"[audio-playback] waiting for GMT ACK command_id={snapshot.command_id} "
            f"stream_id={self.stream_id} sequence={packet_sequence}",
            flush=True,
        )

    def _poll_audio_ack(self) -> None:
        pending = self._pending_audio_ack
        if pending is None or self.ack_reader is None:
            return
        snapshot = self.controller.snapshot()
        if (
            snapshot.command_type != "audio"
            or snapshot.command_id != pending.command_id
            or snapshot.revision != pending.command_revision
        ):
            self._pending_audio_ack = None
            return
        ack: GmtTrajectoryAck | None = self.ack_reader.latest_ack()
        matches = bool(
            ack is not None
            and ack.stream_id == self.stream_id
            and ack.sequence >= pending.packet_sequence
            and ack.command_revision == pending.command_revision
        )
        if matches:
            assert ack is not None
            self._pending_audio_ack = None
            started = self.controller.mark_audio_execution_started(
                snapshot, current_tracker_frame=self.cursor
            )
            if not started:
                return
            active = self.controller.active_status()
            audio_path = active.get("audio_path")
            if audio_path is None:
                return
            self.audio_player.start(
                audio_path,
                command_id=snapshot.command_id,
                tracker_frame=self.cursor,
            )
            latency = max(0.0, float(self.clock()) - pending.armed_perf_time)
            self._emit(
                {
                    "kind": "audio_started_after_gmt_ack",
                    "tracker_frame": self.cursor,
                    "first_motion_tracker_frame": pending.first_motion_tracker_frame,
                    "command_id": snapshot.command_id,
                    "command_revision": snapshot.revision,
                    "stream_id": self.stream_id,
                    "requested_sequence": pending.packet_sequence,
                    "ack_sequence": ack.sequence,
                    "ack_received_unix_ns": ack.received_unix_ns,
                    "ack_latency_seconds": latency,
                }
            )
            print(
                f"[audio-playback] GMT ACK received command_id={snapshot.command_id} "
                f"sequence={ack.sequence} latency={latency * 1000.0:.1f}ms; "
                "starting ffplay",
                flush=True,
            )
            return
        elapsed = max(0.0, float(self.clock()) - pending.armed_perf_time)
        if elapsed < float(self.config.audio_ack_timeout_seconds):
            return
        self._pending_audio_ack = None
        self._emit(
            {
                "kind": "audio_gmt_ack_timeout",
                "tracker_frame": self.cursor,
                "command_id": pending.command_id,
                "command_revision": pending.command_revision,
                "stream_id": self.stream_id,
                "packet_sequence": pending.packet_sequence,
                "timeout_seconds": float(self.config.audio_ack_timeout_seconds),
            }
        )
        print(
            f"[audio-playback] GMT ACK timeout command_id={pending.command_id} "
            f"sequence={pending.packet_sequence}; switching to fixed idle",
            flush=True,
        )
        self.controller.accept_stand()

    def _periodic_status(
        self, tick: BumiMotionTick, snapshot: ConditionSnapshot, redis_queued: bool
    ) -> None:
        now = float(self.clock())
        if now - self._last_status_time < self.config.status_interval_seconds:
            return
        self._last_status_time = now
        self._emit(
            {
                "kind": "bridge_status",
                "tracker_frame": self.cursor,
                "command_id": snapshot.command_id,
                "command_type": snapshot.command_type,
                "command_revision": snapshot.revision,
                "pending_replan": self.pending is not None,
                "active_plan_revision": self.active_plan_revision,
                "output_state": tick.state,
                "output_source": tick.source,
                "plan_remaining_frames": tick.plan_remaining_frames,
                "packet_sequence": self.packet_sequence,
                "audio_source_duration_seconds": (
                    snapshot.audio_source_duration_seconds
                ),
                "audio_effective_duration_seconds": (
                    snapshot.audio_effective_duration_seconds
                ),
                "audio_trailing_silence_seconds": (
                    snapshot.audio_trailing_silence_seconds
                ),
                "audio_start_tracker_frame": snapshot.audio_start_tracker_frame,
                "audio_end_tracker_frame": snapshot.audio_end_tracker_frame,
                "redis_queued": redis_queued,
                "redis": self.publisher.status() if hasattr(self.publisher, "status") else {},
                "audio_playback": self.audio_player.status(),
                "audio_ack_pending": self._pending_audio_ack is not None,
                "redis_ack": (
                    self.ack_reader.status() if self.ack_reader is not None else None
                ),
                "planner_error_count": self._planner_error_count,
                "failed_command_revision": self._failed_command_revision,
                "stale_plan_count": self._stale_plan_count,
                "planner_history": self._history_status(),
            }
        )

    def step(self) -> BumiMotionTick:
        if self._closed:
            raise RuntimeError("BUMI→GMT runtime is closed")
        self._poll_audio_ack()
        audio_end = self.controller.update_for_tracker_frame(self.cursor)
        if audio_end is not None:
            self.audio_player.stop(reason="audio_ended")
            self._emit(audio_end)
        snapshot = self._handle_command_transition()
        self._poll_response()
        snapshot = self.controller.snapshot()
        if self._should_replan(snapshot):
            self._begin_replan()
        tick = self.mux.tick(self.cursor)
        snapshot = self.controller.snapshot()
        self._start_audio_without_ack_if_needed(tick, snapshot)

        prefix = self.trajectory_history.packet_prefix(tick.qpos)
        future = self.mux.preview_future(TRAJECTORY_FRAME_COUNT - prefix.shape[0])
        timeline = np.concatenate((prefix, future), axis=0)
        packet = make_trajectory_packet(
            timeline,
            fps=self.config.tracker_fps,
            native_to_gmt=self.native_to_gmt,
            joint_order_hash=self.joint_order_hash,
            stream_id=self.stream_id,
            sequence=self.packet_sequence,
            command_revision=snapshot.revision,
            plan_id=tick.plan_id,
            flags=self._packet_flags(tick, snapshot),
        )
        redis_queued = bool(self.publisher.publish(packet))
        self._arm_audio_ack_if_needed(
            tick,
            snapshot,
            packet_sequence=packet.sequence,
            redis_queued=redis_queued,
        )
        # A synchronous/fake ACK source used by tests may already expose the
        # acknowledgement for the packet that was just queued.
        self._poll_audio_ack()
        self.packet_sequence += 1
        self._append_history_sample(tick.qpos)

        if self.sim_stream is not None:
            playback = self.audio_player.status()
            self.sim_stream.update(
                tick.qpos,
                frame_index=self.cursor,
                overlay_lines=[
                    (
                        f"command: {snapshot.command_type} "
                        f"id={snapshot.command_id} rev={snapshot.revision}"
                    ),
                    f"planner pending: {self.pending is not None}",
                    f"output: {tick.source} ({tick.state})",
                    f"tracker frame: {self.cursor}",
                    f"plan id: {tick.plan_id} remaining: {tick.plan_remaining_frames}",
                    f"audio: {'playing' if playback['playing'] else 'off'}",
                    f"GMT Redis: {'queued' if redis_queued else 'not connected'}",
                    (
                        f"history: {self.config.history_source} "
                        f"({'ready' if self._lowstate_history_ready() else 'waiting'})"
                    ),
                ],
            )
        self._periodic_status(tick, snapshot, redis_queued)
        self.cursor += 1
        return tick

    def status(self) -> dict[str, Any]:
        snapshot = self.controller.snapshot()
        return {
            "tracker_frame": self.cursor,
            "command_id": snapshot.command_id,
            "command_type": snapshot.command_type,
            "command_revision": snapshot.revision,
            "pending_replan": self.pending is not None,
            "active_plan_revision": self.active_plan_revision,
            "last_plan_id": self.last_plan_id,
            "packet_sequence": self.packet_sequence,
            "audio_source_duration_seconds": (
                snapshot.audio_source_duration_seconds
            ),
            "audio_effective_duration_seconds": (
                snapshot.audio_effective_duration_seconds
            ),
            "audio_trailing_silence_seconds": (
                snapshot.audio_trailing_silence_seconds
            ),
            "audio_start_tracker_frame": snapshot.audio_start_tracker_frame,
            "audio_end_tracker_frame": snapshot.audio_end_tracker_frame,
            "failed_command_revision": self._failed_command_revision,
            "mux": self.mux.status(),
            "audio_playback": self.audio_player.status(),
            "audio_ack_pending": self._pending_audio_ack is not None,
            "redis_ack": (
                self.ack_reader.status() if self.ack_reader is not None else None
            ),
            "publisher": self.publisher.status() if hasattr(self.publisher, "status") else {},
            "planner_history": self._history_status(),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.audio_player.close()
        self.planner_client.close()


def _parse_hybrid_history_beta(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(part.strip()) for part in str(value).split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "hybrid beta must be a comma-separated list of numbers"
        ) from exc
    if not result:
        raise argparse.ArgumentTypeError("hybrid beta must not be empty")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish native OMG BUMI qpos to the GMT trajectory_v1 Redis input."
    )
    parser.add_argument("--connect", default="tcp://127.0.0.1:5571")
    parser.add_argument("--command-bind", default="tcp://127.0.0.1:5581")
    parser.add_argument("--gmt-policy-onnx", required=True)
    parser.add_argument(
        "--kinematics-path", default="assets/robots/bumi/bumi_kinematics.json"
    )
    parser.add_argument("--tracker-fps", type=float, default=50.0)
    parser.add_argument("--history-fps", type=float, default=30.0)
    parser.add_argument("--history-frames", type=int, default=10)
    parser.add_argument(
        "--history-source",
        choices=["reference", "lowstate", "hybrid"],
        default="reference",
        help=(
            "Planner history source. lowstate uses GMT IMU quaternion and measured "
            "joints; hybrid progressively pulls all history frames toward those "
            "measurements. Root xyz always remains reference."
        ),
    )
    parser.add_argument(
        "--hybrid-history-beta",
        type=_parse_hybrid_history_beta,
        default=DEFAULT_HYBRID_HISTORY_BETA,
        help=(
            "Comma-separated oldest-to-newest fusion weights; count must equal "
            "--history-frames and the last value must be 1.0"
        ),
    )
    parser.add_argument(
        "--hybrid-max-rotation-error-deg", type=float, default=25.0
    )
    parser.add_argument(
        "--hybrid-max-tilt-error-deg", type=float, default=20.0
    )
    parser.add_argument(
        "--hybrid-hard-root-tilt-deg", type=float, default=45.0
    )
    parser.add_argument("--planner-frames", type=int, default=60)
    parser.add_argument("--replan-remaining-frames", type=int, default=60)
    parser.add_argument("--blend-seconds", type=float, default=0.2)
    parser.add_argument("--return-seconds", type=float, default=1.0)
    parser.add_argument("--return-max-joint-speed", type=float, default=1.5)
    parser.add_argument("--audio-fps", type=float, default=30.0)
    parser.add_argument("--audio-type", choices=["audio", "feature"], default="audio")
    parser.add_argument("--audio-feature-type", choices=["current35"], default="current35")
    parser.add_argument(
        "--audio-tail-silence-dbfs",
        type=float,
        default=-50.0,
        help="RMS threshold used to detect a continuous silent WAV tail",
    )
    parser.add_argument(
        "--audio-tail-silence-min-seconds",
        type=float,
        default=0.5,
        help="Minimum continuous silent tail to trim",
    )
    parser.add_argument(
        "--audio-tail-analysis-window-ms",
        type=float,
        default=20.0,
        help="RMS analysis window for WAV tail-silence detection",
    )
    parser.add_argument("--condition-audio-step-frames", type=int, default=None)
    parser.add_argument("--play-audio", action="store_true")
    parser.add_argument("--ffplay", default="ffplay")
    parser.add_argument("--timeout-ms", type=int, default=120000)
    parser.add_argument("--redis-host", default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--redis-key", default="gmt_online_frame_bumi")
    parser.add_argument(
        "--redis-ack-key",
        default=None,
        help="GMT trajectory ACK key (default: <redis-key>_ack)",
    )
    parser.add_argument("--redis-ack-poll-ms", type=float, default=5.0)
    parser.add_argument(
        "--redis-lowstate-key",
        default=None,
        help="GMT LowState feedback key (default: <redis-key>_lowstate)",
    )
    parser.add_argument("--redis-lowstate-poll-ms", type=float, default=5.0)
    parser.add_argument("--lowstate-max-age-ms", type=float, default=200.0)
    parser.add_argument("--audio-ack-timeout-ms", type=float, default=2000.0)
    parser.add_argument("--redis-ttl-ms", type=int, default=500)
    parser.add_argument("--sim-stream-bind", default=None)
    parser.add_argument("--sim-stream-fps", type=float, default=20.0)
    parser.add_argument("--sim-stream-width", type=int, default=1280)
    parser.add_argument("--sim-stream-height", type=int, default=720)
    parser.add_argument(
        "--sim-camera-view", choices=["back", "side", "iso", "front"], default="iso"
    )
    parser.add_argument(
        "--sim-follow-mode", choices=["none", "xy", "xyz", "heading"], default="xy"
    )
    parser.add_argument("--sim-camera-distance", type=float, default=4.5)
    parser.add_argument("--sim-camera-elevation", type=float, default=-18.0)
    parser.add_argument("--status-jsonl", default=None)
    parser.add_argument("--status-interval-seconds", type=float, default=1.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--num-frames", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.continuous and int(args.num_frames) <= 0:
        raise ValueError("Use --continuous or provide a positive --num-frames")
    contract = GmtPolicyContract.from_onnx(args.gmt_policy_onnx)
    idle_qpos, kinematics, native_to_gmt = build_policy_default_idle_qpos(
        contract, kinematics_path=args.kinematics_path
    )
    audio_step = (
        compute_condition_audio_step_frames(
            planner_frames=args.planner_frames,
            history_fps=args.history_fps,
            tracker_fps=args.tracker_fps,
            replan_remaining_frames=args.replan_remaining_frames,
            audio_fps=args.audio_fps,
        )
        if args.condition_audio_step_frames is None
        else int(args.condition_audio_step_frames)
    )
    config = BumiGmtRuntimeConfig(
        tracker_fps=args.tracker_fps,
        history_fps=args.history_fps,
        history_frames=args.history_frames,
        planner_frames=args.planner_frames,
        replan_remaining_frames=args.replan_remaining_frames,
        audio_fps=args.audio_fps,
        audio_type=args.audio_type,
        audio_feature_type=args.audio_feature_type,
        condition_audio_step_frames=audio_step,
        request_timeout_ms=args.timeout_ms,
        audio_ack_timeout_seconds=float(args.audio_ack_timeout_ms) / 1000.0,
        status_interval_seconds=args.status_interval_seconds,
        history_source=args.history_source,
        lowstate_max_age_seconds=float(args.lowstate_max_age_ms) / 1000.0,
        hybrid_history_beta=tuple(args.hybrid_history_beta),
        hybrid_max_rotation_error_degrees=args.hybrid_max_rotation_error_deg,
        hybrid_max_tilt_error_degrees=args.hybrid_max_tilt_error_deg,
        hybrid_hard_root_tilt_degrees=args.hybrid_hard_root_tilt_deg,
    )
    controller = DynamicConditionController(
        tracker_fps=args.tracker_fps,
        initial_condition_sequence="text: stand still",
        audio_tail_silence_threshold_dbfs=args.audio_tail_silence_dbfs,
        audio_tail_silence_min_seconds=args.audio_tail_silence_min_seconds,
        audio_tail_analysis_window_seconds=(
            float(args.audio_tail_analysis_window_ms) / 1000.0
        ),
    )
    mux = BumiMotionSourceMux(
        idle_qpos,
        BumiMotionMuxConfig(
            tracker_fps=args.tracker_fps,
            blend_seconds=args.blend_seconds,
            return_seconds=args.return_seconds,
            return_max_joint_speed_radians=args.return_max_joint_speed,
        ),
    )
    initial = np.repeat(idle_qpos[None, :], max(2, args.history_frames), axis=0)
    planner_history = BumiTrackerExecutionHistory(
        initial,
        tracker_fps=args.tracker_fps,
        history_fps=args.history_fps,
        history_frames=args.history_frames,
    )
    lowstate_history = (
        BumiTrackerExecutionHistory(
            initial,
            tracker_fps=args.tracker_fps,
            history_fps=args.history_fps,
            history_frames=args.history_frames,
        )
        if args.history_source in {"lowstate", "hybrid"}
        else None
    )
    trajectory_history = BumiTrajectoryHistory(idle_qpos)
    planner = ZmqPlanClient(args.connect)
    sync_publisher = RedisTrajectoryPublisher(
        RedisTrajectoryPublisherConfig(
            host=args.redis_host,
            port=args.redis_port,
            db=args.redis_db,
            key=args.redis_key,
            ttl_ms=args.redis_ttl_ms,
        )
    )
    publisher = AsyncRedisTrajectoryPublisher(sync_publisher)
    ack_key = args.redis_ack_key or f"{args.redis_key}_ack"
    ack_reader = (
        RedisTrajectoryAckReader(
            RedisTrajectoryAckReaderConfig(
                host=args.redis_host,
                port=args.redis_port,
                db=args.redis_db,
                key=ack_key,
                poll_interval_seconds=float(args.redis_ack_poll_ms) / 1000.0,
            )
        )
        if args.play_audio
        else None
    )
    lowstate_key = args.redis_lowstate_key or f"{args.redis_key}_lowstate"
    lowstate_reader = (
        RedisLowStateFeedbackReader(
            RedisLowStateFeedbackReaderConfig(
                host=args.redis_host,
                port=args.redis_port,
                db=args.redis_db,
                key=lowstate_key,
                poll_interval_seconds=(
                    float(args.redis_lowstate_poll_ms) / 1000.0
                ),
            )
        )
        if args.history_source in {"lowstate", "hybrid"}
        else None
    )
    audio_player = SynchronizedAudioPlayer(
        enabled=args.play_audio, executable=args.ffplay
    )
    status_callback = lambda event: append_jsonl(args.status_jsonl, event)
    sim_stream = None
    runtime: BumiGmtRuntime | None = None
    command_server: DynamicCommandServer | None = None
    executed: list[np.ndarray] = []
    try:
        publisher.start()
        if ack_reader is not None:
            ack_reader.start()
        if lowstate_reader is not None:
            lowstate_reader.start()
        if args.sim_stream_bind is not None:
            from omg.realtime.sim_stream import SimStreamConfig, SimStreamServer

            sim_stream = SimStreamServer(
                SimStreamConfig(
                    bind=args.sim_stream_bind,
                    fps=args.sim_stream_fps,
                    width=args.sim_stream_width,
                    height=args.sim_stream_height,
                    camera_view=args.sim_camera_view,
                    follow_mode=args.sim_follow_mode,
                    camera_distance=args.sim_camera_distance,
                    camera_elevation=args.sim_camera_elevation,
                    robot_name="bumi",
                    state_dim=BUMI_QPOS_DIM,
                    kinematics_path=args.kinematics_path,
                    mjcf_path="assets/robots/bumi/bumi3.xml",
                )
            )
            sim_stream.start()
        runtime = BumiGmtRuntime(
            config=config,
            controller=controller,
            planner_client=planner,
            mux=mux,
            planner_history=planner_history,
            trajectory_history=trajectory_history,
            publisher=publisher,
            native_to_gmt=native_to_gmt,
            joint_order_hash=contract.joint_order_hash,
            audio_player=audio_player,
            ack_reader=ack_reader,
            lowstate_history=lowstate_history,
            lowstate_reader=lowstate_reader,
            sim_stream=sim_stream,
            status_callback=status_callback,
            planner_client_factory=lambda: ZmqPlanClient(args.connect),
        )
        command_server = DynamicCommandServer(
            CommandServerConfig(bind=args.command_bind),
            controller,
            status_callback=status_callback,
            status_provider=runtime.status,
        )
        command_server.start()
        print(
            f"[BUMI→GMT] fixed idle root_z={idle_qpos[2]:.4f} "
            f"left_shoulder_roll={idle_qpos[7 + kinematics.joint_order.index('l_arm_roll_joint')]:.3f} "
            f"right_shoulder_roll={idle_qpos[7 + kinematics.joint_order.index('r_arm_roll_joint')]:.3f}",
            flush=True,
        )
        print(
            "[BUMI→GMT] startup/stand/audio-end use fixed qpos; no stand prompt is sent to Planner",
            flush=True,
        )
        print(
            f"[BUMI→GMT] Redis={args.redis_host}:{args.redis_port}/{args.redis_db} "
            f"key={args.redis_key} ack_key={ack_key} "
            f"protocol=trajectory_v1 audio_step={audio_step}",
            flush=True,
        )
        print(
            f"[BUMI→GMT] planner history source={args.history_source} "
            + (
                f"lowstate_key={lowstate_key} max_age={args.lowstate_max_age_ms:g}ms; "
                "root xyz remains reference"
                if args.history_source in {"lowstate", "hybrid"}
                else "(emitted reference trajectory)"
            ),
            flush=True,
        )
        period = 1.0 / args.tracker_fps
        deadline = time.perf_counter()
        while args.continuous or runtime.cursor < args.num_frames:
            tick = runtime.step()
            if args.output is not None:
                executed.append(tick.qpos.copy())
            deadline += period
            remaining = deadline - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
            elif remaining < -period:
                deadline = time.perf_counter()
    except KeyboardInterrupt:
        print("[BUMI→GMT] interrupted", flush=True)
    finally:
        close_errors: list[BaseException] = []
        for close in (
            command_server.close if command_server is not None else None,
            runtime.close if runtime is not None else planner.close,
            ack_reader.close if ack_reader is not None else None,
            lowstate_reader.close if lowstate_reader is not None else None,
            publisher.close,
            sim_stream.close if sim_stream is not None else None,
        ):
            if close is None:
                continue
            try:
                close()
            except BaseException as exc:
                close_errors.append(exc)
        if args.output is not None:
            output = Path(args.output).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            qpos = (
                np.stack(executed).astype(np.float32)
                if executed
                else np.zeros((0, BUMI_QPOS_DIM), dtype=np.float32)
            )
            np.savez_compressed(
                output,
                qpos=qpos,
                executed_qpos=qpos,
                fps=np.float32(args.tracker_fps),
                robot_name=np.asarray("bumi"),
                joint_names=np.asarray(kinematics.joint_order),
                quaternion_convention=np.asarray("wxyz"),
                history_source=np.asarray(args.history_source),
                hybrid_history_beta=np.asarray(
                    args.hybrid_history_beta, dtype=np.float32
                ),
                hybrid_max_rotation_error_degrees=np.float32(
                    args.hybrid_max_rotation_error_deg
                ),
                hybrid_max_tilt_error_degrees=np.float32(
                    args.hybrid_max_tilt_error_deg
                ),
                hybrid_hard_root_tilt_degrees=np.float32(
                    args.hybrid_hard_root_tilt_deg
                ),
            )
        if close_errors:
            raise RuntimeError(
                "Errors while closing BUMI→GMT bridge: "
                + "; ".join(str(error) for error in close_errors)
            )


if __name__ == "__main__":
    main()
