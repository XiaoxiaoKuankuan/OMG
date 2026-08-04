from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from omg.tracking.holomotion.reference import normalize_quat_wxyz, resample_qpos


QPOS_DIM = 36


def _positive_finite(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}")
    return result


def _non_negative_finite(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value}")
    return result


def coerce_g1_qpos_frame(value: Any, *, name: str = "qpos_36") -> np.ndarray:
    frame = np.asarray(value, dtype=np.float32).reshape(-1)
    if frame.shape != (QPOS_DIM,):
        raise ValueError(f"{name} must contain {QPOS_DIM} values, got {frame.shape}")
    if not np.isfinite(frame).all():
        raise ValueError(f"{name} contains non-finite values")
    result = frame.astype(np.float32, copy=True)
    result[3:7] = normalize_quat_wxyz(result[3:7])
    return result


def coerce_g1_qpos_motion(value: Any, *, name: str = "qpos_36") -> np.ndarray:
    motion = np.asarray(value, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] != QPOS_DIM or motion.shape[0] <= 0:
        raise ValueError(f"{name} must have shape (T,{QPOS_DIM}) with T > 0, got {motion.shape}")
    if not np.isfinite(motion).all():
        raise ValueError(f"{name} contains non-finite values")
    result = motion.astype(np.float32, copy=True)
    result[:, 3:7] = np.stack(
        [normalize_quat_wxyz(quat) for quat in result[:, 3:7]],
        axis=0,
    )
    return result


def smoothstep(alpha: float) -> float:
    value = float(np.clip(float(alpha), 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def slerp_wxyz(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    q0 = normalize_quat_wxyz(first).astype(np.float64)
    q1 = normalize_quat_wxyz(second).astype(np.float64)
    amount = float(np.clip(float(alpha), 0.0, 1.0))
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = q0 + amount * (q1 - q0)
    else:
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        result = (
            math.sin((1.0 - amount) * theta) / sin_theta * q0
            + math.sin(amount * theta) / sin_theta * q1
        )
    return normalize_quat_wxyz(result.astype(np.float32))


def multiply_quat_wxyz(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    q0 = normalize_quat_wxyz(first).astype(np.float64)
    q1 = normalize_quat_wxyz(second).astype(np.float64)
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    result = np.asarray(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ],
        dtype=np.float32,
    )
    return normalize_quat_wxyz(result)


def yaw_radians_from_wxyz(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = normalize_quat_wxyz(quat_wxyz).astype(np.float64)
    return float(
        math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
    )


def quat_with_preserved_heading_wxyz(
    canonical_quat_wxyz: np.ndarray,
    current_quat_wxyz: np.ndarray,
) -> np.ndarray:
    """Use canonical roll/pitch while retaining the current world yaw."""

    canonical = normalize_quat_wxyz(canonical_quat_wxyz).astype(np.float32)
    canonical_yaw = yaw_radians_from_wxyz(canonical)
    canonical_heading_inverse = np.asarray(
        [
            math.cos(-0.5 * canonical_yaw),
            0.0,
            0.0,
            math.sin(-0.5 * canonical_yaw),
        ],
        dtype=np.float32,
    )
    canonical_tilt = multiply_quat_wxyz(
        canonical_heading_inverse,
        canonical,
    )
    current_yaw = yaw_radians_from_wxyz(current_quat_wxyz)
    current_heading = np.asarray(
        [
            math.cos(0.5 * current_yaw),
            0.0,
            0.0,
            math.sin(0.5 * current_yaw),
        ],
        dtype=np.float32,
    )
    return multiply_quat_wxyz(current_heading, canonical_tilt)


def interpolate_g1_qpos(
    first: np.ndarray,
    second: np.ndarray,
    alpha: float,
    *,
    easing: bool = True,
) -> np.ndarray:
    q0 = coerce_g1_qpos_frame(first, name="first")
    q1 = coerce_g1_qpos_frame(second, name="second")
    amount = smoothstep(alpha) if easing else float(np.clip(float(alpha), 0.0, 1.0))
    result = q0 + np.float32(amount) * (q1 - q0)
    result[3:7] = slerp_wxyz(q0[3:7], q1[3:7], amount)
    return result.astype(np.float32, copy=False)


class G1MotionState(str, Enum):
    IDLE = "IDLE"
    WAITING_PLAN = "WAITING_PLAN"
    BLENDING = "BLENDING"
    GENERATED = "GENERATED"
    RETURNING = "RETURNING"
    ERROR_IDLE = "ERROR_IDLE"


@dataclass(frozen=True)
class G1MotionMuxConfig:
    tracker_fps: float = 50.0
    blend_seconds: float = 0.20
    return_seconds: float = 0.50
    preserve_idle_xy: bool = True
    preserve_idle_heading: bool = True
    return_max_joint_speed_radians: float | None = None

    def __post_init__(self) -> None:
        _positive_finite(self.tracker_fps, name="tracker_fps")
        _non_negative_finite(self.blend_seconds, name="blend_seconds")
        _non_negative_finite(self.return_seconds, name="return_seconds")
        if self.return_max_joint_speed_radians is not None:
            _positive_finite(
                self.return_max_joint_speed_radians,
                name="return_max_joint_speed_radians",
            )


@dataclass(frozen=True)
class G1MotionTick:
    frame_index: int
    qpos_36: np.ndarray
    state: str
    source: str
    plan_remaining_frames: int
    underflow_count: int


class G1MotionSourceMux:
    """Select exactly one continuous G1 qpos source for every tracker tick.

    The mux owns only motion selection; network publishing and planner requests
    remain outside it.  An idle command never requires a generated plan.
    """

    def __init__(self, idle_qpos_36: np.ndarray, config: G1MotionMuxConfig | None = None) -> None:
        self.config = config or G1MotionMuxConfig()
        self._canonical_idle = coerce_g1_qpos_frame(idle_qpos_36, name="idle_qpos_36")
        self._idle_target = self._canonical_idle.copy()
        self._last_output = self._canonical_idle.copy()
        self._state = G1MotionState.IDLE
        self._plan = np.zeros((0, QPOS_DIM), dtype=np.float32)
        self._plan_cursor = 0
        self._transition_from = self._last_output.copy()
        self._transition_index = 0
        self._transition_frames = 0
        self._return_reason = "stand"
        self._active_command_id: str | None = None
        self._active_command_revision: int | None = None
        self._active_plan_id: int | None = None
        self._underflow_count = 0
        self._accepted_plan_count = 0
        self._expired_plan_count = 0

    @property
    def state(self) -> G1MotionState:
        return self._state

    @property
    def last_output(self) -> np.ndarray:
        return self._last_output.copy()

    @property
    def idle_qpos_36(self) -> np.ndarray:
        return self._idle_target.copy()

    @property
    def plan_remaining_frames(self) -> int:
        return max(0, int(self._plan.shape[0]) - int(self._plan_cursor))

    @property
    def underflow_count(self) -> int:
        return int(self._underflow_count)

    def mark_waiting_for_plan(self) -> None:
        if self._state in {G1MotionState.IDLE, G1MotionState.WAITING_PLAN}:
            self._state = G1MotionState.WAITING_PLAN

    def accept_plan(
        self,
        qpos_36: np.ndarray,
        *,
        source_fps: float,
        skip_tracker_frames: int = 0,
        plan_id: int | None = None,
        command_id: str | None = None,
        command_revision: int | None = None,
    ) -> bool:
        source = coerce_g1_qpos_motion(qpos_36)
        source_fps = _positive_finite(source_fps, name="source_fps")
        skip = int(skip_tracker_frames)
        if skip < 0:
            raise ValueError(f"skip_tracker_frames must be non-negative, got {skip_tracker_frames}")
        plan = resample_qpos(
            source,
            source_fps=source_fps,
            target_fps=float(self.config.tracker_fps),
        )
        if skip >= int(plan.shape[0]):
            self._expired_plan_count += 1
            return False
        self._plan = plan[skip:].astype(np.float32, copy=True)
        self._plan_cursor = 0
        self._transition_from = self._last_output.copy()
        self._transition_index = 0
        self._transition_frames = int(round(self.config.blend_seconds * self.config.tracker_fps))
        self._state = (
            G1MotionState.BLENDING
            if self._transition_frames > 0
            else G1MotionState.GENERATED
        )
        self._active_plan_id = None if plan_id is None else int(plan_id)
        self._active_command_id = command_id
        self._active_command_revision = (
            None if command_revision is None else int(command_revision)
        )
        self._accepted_plan_count += 1
        return True

    def _current_idle_target(self) -> np.ndarray:
        target = self._canonical_idle.copy()
        if self.config.preserve_idle_xy:
            target[:2] = self._last_output[:2]
        if self.config.preserve_idle_heading:
            target[3:7] = quat_with_preserved_heading_wxyz(
                self._canonical_idle[3:7],
                self._last_output[3:7],
            )
        return target

    def _return_transition_frames(self, target: np.ndarray) -> int:
        seconds = float(self.config.return_seconds)
        max_speed = self.config.return_max_joint_speed_radians
        if max_speed is not None:
            maximum_joint_delta = float(
                np.max(np.abs(target[7:] - self._last_output[7:]))
            )
            # smoothstep's peak derivative is 1.5, so include that factor to
            # make the configured value a peak rate rather than an average.
            seconds = max(
                seconds,
                1.5 * maximum_joint_delta / float(max_speed),
            )
        return int(math.ceil(seconds * float(self.config.tracker_fps)))

    def return_to_idle(self, *, reason: str = "stand", error: bool = False) -> None:
        self._idle_target = self._current_idle_target()
        self._transition_from = self._last_output.copy()
        self._transition_index = 0
        self._transition_frames = self._return_transition_frames(self._idle_target)
        self._return_reason = str(reason)
        self._plan = np.zeros((0, QPOS_DIM), dtype=np.float32)
        self._plan_cursor = 0
        self._active_plan_id = None
        self._active_command_id = None
        self._active_command_revision = None
        if self._transition_frames <= 0:
            self._last_output = self._idle_target.copy()
            self._state = G1MotionState.IDLE
        else:
            self._state = G1MotionState.ERROR_IDLE if error else G1MotionState.RETURNING

    def _next_plan_target(self) -> np.ndarray | None:
        if self._plan_cursor >= int(self._plan.shape[0]):
            return None
        target = self._plan[self._plan_cursor].copy()
        self._plan_cursor += 1
        return target

    def _return_tick(self) -> tuple[np.ndarray, str]:
        if self._transition_frames <= 0:
            self._state = G1MotionState.IDLE
            return self._idle_target.copy(), "idle"
        self._transition_index += 1
        alpha = self._transition_index / self._transition_frames
        frame = interpolate_g1_qpos(self._transition_from, self._idle_target, alpha)
        source = "error_return" if self._state == G1MotionState.ERROR_IDLE else "return_to_idle"
        if self._transition_index >= self._transition_frames:
            frame = self._idle_target.copy()
            self._state = G1MotionState.IDLE
        return frame, source

    def tick(self, frame_index: int) -> G1MotionTick:
        index = int(frame_index)
        if index < 0:
            raise ValueError(f"frame_index must be non-negative, got {frame_index}")
        source = "idle"
        if self._state in {G1MotionState.IDLE, G1MotionState.WAITING_PLAN}:
            frame = self._idle_target.copy()
            source = "waiting_plan" if self._state == G1MotionState.WAITING_PLAN else "idle"
        elif self._state == G1MotionState.BLENDING:
            target = self._next_plan_target()
            if target is None:
                self._underflow_count += 1
                self.return_to_idle(reason="plan_underflow", error=True)
                frame, source = self._return_tick()
            else:
                self._transition_index += 1
                alpha = self._transition_index / max(1, self._transition_frames)
                frame = interpolate_g1_qpos(self._transition_from, target, alpha)
                source = "generated_blend"
                if self._transition_index >= self._transition_frames:
                    self._state = G1MotionState.GENERATED
        elif self._state == G1MotionState.GENERATED:
            target = self._next_plan_target()
            if target is None:
                self._underflow_count += 1
                self.return_to_idle(reason="plan_underflow", error=True)
                frame, source = self._return_tick()
            else:
                frame = target
                source = "generated"
        else:
            frame, source = self._return_tick()
        self._last_output = coerce_g1_qpos_frame(frame)
        return G1MotionTick(
            frame_index=index,
            qpos_36=self._last_output.copy(),
            state=self._state.value,
            source=source,
            plan_remaining_frames=self.plan_remaining_frames,
            underflow_count=int(self._underflow_count),
        )

    def status(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "plan_remaining_frames": self.plan_remaining_frames,
            "active_plan_id": self._active_plan_id,
            "active_command_id": self._active_command_id,
            "active_command_revision": self._active_command_revision,
            "accepted_plan_count": int(self._accepted_plan_count),
            "expired_plan_count": int(self._expired_plan_count),
            "underflow_count": int(self._underflow_count),
            "return_reason": self._return_reason,
            "preserve_idle_xy": bool(self.config.preserve_idle_xy),
            "preserve_idle_heading": bool(self.config.preserve_idle_heading),
            "return_transition_frames": int(self._transition_frames),
            "return_max_joint_speed_radians": (
                None
                if self.config.return_max_joint_speed_radians is None
                else float(self.config.return_max_joint_speed_radians)
            ),
        }


class TrackerExecutionHistory:
    """Keep actual tracker output and construct a correctly timed planner history."""

    def __init__(
        self,
        initial_qpos_36: np.ndarray,
        *,
        tracker_fps: float,
        history_fps: float,
        history_frames: int,
    ) -> None:
        self.tracker_fps = _positive_finite(tracker_fps, name="tracker_fps")
        self.history_fps = _positive_finite(history_fps, name="history_fps")
        self.history_frames = int(history_frames)
        if self.history_frames <= 0:
            raise ValueError(f"history_frames must be positive, got {history_frames}")
        required = int(
            math.ceil(
                max(0, self.history_frames - 1)
                * self.tracker_fps
                / self.history_fps
            )
        ) + 1
        self._max_tracker_frames = max(2, required + 2)
        initial = coerce_g1_qpos_motion(initial_qpos_36, name="initial_qpos_36")
        self._frames = [frame.copy() for frame in initial[-self._max_tracker_frames :]]
        if len(self._frames) < self._max_tracker_frames:
            pad = [self._frames[0].copy() for _ in range(self._max_tracker_frames - len(self._frames))]
            self._frames = pad + self._frames

    @property
    def tracker_frames(self) -> int:
        return len(self._frames)

    def append(self, qpos_36: np.ndarray) -> None:
        self._frames.append(coerce_g1_qpos_frame(qpos_36))
        if len(self._frames) > self._max_tracker_frames:
            del self._frames[: len(self._frames) - self._max_tracker_frames]

    def planner_history(self) -> np.ndarray:
        source = np.stack(self._frames, axis=0).astype(np.float32, copy=False)
        history = resample_qpos(
            source,
            source_fps=self.tracker_fps,
            target_fps=self.history_fps,
        )
        if int(history.shape[0]) < self.history_frames:
            padding = np.repeat(
                history[:1],
                self.history_frames - int(history.shape[0]),
                axis=0,
            )
            history = np.concatenate([padding, history], axis=0)
        result = history[-self.history_frames :].astype(np.float32, copy=True)
        # The planner seed must end at the pose that was actually emitted on
        # the latest tracker tick, even when the two FPS grids do not align.
        result[-1] = source[-1]
        return result
