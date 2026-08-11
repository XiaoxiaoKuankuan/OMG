from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from omg.realtime.g1_motion_mux import (
    quat_with_preserved_heading_wxyz,
    slerp_wxyz,
    smoothstep,
)
from omg.tracking.holomotion.reference import normalize_quat_wxyz


BUMI_QPOS_DIM = 28


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


def coerce_bumi_qpos_frame(value: Any, *, name: str = "qpos") -> np.ndarray:
    frame = np.asarray(value, dtype=np.float32).reshape(-1)
    if frame.shape != (BUMI_QPOS_DIM,):
        raise ValueError(f"{name} must contain {BUMI_QPOS_DIM} values, got {frame.shape}")
    if not np.isfinite(frame).all():
        raise ValueError(f"{name} contains non-finite values")
    result = frame.astype(np.float32, copy=True)
    result[3:7] = normalize_quat_wxyz(result[3:7])
    return result


def coerce_bumi_qpos_motion(value: Any, *, name: str = "qpos") -> np.ndarray:
    motion = np.asarray(value, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] != BUMI_QPOS_DIM or motion.shape[0] <= 0:
        raise ValueError(
            f"{name} must have shape (T,{BUMI_QPOS_DIM}) with T > 0, got {motion.shape}"
        )
    if not np.isfinite(motion).all():
        raise ValueError(f"{name} contains non-finite values")
    result = motion.astype(np.float32, copy=True)
    result[:, 3:7] = np.stack(
        [normalize_quat_wxyz(quat) for quat in result[:, 3:7]], axis=0
    )
    for index in range(1, result.shape[0]):
        if float(np.dot(result[index - 1, 3:7], result[index, 3:7])) < 0.0:
            result[index, 3:7] *= -1.0
    return result


def resample_bumi_plan(
    qpos: np.ndarray,
    *,
    source_fps: float,
    target_fps: float,
) -> np.ndarray:
    """Resample a frame-count-defined plan while preserving its nominal duration.

    OMG exports 60 frames at 30 Hz as a two-second plan.  The generic legacy
    resampler treats the last sample time as the duration and consequently
    returns 99 frames at 50 Hz.  The realtime GMT contract is frame-count
    based, so this helper deliberately returns round(T * target/source) frames.
    """

    source = coerce_bumi_qpos_motion(qpos)
    source_rate = _positive_finite(source_fps, name="source_fps")
    target_rate = _positive_finite(target_fps, name="target_fps")
    target_frames = max(1, int(round(source.shape[0] * target_rate / source_rate)))
    if source.shape[0] == 1:
        return np.repeat(source, target_frames, axis=0)
    source_t = np.arange(source.shape[0], dtype=np.float64) / source_rate
    target_t = np.arange(target_frames, dtype=np.float64) / target_rate
    target_t = np.clip(target_t, source_t[0], source_t[-1])
    result = np.empty((target_frames, BUMI_QPOS_DIM), dtype=np.float32)
    for dim in (*range(3), *range(7, BUMI_QPOS_DIM)):
        result[:, dim] = np.interp(target_t, source_t, source[:, dim]).astype(np.float32)
    quats = np.stack([normalize_quat_wxyz(value) for value in source[:, 3:7]], axis=0)
    slerp = Slerp(source_t, Rotation.from_quat(quats[:, [1, 2, 3, 0]]))
    xyzw = slerp(target_t).as_quat().astype(np.float32)
    result[:, 3:7] = xyzw[:, [3, 0, 1, 2]]
    return coerce_bumi_qpos_motion(result)


def interpolate_bumi_qpos(
    first: np.ndarray,
    second: np.ndarray,
    alpha: float,
    *,
    easing: bool = True,
) -> np.ndarray:
    q0 = coerce_bumi_qpos_frame(first, name="first")
    q1 = coerce_bumi_qpos_frame(second, name="second")
    amount = smoothstep(alpha) if easing else float(np.clip(alpha, 0.0, 1.0))
    result = q0 + np.float32(amount) * (q1 - q0)
    result[3:7] = slerp_wxyz(q0[3:7], q1[3:7], amount)
    return result.astype(np.float32, copy=False)


class BumiMotionState(str, Enum):
    FIXED_IDLE = "FIXED_IDLE"
    WAITING_PLAN = "WAITING_PLAN"
    BLENDING = "BLENDING"
    GENERATED = "GENERATED"
    RETURN_TO_IDLE = "RETURN_TO_IDLE"
    ERROR_IDLE = "ERROR_IDLE"


@dataclass(frozen=True)
class BumiMotionMuxConfig:
    tracker_fps: float = 50.0
    blend_seconds: float = 0.2
    return_seconds: float = 1.0
    preserve_idle_xy: bool = True
    preserve_idle_heading: bool = True
    return_max_joint_speed_radians: float = 1.5

    def __post_init__(self) -> None:
        _positive_finite(self.tracker_fps, name="tracker_fps")
        _non_negative_finite(self.blend_seconds, name="blend_seconds")
        _non_negative_finite(self.return_seconds, name="return_seconds")
        _positive_finite(
            self.return_max_joint_speed_radians,
            name="return_max_joint_speed_radians",
        )


@dataclass(frozen=True)
class BumiMotionTick:
    frame_index: int
    qpos: np.ndarray
    state: str
    source: str
    plan_remaining_frames: int
    plan_id: int | None
    command_revision: int | None
    underflow_count: int


class BumiMotionSourceMux:
    def __init__(
        self,
        idle_qpos: np.ndarray,
        config: BumiMotionMuxConfig | None = None,
    ) -> None:
        self.config = config or BumiMotionMuxConfig()
        self._canonical_idle = coerce_bumi_qpos_frame(idle_qpos, name="idle_qpos")
        self._idle_target = self._canonical_idle.copy()
        self._last_output = self._canonical_idle.copy()
        self._state = BumiMotionState.FIXED_IDLE
        self._plan = np.zeros((0, BUMI_QPOS_DIM), dtype=np.float32)
        self._plan_cursor = 0
        self._transition_from = self._last_output.copy()
        self._transition_index = 0
        self._transition_frames = 0
        self._return_reason = "startup"
        self._active_plan_id: int | None = None
        self._active_command_id: str | None = None
        self._active_command_revision: int | None = None
        self._underflow_count = 0
        self._accepted_plan_count = 0
        self._expired_plan_count = 0

    @property
    def state(self) -> BumiMotionState:
        return self._state

    @property
    def last_output(self) -> np.ndarray:
        return self._last_output.copy()

    @property
    def idle_qpos(self) -> np.ndarray:
        return self._idle_target.copy()

    @property
    def plan_remaining_frames(self) -> int:
        return max(0, int(self._plan.shape[0]) - int(self._plan_cursor))

    @property
    def active_plan_id(self) -> int | None:
        return self._active_plan_id

    @property
    def active_command_revision(self) -> int | None:
        return self._active_command_revision

    def mark_waiting_for_plan(self) -> None:
        if self._state in {BumiMotionState.FIXED_IDLE, BumiMotionState.WAITING_PLAN}:
            self._state = BumiMotionState.WAITING_PLAN

    def accept_plan(
        self,
        qpos: np.ndarray,
        *,
        source_fps: float,
        skip_tracker_frames: int = 0,
        plan_id: int | None = None,
        command_id: str | None = None,
        command_revision: int | None = None,
    ) -> bool:
        plan = resample_bumi_plan(
            qpos,
            source_fps=source_fps,
            target_fps=float(self.config.tracker_fps),
        )
        skip = int(skip_tracker_frames)
        if skip < 0:
            raise ValueError("skip_tracker_frames must be non-negative")
        if skip >= int(plan.shape[0]):
            self._expired_plan_count += 1
            return False
        self._plan = plan[skip:].copy()
        self._plan_cursor = 0
        self._transition_from = self._last_output.copy()
        self._transition_index = 0
        self._transition_frames = int(round(self.config.blend_seconds * self.config.tracker_fps))
        self._state = (
            BumiMotionState.BLENDING
            if self._transition_frames > 0
            else BumiMotionState.GENERATED
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
                self._canonical_idle[3:7], self._last_output[3:7]
            )
        return target

    def _return_frame_count(self, target: np.ndarray) -> int:
        maximum_joint_delta = float(np.max(np.abs(target[7:] - self._last_output[7:])))
        seconds = max(
            float(self.config.return_seconds),
            1.5 * maximum_joint_delta / float(self.config.return_max_joint_speed_radians),
        )
        return int(math.ceil(seconds * float(self.config.tracker_fps)))

    def return_to_idle(self, *, reason: str = "stand", error: bool = False) -> None:
        self._idle_target = self._current_idle_target()
        self._transition_from = self._last_output.copy()
        self._transition_index = 0
        self._transition_frames = self._return_frame_count(self._idle_target)
        self._return_reason = str(reason)
        self._plan = np.zeros((0, BUMI_QPOS_DIM), dtype=np.float32)
        self._plan_cursor = 0
        self._active_plan_id = None
        self._active_command_id = None
        self._active_command_revision = None
        if self._transition_frames <= 0:
            self._last_output = self._idle_target.copy()
            self._state = BumiMotionState.FIXED_IDLE
        else:
            self._state = (
                BumiMotionState.ERROR_IDLE if error else BumiMotionState.RETURN_TO_IDLE
            )

    def _next_plan_target(self) -> np.ndarray | None:
        if self._plan_cursor >= self._plan.shape[0]:
            return None
        frame = self._plan[self._plan_cursor].copy()
        self._plan_cursor += 1
        return frame

    def _return_tick(self) -> tuple[np.ndarray, str]:
        if self._transition_frames <= 0:
            self._state = BumiMotionState.FIXED_IDLE
            return self._idle_target.copy(), "fixed_idle"
        self._transition_index += 1
        alpha = self._transition_index / self._transition_frames
        frame = interpolate_bumi_qpos(self._transition_from, self._idle_target, alpha)
        source = "error_return" if self._state == BumiMotionState.ERROR_IDLE else "return_to_idle"
        if self._transition_index >= self._transition_frames:
            frame = self._idle_target.copy()
            self._state = BumiMotionState.FIXED_IDLE
        return frame, source

    def tick(self, frame_index: int) -> BumiMotionTick:
        index = int(frame_index)
        if index < 0:
            raise ValueError("frame_index must be non-negative")
        if self._state in {BumiMotionState.FIXED_IDLE, BumiMotionState.WAITING_PLAN}:
            frame = self._idle_target.copy()
            source = (
                "waiting_plan"
                if self._state == BumiMotionState.WAITING_PLAN
                else "fixed_idle"
            )
        elif self._state == BumiMotionState.BLENDING:
            target = self._next_plan_target()
            if target is None:
                self._underflow_count += 1
                self.return_to_idle(reason="plan_underflow", error=True)
                frame, source = self._return_tick()
            else:
                self._transition_index += 1
                alpha = self._transition_index / max(1, self._transition_frames)
                frame = interpolate_bumi_qpos(self._transition_from, target, alpha)
                source = "generated_blend"
                if self._transition_index >= self._transition_frames:
                    self._state = BumiMotionState.GENERATED
        elif self._state == BumiMotionState.GENERATED:
            target = self._next_plan_target()
            if target is None:
                self._underflow_count += 1
                self.return_to_idle(reason="plan_underflow", error=True)
                frame, source = self._return_tick()
            else:
                frame, source = target, "generated"
        else:
            frame, source = self._return_tick()
        self._last_output = coerce_bumi_qpos_frame(frame)
        return BumiMotionTick(
            frame_index=index,
            qpos=self._last_output.copy(),
            state=self._state.value,
            source=source,
            plan_remaining_frames=self.plan_remaining_frames,
            plan_id=self._active_plan_id,
            command_revision=self._active_command_revision,
            underflow_count=int(self._underflow_count),
        )

    def preview_future(self, frame_count: int) -> np.ndarray:
        count = int(frame_count)
        if count < 0:
            raise ValueError("frame_count must be non-negative")
        if count == 0:
            return np.zeros((0, BUMI_QPOS_DIM), dtype=np.float32)
        clone = copy.deepcopy(self)
        frames = [clone.tick(index).qpos for index in range(count)]
        return np.stack(frames, axis=0).astype(np.float32, copy=False)

    def status(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "plan_remaining_frames": self.plan_remaining_frames,
            "active_plan_id": self._active_plan_id,
            "active_command_id": self._active_command_id,
            "active_command_revision": self._active_command_revision,
            "accepted_plan_count": self._accepted_plan_count,
            "expired_plan_count": self._expired_plan_count,
            "underflow_count": self._underflow_count,
            "return_reason": self._return_reason,
        }


class BumiTrackerExecutionHistory:
    def __init__(
        self,
        initial_qpos: np.ndarray,
        *,
        tracker_fps: float,
        history_fps: float,
        history_frames: int,
    ) -> None:
        self.tracker_fps = _positive_finite(tracker_fps, name="tracker_fps")
        self.history_fps = _positive_finite(history_fps, name="history_fps")
        self.history_frames = int(history_frames)
        if self.history_frames <= 0:
            raise ValueError("history_frames must be positive")
        required = int(
            math.ceil(
                max(0, self.history_frames - 1)
                * self.tracker_fps
                / self.history_fps
            )
        ) + 1
        self.required_tracker_frames = required
        self._max_tracker_frames = max(2, required + 2)
        initial = coerce_bumi_qpos_motion(initial_qpos, name="initial_qpos")
        self._frames = [frame.copy() for frame in initial[-self._max_tracker_frames :]]
        while len(self._frames) < self._max_tracker_frames:
            self._frames.insert(0, self._frames[0].copy())

    def reset(self, qpos: np.ndarray) -> None:
        """Reset the timestamp history without carrying samples across a gap."""

        frame = coerce_bumi_qpos_frame(qpos)
        self._frames = [frame.copy() for _ in range(self._max_tracker_frames)]

    def append(self, qpos: np.ndarray) -> None:
        self._frames.append(coerce_bumi_qpos_frame(qpos))
        if len(self._frames) > self._max_tracker_frames:
            del self._frames[: len(self._frames) - self._max_tracker_frames]

    def planner_history(self) -> np.ndarray:
        source = np.stack(self._frames, axis=0)
        # History follows timestamp semantics rather than plan frame-count
        # semantics, so sample the exact requested past timestamps.
        offsets = (
            np.arange(self.history_frames - 1, -1, -1, dtype=np.float64)
            * self.tracker_fps
            / self.history_fps
        )
        positions = (source.shape[0] - 1) - offsets
        positions = np.clip(positions, 0.0, source.shape[0] - 1.0)
        lo = np.floor(positions).astype(np.int64)
        hi = np.minimum(lo + 1, source.shape[0] - 1)
        alpha = positions - lo
        frames = [
            interpolate_bumi_qpos(source[a], source[b], amount, easing=False)
            for a, b, amount in zip(lo, hi, alpha)
        ]
        result = np.stack(frames, axis=0).astype(np.float32)
        result[-1] = source[-1]
        return result


class BumiTrajectoryHistory:
    """Keep the ten reference frames immediately preceding the current tick."""

    def __init__(self, initial_qpos: np.ndarray, *, history_frames: int = 10) -> None:
        self.history_frames = int(history_frames)
        if self.history_frames <= 0:
            raise ValueError("history_frames must be positive")
        initial = coerce_bumi_qpos_frame(initial_qpos)
        self._frames = [initial.copy() for _ in range(self.history_frames)]

    def packet_prefix(self, current_qpos: np.ndarray) -> np.ndarray:
        current = coerce_bumi_qpos_frame(current_qpos)
        prefix = np.stack([*self._frames, current], axis=0).astype(np.float32)
        self._frames.append(current.copy())
        del self._frames[: len(self._frames) - self.history_frames]
        return prefix
