from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np


CommandType = Literal["text", "audio", "stand"]


@dataclass(frozen=True)
class RuntimeCommand:
    command_id: str
    command_type: CommandType
    text: str | None
    audio_path: str | None
    audio_type: str
    on_end: str
    accepted_wall_time: float
    revision: int


@dataclass(frozen=True)
class ConditionSnapshot:
    command_id: str
    command_type: str
    revision: int
    condition_sequence: str
    condition_session_id: str
    condition_index: int
    audio_duration_seconds: float | None
    audio_start_tracker_frame: int | None
    audio_end_tracker_frame: int | None

    def metadata(
        self,
        *,
        audio_fps: float,
        tracker_fps: float,
        audio_type: str,
        audio_feature_type: str,
        condition_audio_step_frames: int,
    ) -> dict[str, Any]:
        return {
            "condition_sequence": self.condition_sequence,
            "condition_index": int(self.condition_index),
            "condition_session_id": self.condition_session_id,
            "condition_source": "dynamic_command",
            "command_id": self.command_id,
            "command_type": self.command_type,
            "command_revision": int(self.revision),
            "audio_fps": float(audio_fps),
            "tracker_fps": float(tracker_fps),
            "audio_type": str(audio_type),
            "audio_feature_type": str(audio_feature_type),
            "condition_audio_step_frames": int(condition_audio_step_frames),
        }


def compute_condition_audio_step_frames(
    *,
    planner_frames: int,
    history_fps: float,
    tracker_fps: float,
    replan_remaining_frames: int,
    audio_fps: float,
) -> int:
    planner_frames = int(planner_frames)
    history_fps = float(history_fps)
    tracker_fps = float(tracker_fps)
    audio_fps = float(audio_fps)
    replan_remaining_frames = int(replan_remaining_frames)
    if planner_frames <= 0:
        raise ValueError(f"planner_frames must be positive, got {planner_frames}")
    for name, value in (
        ("history_fps", history_fps),
        ("tracker_fps", tracker_fps),
        ("audio_fps", audio_fps),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite, got {value}")
    if replan_remaining_frames < 0:
        raise ValueError(
            "replan_remaining_frames must be non-negative, "
            f"got {replan_remaining_frames}"
        )
    plan_tracker_frames = int(round(planner_frames * tracker_fps / history_fps))
    replan_interval_tracker_frames = max(1, plan_tracker_frames - replan_remaining_frames)
    return max(1, int(round(replan_interval_tracker_frames * audio_fps / tracker_fps)))


def should_request_replan(
    *,
    dynamic_enabled: bool,
    current_revision: int | None,
    active_plan_revision: int | None,
    pending: bool,
    remaining_buffer_frames: int,
    replan_remaining_frames: int,
) -> bool:
    if pending:
        return False
    if dynamic_enabled and current_revision != active_plan_revision:
        return True
    return int(remaining_buffer_frames) <= int(replan_remaining_frames)


class DynamicConditionController:
    def __init__(
        self,
        *,
        tracker_fps: float,
        initial_condition_sequence: str = "text: stand still",
    ) -> None:
        fps = float(tracker_fps)
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"tracker_fps must be positive and finite, got {tracker_fps}")
        initial = str(initial_condition_sequence).strip()
        if not initial:
            raise ValueError("initial_condition_sequence must be non-empty")
        self._tracker_fps = fps
        self._lock = threading.RLock()
        self._revision = 0
        self._condition_session_id = uuid.uuid4().hex
        self._condition_index = 0
        self._condition_sequence = initial
        self._audio_start_tracker_frame: int | None = None
        self._audio_duration_seconds: float | None = None
        self._audio_end_tracker_frame: int | None = None
        self._last_requested_revision: int | None = None
        self._auto_stand_transitioned = False
        self._command = self._initial_command(initial)

    @property
    def current_revision(self) -> int:
        with self._lock:
            return int(self._revision)

    @property
    def last_requested_revision(self) -> int | None:
        with self._lock:
            return self._last_requested_revision

    def _initial_command(self, sequence: str) -> RuntimeCommand:
        now = time.time()
        command_id = uuid.uuid4().hex
        lowered = sequence.lower()
        if lowered == "text: stand still" or lowered == "stand":
            self._condition_sequence = "text: stand still"
            return RuntimeCommand(
                command_id=command_id,
                command_type="stand",
                text="stand still",
                audio_path=None,
                audio_type="audio",
                on_end="stand",
                accepted_wall_time=now,
                revision=0,
            )
        if lowered.startswith("text:") and "|" not in sequence:
            text = sequence.split(":", 1)[1].strip()
            if not text:
                raise ValueError("Initial text condition must be non-empty")
            return RuntimeCommand(
                command_id=command_id,
                command_type="text",
                text=text,
                audio_path=None,
                audio_type="audio",
                on_end="stand",
                accepted_wall_time=now,
                revision=0,
            )
        if lowered.startswith("audio:") and "|" not in sequence:
            audio_path = sequence.split(":", 1)[1].strip()
            path, duration = self._validate_audio_path(audio_path)
            self._condition_sequence = f"audio: {path}"
            self._audio_duration_seconds = duration
            return RuntimeCommand(
                command_id=command_id,
                command_type="audio",
                text=None,
                audio_path=str(path),
                audio_type="audio",
                on_end="stand",
                accepted_wall_time=now,
                revision=0,
            )
        # Preserve an existing fixed sequence when it is used as the dynamic
        # mode's initial value. A later command replaces it atomically.
        return RuntimeCommand(
            command_id=command_id,
            command_type="text",
            text=sequence,
            audio_path=None,
            audio_type="audio",
            on_end="stand",
            accepted_wall_time=now,
            revision=0,
        )

    @staticmethod
    def _command_id(value: object | None) -> str:
        if value is None or str(value).strip() == "":
            return uuid.uuid4().hex
        return str(value).strip()

    @staticmethod
    def _validate_text(value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("text must be a non-empty string")
        text = value.strip()
        if not text:
            raise ValueError("text must be a non-empty string")
        if "\n" in text or "\r" in text:
            raise ValueError("text must not contain newline characters")
        if "|" in text:
            raise ValueError("text must not contain '|' because it separates condition chunks")
        return text

    @staticmethod
    def _validate_audio_path(value: object) -> tuple[Path, float]:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("audio_path must be a non-empty absolute path")
        path = Path(value.strip())
        if not path.is_absolute():
            raise ValueError(f"audio_path must be absolute, got {path}")
        if not path.exists():
            raise FileNotFoundError(f"Audio path does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"audio_path must be a file, got {path}")
        if path.suffix.lower() != ".wav":
            raise ValueError(f"audio_path must have .wav suffix, got {path}")
        from scipy.io import wavfile

        sample_rate, waveform = wavfile.read(path)
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError(f"WAV sample rate must be positive, got {sample_rate} for {path}")
        audio = np.asarray(waveform)
        if audio.ndim <= 0 or int(audio.shape[0]) <= 0:
            raise ValueError(f"WAV must contain at least one sample: {path}")
        duration = float(audio.shape[0]) / float(sample_rate)
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError(f"WAV duration must be positive and finite, got {duration} for {path}")
        return path.resolve(), duration

    def _replace_command_locked(
        self,
        *,
        command_id: str,
        command_type: CommandType,
        text: str | None,
        audio_path: str | None,
        audio_duration_seconds: float | None,
        condition_sequence: str,
    ) -> RuntimeCommand:
        self._revision += 1
        self._condition_session_id = uuid.uuid4().hex
        self._condition_index = 0
        self._condition_sequence = str(condition_sequence)
        self._audio_start_tracker_frame = None
        self._audio_end_tracker_frame = None
        self._audio_duration_seconds = audio_duration_seconds
        self._auto_stand_transitioned = False
        command = RuntimeCommand(
            command_id=command_id,
            command_type=command_type,
            text=text,
            audio_path=audio_path,
            audio_type="audio",
            on_end="stand",
            accepted_wall_time=time.time(),
            revision=int(self._revision),
        )
        self._command = command
        print(
            f"[dynamic-condition] accepted command_id={command.command_id} "
            f"type={command.command_type} revision={command.revision}",
            flush=True,
        )
        return command

    def accept_text(self, text: object, *, command_id: object | None = None) -> RuntimeCommand:
        validated = self._validate_text(text)
        with self._lock:
            return self._replace_command_locked(
                command_id=self._command_id(command_id),
                command_type="text",
                text=validated,
                audio_path=None,
                audio_duration_seconds=None,
                condition_sequence=f"text: {validated}",
            )

    def accept_audio(
        self,
        audio_path: object,
        *,
        audio_type: object = "audio",
        on_end: object = "stand",
        command_id: object | None = None,
    ) -> RuntimeCommand:
        if audio_type != "audio":
            raise ValueError("audio_type must be 'audio'; looping/precomputed modes are not supported")
        if on_end != "stand":
            raise ValueError("on_end must be 'stand'")
        path, duration = self._validate_audio_path(audio_path)
        with self._lock:
            return self._replace_command_locked(
                command_id=self._command_id(command_id),
                command_type="audio",
                text=None,
                audio_path=str(path),
                audio_duration_seconds=duration,
                condition_sequence=f"audio: {path}",
            )

    def accept_stand(self, *, command_id: object | None = None) -> RuntimeCommand:
        with self._lock:
            return self._replace_command_locked(
                command_id=self._command_id(command_id),
                command_type="stand",
                text="stand still",
                audio_path=None,
                audio_duration_seconds=None,
                condition_sequence="text: stand still",
            )

    def accept_command(self, request: Mapping[str, Any]) -> RuntimeCommand:
        command_type = request.get("type")
        if not isinstance(command_type, str):
            raise ValueError("Command type must be one of: text, audio, stand")
        normalized = command_type.strip().lower()
        command_id = request.get("command_id")
        if normalized == "text":
            return self.accept_text(request.get("text"), command_id=command_id)
        if normalized == "audio":
            return self.accept_audio(
                request.get("audio_path"),
                audio_type=request.get("audio_type", "audio"),
                on_end=request.get("on_end", "stand"),
                command_id=command_id,
            )
        if normalized == "stand":
            return self.accept_stand(command_id=command_id)
        raise ValueError(f"Unsupported command type: {command_type!r}")

    def _snapshot_locked(self) -> ConditionSnapshot:
        return ConditionSnapshot(
            command_id=self._command.command_id,
            command_type=self._command.command_type,
            revision=int(self._revision),
            condition_sequence=self._condition_sequence,
            condition_session_id=self._condition_session_id,
            condition_index=int(self._condition_index),
            audio_duration_seconds=self._audio_duration_seconds,
            audio_start_tracker_frame=self._audio_start_tracker_frame,
            audio_end_tracker_frame=self._audio_end_tracker_frame,
        )

    def snapshot(self) -> ConditionSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def snapshot_for_replan(self, *, current_tracker_frame: int) -> ConditionSnapshot:
        frame = int(current_tracker_frame)
        if frame < 0:
            raise ValueError(f"current_tracker_frame must be non-negative, got {frame}")
        with self._lock:
            if self._command.command_type == "audio" and self._audio_start_tracker_frame is None:
                duration = float(self._audio_duration_seconds)
                self._audio_start_tracker_frame = frame
                self._audio_end_tracker_frame = frame + int(math.ceil(duration * self._tracker_fps))
                print(
                    f"[dynamic-condition] audio start command_id={self._command.command_id} "
                    f"path={self._command.audio_path} duration={duration:.6f}s start_frame={frame}",
                    flush=True,
                )
            return self._snapshot_locked()

    def mark_replan_submitted(self, snapshot: ConditionSnapshot) -> bool:
        with self._lock:
            matches = (
                snapshot.command_id == self._command.command_id
                and snapshot.revision == self._revision
                and snapshot.condition_session_id == self._condition_session_id
                and snapshot.condition_index == self._condition_index
            )
            if not matches:
                return False
            self._condition_index += 1
            self._last_requested_revision = int(snapshot.revision)
            return True

    def update_for_tracker_frame(self, current_tracker_frame: int) -> dict[str, Any] | None:
        frame = int(current_tracker_frame)
        if frame < 0:
            raise ValueError(f"current_tracker_frame must be non-negative, got {frame}")
        with self._lock:
            if (
                self._command.command_type != "audio"
                or self._audio_end_tracker_frame is None
                or frame < self._audio_end_tracker_frame
            ):
                return None
            ended_command = self._command
            audio_path = ended_command.audio_path
            duration = float(self._audio_duration_seconds)
            self._revision += 1
            self._condition_session_id = uuid.uuid4().hex
            self._condition_index = 0
            self._condition_sequence = "text: stand still"
            self._audio_start_tracker_frame = None
            self._audio_end_tracker_frame = None
            self._audio_duration_seconds = None
            self._auto_stand_transitioned = True
            self._command = RuntimeCommand(
                command_id=uuid.uuid4().hex,
                command_type="stand",
                text="stand still",
                audio_path=None,
                audio_type="audio",
                on_end="stand",
                accepted_wall_time=time.time(),
                revision=int(self._revision),
            )
            print(
                f"[dynamic-condition] audio ended command_id={ended_command.command_id} "
                f"frame={frame}; switching to stand",
                flush=True,
            )
            return {
                "kind": "audio_end",
                "command_id": ended_command.command_id,
                "audio_path": audio_path,
                "duration_seconds": duration,
                "tracker_frame": frame,
                "next_condition": "text: stand still",
            }

    def active_status(self) -> dict[str, Any]:
        with self._lock:
            snapshot = self._snapshot_locked()
            return {
                "command_id": snapshot.command_id,
                "type": snapshot.command_type,
                "text": self._command.text,
                "audio_path": self._command.audio_path,
                "audio_type": self._command.audio_type,
                "on_end": self._command.on_end,
                "accepted_wall_time": float(self._command.accepted_wall_time),
                "revision": int(snapshot.revision),
                "condition_sequence": snapshot.condition_sequence,
                "condition_session_id": snapshot.condition_session_id,
                "condition_index": int(snapshot.condition_index),
                "audio_duration_seconds": snapshot.audio_duration_seconds,
                "audio_start_tracker_frame": snapshot.audio_start_tracker_frame,
                "audio_end_tracker_frame": snapshot.audio_end_tracker_frame,
            }
