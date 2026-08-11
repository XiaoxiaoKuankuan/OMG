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
class WavTiming:
    path: Path
    sample_rate: int
    sample_count: int
    source_duration_seconds: float
    effective_duration_seconds: float
    trailing_silence_seconds: float
    silence_threshold_dbfs: float
    minimum_trailing_silence_seconds: float
    analysis_window_seconds: float
    has_audible_content: bool


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
    audio_source_duration_seconds: float | None = None
    audio_effective_duration_seconds: float | None = None
    audio_trailing_silence_seconds: float | None = None

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
            "audio_source_duration_seconds": self.audio_source_duration_seconds,
            "audio_effective_duration_seconds": self.audio_effective_duration_seconds,
            "audio_trailing_silence_seconds": self.audio_trailing_silence_seconds,
        }


def _wav_full_scale(dtype: np.dtype[Any]) -> tuple[float, float]:
    """Return the PCM zero point and positive full-scale magnitude."""

    if np.issubdtype(dtype, np.unsignedinteger):
        info = np.iinfo(dtype)
        zero = float(info.min + (info.max - info.min + 1) / 2)
        return zero, max(1.0, float(info.max) - zero + 1.0)
    if np.issubdtype(dtype, np.signedinteger):
        info = np.iinfo(dtype)
        return 0.0, max(abs(float(info.min)), abs(float(info.max)))
    if np.issubdtype(dtype, np.floating):
        return 0.0, 1.0
    raise ValueError(f"Unsupported WAV sample dtype: {dtype}")


def analyze_wav_timing(
    value: object,
    *,
    silence_threshold_dbfs: float = -50.0,
    minimum_trailing_silence_seconds: float = 0.5,
    analysis_window_seconds: float = 0.02,
) -> WavTiming:
    """Validate a WAV and find a sufficiently long continuous silent tail.

    Silence is measured as the RMS level of short windows relative to PCM full
    scale.  The final non-silent window is retained, so the effective endpoint
    is never rounded earlier than audible content.  An entirely silent WAV is
    intentionally left untrimmed for backwards compatibility.
    """

    threshold_dbfs = float(silence_threshold_dbfs)
    minimum_silence = float(minimum_trailing_silence_seconds)
    window_seconds = float(analysis_window_seconds)
    if not np.isfinite(threshold_dbfs) or threshold_dbfs >= 0.0:
        raise ValueError("silence_threshold_dbfs must be finite and below 0 dBFS")
    if not np.isfinite(minimum_silence) or minimum_silence < 0.0:
        raise ValueError(
            "minimum_trailing_silence_seconds must be non-negative and finite"
        )
    if not np.isfinite(window_seconds) or window_seconds <= 0.0:
        raise ValueError("analysis_window_seconds must be positive and finite")
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
    if not np.isfinite(audio).all():
        raise ValueError(f"WAV contains non-finite samples: {path}")

    sample_count = int(audio.shape[0])
    source_duration = float(sample_count) / float(sample_rate)
    if not np.isfinite(source_duration) or source_duration <= 0.0:
        raise ValueError(
            f"WAV duration must be positive and finite, got {source_duration} for {path}"
        )

    zero, full_scale = _wav_full_scale(audio.dtype)
    threshold = 10.0 ** (threshold_dbfs / 20.0)
    window_samples = max(1, int(round(window_seconds * sample_rate)))
    last_audible_window_end: int | None = None
    window_end = sample_count
    while window_end > 0:
        window_start = max(0, window_end - window_samples)
        values = (audio[window_start:window_end].astype(np.float64) - zero) / full_scale
        if values.ndim > 1:
            channel_axes = tuple(range(1, values.ndim))
            values = np.max(np.abs(values), axis=channel_axes)
        rms = float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))
        if rms > threshold:
            last_audible_window_end = window_end
            break
        window_end = window_start

    has_audible_content = last_audible_window_end is not None
    effective_sample_count = sample_count
    trailing_silence = 0.0
    if has_audible_content:
        candidate = int(last_audible_window_end)
        candidate_trailing = float(sample_count - candidate) / float(sample_rate)
        if candidate_trailing + (0.5 / sample_rate) >= minimum_silence:
            effective_sample_count = max(1, candidate)
            trailing_silence = float(sample_count - effective_sample_count) / float(
                sample_rate
            )

    effective_duration = float(effective_sample_count) / float(sample_rate)
    return WavTiming(
        path=path.resolve(),
        sample_rate=sample_rate,
        sample_count=sample_count,
        source_duration_seconds=source_duration,
        effective_duration_seconds=effective_duration,
        trailing_silence_seconds=trailing_silence,
        silence_threshold_dbfs=threshold_dbfs,
        minimum_trailing_silence_seconds=minimum_silence,
        analysis_window_seconds=window_seconds,
        has_audible_content=has_audible_content,
    )


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
        audio_tail_silence_threshold_dbfs: float = -50.0,
        audio_tail_silence_min_seconds: float = 0.5,
        audio_tail_analysis_window_seconds: float = 0.02,
    ) -> None:
        fps = float(tracker_fps)
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"tracker_fps must be positive and finite, got {tracker_fps}")
        initial = str(initial_condition_sequence).strip()
        if not initial:
            raise ValueError("initial_condition_sequence must be non-empty")
        threshold_dbfs = float(audio_tail_silence_threshold_dbfs)
        minimum_silence = float(audio_tail_silence_min_seconds)
        analysis_window = float(audio_tail_analysis_window_seconds)
        if not np.isfinite(threshold_dbfs) or threshold_dbfs >= 0.0:
            raise ValueError(
                "audio_tail_silence_threshold_dbfs must be finite and below 0"
            )
        if not np.isfinite(minimum_silence) or minimum_silence < 0.0:
            raise ValueError(
                "audio_tail_silence_min_seconds must be non-negative and finite"
            )
        if not np.isfinite(analysis_window) or analysis_window <= 0.0:
            raise ValueError(
                "audio_tail_analysis_window_seconds must be positive and finite"
            )
        self._tracker_fps = fps
        self._audio_tail_silence_threshold_dbfs = threshold_dbfs
        self._audio_tail_silence_min_seconds = minimum_silence
        self._audio_tail_analysis_window_seconds = analysis_window
        self._lock = threading.RLock()
        self._revision = 0
        self._condition_session_id = uuid.uuid4().hex
        self._condition_index = 0
        self._condition_sequence = initial
        self._audio_start_tracker_frame: int | None = None
        self._audio_duration_seconds: float | None = None
        self._audio_source_duration_seconds: float | None = None
        self._audio_trailing_silence_seconds: float | None = None
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
            timing = self._validate_audio_path(audio_path)
            self._condition_sequence = f"audio: {timing.path}"
            self._set_audio_timing_locked(timing)
            return RuntimeCommand(
                command_id=command_id,
                command_type="audio",
                text=None,
                audio_path=str(timing.path),
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

    def _validate_audio_path(self, value: object) -> WavTiming:
        return analyze_wav_timing(
            value,
            silence_threshold_dbfs=self._audio_tail_silence_threshold_dbfs,
            minimum_trailing_silence_seconds=self._audio_tail_silence_min_seconds,
            analysis_window_seconds=self._audio_tail_analysis_window_seconds,
        )

    def _set_audio_timing_locked(self, timing: WavTiming | None) -> None:
        self._audio_duration_seconds = (
            None if timing is None else float(timing.effective_duration_seconds)
        )
        self._audio_source_duration_seconds = (
            None if timing is None else float(timing.source_duration_seconds)
        )
        self._audio_trailing_silence_seconds = (
            None if timing is None else float(timing.trailing_silence_seconds)
        )

    def _replace_command_locked(
        self,
        *,
        command_id: str,
        command_type: CommandType,
        text: str | None,
        audio_path: str | None,
        audio_timing: WavTiming | None,
        condition_sequence: str,
    ) -> RuntimeCommand:
        self._revision += 1
        self._condition_session_id = uuid.uuid4().hex
        self._condition_index = 0
        self._condition_sequence = str(condition_sequence)
        self._audio_start_tracker_frame = None
        self._audio_end_tracker_frame = None
        self._set_audio_timing_locked(audio_timing)
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
                audio_timing=None,
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
        timing = self._validate_audio_path(audio_path)
        with self._lock:
            command = self._replace_command_locked(
                command_id=self._command_id(command_id),
                command_type="audio",
                text=None,
                audio_path=str(timing.path),
                audio_timing=timing,
                condition_sequence=f"audio: {timing.path}",
            )
            print(
                "[dynamic-condition] audio timing "
                f"path={timing.path} source={timing.source_duration_seconds:.6f}s "
                f"effective={timing.effective_duration_seconds:.6f}s "
                f"trimmed_tail={timing.trailing_silence_seconds:.6f}s "
                f"threshold={timing.silence_threshold_dbfs:.1f}dBFS",
                flush=True,
            )
            return command

    def accept_stand(self, *, command_id: object | None = None) -> RuntimeCommand:
        with self._lock:
            return self._replace_command_locked(
                command_id=self._command_id(command_id),
                command_type="stand",
                text="stand still",
                audio_path=None,
                audio_timing=None,
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
            audio_source_duration_seconds=self._audio_source_duration_seconds,
            audio_effective_duration_seconds=self._audio_duration_seconds,
            audio_trailing_silence_seconds=self._audio_trailing_silence_seconds,
        )

    def snapshot(self) -> ConditionSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def snapshot_for_replan(
        self,
        *,
        current_tracker_frame: int,
        start_audio: bool = True,
    ) -> ConditionSnapshot:
        frame = int(current_tracker_frame)
        if frame < 0:
            raise ValueError(f"current_tracker_frame must be non-negative, got {frame}")
        with self._lock:
            if (
                start_audio
                and self._command.command_type == "audio"
                and self._audio_start_tracker_frame is None
            ):
                duration = float(self._audio_duration_seconds)
                self._audio_start_tracker_frame = frame
                self._audio_end_tracker_frame = frame + int(math.ceil(duration * self._tracker_fps))
                print(
                    f"[dynamic-condition] audio start command_id={self._command.command_id} "
                    f"path={self._command.audio_path} duration={duration:.6f}s start_frame={frame}",
                    flush=True,
                )
            return self._snapshot_locked()

    def mark_audio_execution_started(
        self,
        snapshot: ConditionSnapshot,
        *,
        current_tracker_frame: int,
    ) -> bool:
        """Start an audio clock when its first motion frame is actually emitted.

        Existing bridges retain the historical ``snapshot_for_replan``
        behavior.  A bridge that passes ``start_audio=False`` can call this
        method on the first accepted output tick so diffusion latency is not
        counted as music execution time.
        """

        frame = int(current_tracker_frame)
        if frame < 0:
            raise ValueError(f"current_tracker_frame must be non-negative, got {frame}")
        with self._lock:
            matches = (
                self._command.command_type == "audio"
                and snapshot.command_type == "audio"
                and snapshot.command_id == self._command.command_id
                and snapshot.revision == self._revision
                and snapshot.condition_session_id == self._condition_session_id
            )
            if not matches or self._audio_start_tracker_frame is not None:
                return False
            duration = float(self._audio_duration_seconds)
            self._audio_start_tracker_frame = frame
            self._audio_end_tracker_frame = frame + int(
                math.ceil(duration * self._tracker_fps)
            )
            print(
                f"[dynamic-condition] audio start command_id={self._command.command_id} "
                f"path={self._command.audio_path} duration={duration:.6f}s "
                f"start_frame={frame}",
                flush=True,
            )
            return True

    def audio_elapsed_tracker_frames(self, current_tracker_frame: int) -> int:
        frame = int(current_tracker_frame)
        if frame < 0:
            raise ValueError(f"current_tracker_frame must be non-negative, got {frame}")
        with self._lock:
            if (
                self._command.command_type != "audio"
                or self._audio_start_tracker_frame is None
            ):
                return 0
            return max(0, frame - int(self._audio_start_tracker_frame))

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
            source_duration = float(self._audio_source_duration_seconds)
            trailing_silence = float(self._audio_trailing_silence_seconds)
            self._revision += 1
            self._condition_session_id = uuid.uuid4().hex
            self._condition_index = 0
            self._condition_sequence = "text: stand still"
            self._audio_start_tracker_frame = None
            self._audio_end_tracker_frame = None
            self._set_audio_timing_locked(None)
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
                "source_duration_seconds": source_duration,
                "effective_duration_seconds": duration,
                "trimmed_trailing_silence_seconds": trailing_silence,
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
            }
