from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from omg.realtime.dynamic_condition import (
    DynamicConditionController,
    compute_condition_audio_step_frames,
)


def _write_wav(path: Path, *, sample_rate: int = 8000, samples: int = 8000) -> Path:
    waveform = np.zeros((samples,), dtype=np.int16)
    wavfile.write(path, sample_rate, waveform)
    return path


def test_default_state_is_stand() -> None:
    controller = DynamicConditionController(tracker_fps=50.0)

    snapshot = controller.snapshot()

    assert snapshot.command_type == "stand"
    assert snapshot.condition_sequence == "text: stand still"
    assert snapshot.revision == 0
    assert snapshot.condition_index == 0


def test_text_command_is_accepted_and_has_no_end_time() -> None:
    controller = DynamicConditionController(tracker_fps=50.0)

    controller.accept_text("  walk forward slowly  ", command_id="walk-1")
    snapshot = controller.snapshot()

    assert snapshot.command_id == "walk-1"
    assert snapshot.command_type == "text"
    assert snapshot.condition_sequence == "text: walk forward slowly"
    assert snapshot.audio_duration_seconds is None
    assert snapshot.audio_start_tracker_frame is None
    assert snapshot.audio_end_tracker_frame is None


@pytest.mark.parametrize("text", ["", "   ", "turn | left", "turn\nleft", "turn\rleft"])
def test_invalid_text_does_not_replace_current_command(text: str) -> None:
    controller = DynamicConditionController(tracker_fps=50.0)
    before = controller.snapshot()

    with pytest.raises(ValueError):
        controller.accept_text(text)

    assert controller.snapshot() == before


def test_new_text_replaces_old_text_with_new_session_and_reset_index() -> None:
    controller = DynamicConditionController(tracker_fps=50.0)
    controller.accept_text("walk", command_id="walk")
    first = controller.snapshot_for_replan(current_tracker_frame=0)
    assert controller.mark_replan_submitted(first)
    assert controller.snapshot().condition_index == 1

    controller.accept_text("turn left", command_id="turn")
    second = controller.snapshot()

    assert second.command_id == "turn"
    assert second.condition_sequence == "text: turn left"
    assert second.condition_session_id != first.condition_session_id
    assert second.condition_index == 0
    assert second.revision == first.revision + 1


def test_failed_submission_does_not_increment_condition_index() -> None:
    controller = DynamicConditionController(tracker_fps=50.0)
    controller.accept_text("walk")
    old = controller.snapshot_for_replan(current_tracker_frame=0)
    controller.accept_text("turn")

    assert not controller.mark_replan_submitted(old)
    assert controller.snapshot().condition_index == 0


def test_audio_path_validation_preserves_active_command(tmp_path: Path) -> None:
    controller = DynamicConditionController(tracker_fps=50.0)
    before = controller.snapshot()
    non_wav = tmp_path / "music.mp3"
    non_wav.write_bytes(b"not audio")

    for invalid in ("relative.wav", str(tmp_path / "missing.wav"), str(non_wav)):
        with pytest.raises((ValueError, FileNotFoundError)):
            controller.accept_audio(invalid)
        assert controller.snapshot() == before


def test_empty_wav_is_rejected_without_replacing_current(tmp_path: Path) -> None:
    controller = DynamicConditionController(tracker_fps=50.0)
    path = tmp_path / "empty.wav"
    _write_wav(path, samples=0)
    before = controller.snapshot()

    with pytest.raises(ValueError, match="at least one sample"):
        controller.accept_audio(str(path))

    assert controller.snapshot() == before


def test_invalid_wav_sample_rate_is_rejected_without_replacing_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = DynamicConditionController(tracker_fps=50.0)
    path = _write_wav(tmp_path / "invalid-rate.wav")
    before = controller.snapshot()
    monkeypatch.setattr("scipy.io.wavfile.read", lambda _path: (0, np.ones((10,), dtype=np.int16)))

    with pytest.raises(ValueError, match="sample rate"):
        controller.accept_audio(str(path))

    assert controller.snapshot() == before


def test_audio_duration_start_and_end_switch_to_stand(tmp_path: Path) -> None:
    controller = DynamicConditionController(tracker_fps=50.0)
    path = _write_wav(tmp_path / "two-seconds.wav", sample_rate=8000, samples=16000)
    controller.accept_audio(str(path), command_id="music")

    before_start = controller.snapshot()
    assert before_start.audio_duration_seconds == pytest.approx(2.0)
    assert before_start.audio_start_tracker_frame is None
    started = controller.snapshot_for_replan(current_tracker_frame=17)

    assert started.audio_start_tracker_frame == 17
    assert started.audio_end_tracker_frame == 117
    assert controller.update_for_tracker_frame(116) is None

    event = controller.update_for_tracker_frame(117)
    ended = controller.snapshot()
    assert event == {
        "kind": "audio_end",
        "command_id": "music",
        "audio_path": str(path.resolve()),
        "duration_seconds": 2.0,
        "tracker_frame": 117,
        "next_condition": "text: stand still",
    }
    assert ended.command_type == "stand"
    assert ended.condition_sequence == "text: stand still"
    assert ended.condition_session_id != started.condition_session_id
    assert ended.condition_index == 0
    assert ended.revision == started.revision + 1


def test_audio_replaced_by_text_cannot_later_trigger_old_end(tmp_path: Path) -> None:
    controller = DynamicConditionController(tracker_fps=50.0)
    path = _write_wav(tmp_path / "music.wav", sample_rate=10, samples=10)
    controller.accept_audio(str(path), command_id="music")
    music = controller.snapshot_for_replan(current_tracker_frame=10)
    assert music.audio_end_tracker_frame == 60

    controller.accept_text("wave both arms", command_id="wave")
    text = controller.snapshot()

    assert controller.update_for_tracker_frame(1000) is None
    assert controller.snapshot() == text
    assert controller.snapshot().command_type == "text"


def test_new_audio_restarts_timeline_and_cancels_old_audio_end(tmp_path: Path) -> None:
    controller = DynamicConditionController(tracker_fps=50.0)
    first_path = _write_wav(tmp_path / "first.wav", sample_rate=10, samples=10)
    second_path = _write_wav(tmp_path / "second.wav", sample_rate=10, samples=20)
    controller.accept_audio(str(first_path), command_id="first")
    first = controller.snapshot_for_replan(current_tracker_frame=100)
    assert first.audio_end_tracker_frame == 150

    controller.accept_audio(str(second_path), command_id="second")
    before_second_start = controller.snapshot()
    assert before_second_start.audio_start_tracker_frame is None
    assert before_second_start.audio_end_tracker_frame is None
    assert before_second_start.condition_session_id != first.condition_session_id
    second = controller.snapshot_for_replan(current_tracker_frame=150)

    assert second.audio_start_tracker_frame == 150
    assert second.audio_end_tracker_frame == 250
    assert controller.update_for_tracker_frame(150) is None
    assert controller.snapshot().command_id == "second"


def test_status_contains_required_fields(tmp_path: Path) -> None:
    controller = DynamicConditionController(tracker_fps=50.0)
    path = _write_wav(tmp_path / "music.wav")
    controller.accept_audio(str(path), command_id="status-audio")
    controller.snapshot_for_replan(current_tracker_frame=23)

    active = controller.active_status()

    required = {
        "command_id",
        "type",
        "text",
        "audio_path",
        "revision",
        "condition_session_id",
        "condition_index",
        "audio_duration_seconds",
        "audio_start_tracker_frame",
        "audio_end_tracker_frame",
    }
    assert required <= active.keys()
    assert active["audio_start_tracker_frame"] == 23


def test_concurrent_reads_and_writes_never_mix_command_fields() -> None:
    controller = DynamicConditionController(tracker_fps=50.0)
    controller.accept_text("motion 0", command_id="cmd-0")
    errors: list[str] = []
    barrier = threading.Barrier(5)

    def writer(offset: int) -> None:
        barrier.wait()
        for index in range(offset, 40, 2):
            controller.accept_text(f"motion {index}", command_id=f"cmd-{index}")

    def reader() -> None:
        barrier.wait()
        for _ in range(300):
            active = controller.active_status()
            if active["type"] != "text":
                errors.append(f"unexpected type: {active}")
                continue
            suffix = str(active["command_id"]).removeprefix("cmd-")
            if active["text"] != f"motion {suffix}":
                errors.append(f"mixed text: {active}")
            if active["condition_sequence"] != f"text: motion {suffix}":
                errors.append(f"mixed sequence: {active}")

    threads = [
        threading.Thread(target=writer, args=(1,)),
        threading.Thread(target=writer, args=(2,)),
        threading.Thread(target=reader),
        threading.Thread(target=reader),
        threading.Thread(target=reader),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []


def test_computed_audio_step_matches_replan_execution_interval() -> None:
    assert compute_condition_audio_step_frames(
        planner_frames=60,
        history_fps=30.0,
        tracker_fps=50.0,
        replan_remaining_frames=40,
        audio_fps=30.0,
    ) == 36
