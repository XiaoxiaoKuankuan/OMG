from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from omg.cli.realtime.holomotion_real_bridge import _load_condition_sequence
from omg.realtime.dynamic_condition import DynamicConditionController, should_request_replan


def test_fixed_condition_sequence_mode_remains_supported() -> None:
    assert _load_condition_sequence(Namespace(condition_sequence="text: walk forward")) == (
        "text: walk forward"
    )


def test_new_dynamic_command_requests_early_replan_with_large_buffer() -> None:
    controller = DynamicConditionController(tracker_fps=50.0)
    active_revision = controller.current_revision
    controller.accept_text("turn left")

    assert should_request_replan(
        dynamic_enabled=True,
        current_revision=controller.current_revision,
        active_plan_revision=active_revision,
        pending=False,
        remaining_buffer_frames=90,
        replan_remaining_frames=40,
    )


def test_pending_request_serializes_new_command_until_completion() -> None:
    controller = DynamicConditionController(tracker_fps=50.0)
    active_revision = controller.current_revision
    controller.accept_text("wave")

    assert not should_request_replan(
        dynamic_enabled=True,
        current_revision=controller.current_revision,
        active_plan_revision=active_revision,
        pending=True,
        remaining_buffer_frames=90,
        replan_remaining_frames=40,
    )
    assert should_request_replan(
        dynamic_enabled=True,
        current_revision=controller.current_revision,
        active_plan_revision=active_revision,
        pending=False,
        remaining_buffer_frames=90,
        replan_remaining_frames=40,
    )


def test_audio_end_revision_triggers_stand_replan(tmp_path: Path) -> None:
    import numpy as np
    from scipy.io import wavfile

    path = tmp_path / "short.wav"
    wavfile.write(path, 10, np.zeros((10,), dtype=np.int16))
    controller = DynamicConditionController(tracker_fps=50.0)
    controller.accept_audio(str(path))
    audio_plan = controller.snapshot_for_replan(current_tracker_frame=5)
    assert controller.mark_replan_submitted(audio_plan)
    controller.update_for_tracker_frame(55)

    assert controller.snapshot().command_type == "stand"
    assert should_request_replan(
        dynamic_enabled=True,
        current_revision=controller.current_revision,
        active_plan_revision=audio_plan.revision,
        pending=False,
        remaining_buffer_frames=70,
        replan_remaining_frames=40,
    )


def test_bridges_do_not_directly_clear_motion_buffer() -> None:
    root = Path(__file__).resolve().parents[2]
    for relative in (
        "src/omg/cli/realtime/holomotion_dry_run.py",
        "src/omg/cli/realtime/holomotion_real_bridge.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        assert ".buffer.clear(" not in source
