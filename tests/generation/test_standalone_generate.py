from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from scipy.io import wavfile

from omg.cli.generation import generate as generate_cli
from omg.motion.representation import BumiMotionRepresentation


class _StandaloneModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.representation = BumiMotionRepresentation(
            num_prev_states=10,
            canonical_frame_idx=9,
            sequence_length=60,
        )
        self._device_dtype_anchor = nn.Parameter(torch.zeros(()), requires_grad=False)


def _args(**overrides) -> argparse.Namespace:
    values = {
        "history_source": "default",
        "history_val_index": 0,
        "fps": 30,
        "num_frames": 60,
        "text": None,
        "text_file": None,
        "music": None,
        "music_mod": "raw",
        "music_feature_type": "current35",
        "disable_audio_condition": False,
        "human_motion": None,
        "disable_human_motion_condition": False,
        "aligned_gt_comparison": False,
        "save_gt_motion": False,
        "render_comparison_video": False,
        "overlay_gt_on_generated": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _cfg(*, use_audio: bool = True, use_human_motion: bool = True):
    return SimpleNamespace(
        model={
            "use_audio": use_audio,
            "audio_dim": 35,
            "use_human_motion": use_human_motion,
            "human_motion_dim": 66,
        }
    )


def test_default_history_shapes_are_finite_and_canonical_quaternion_is_normalized() -> None:
    model = _StandaloneModel()
    prev_qpos = model.representation.get_default_prev_qpos(
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    batch = generate_cli._default_history_batch(model, fps=30.0, num_frames=120)

    assert prev_qpos.shape == (1, 10, 28)
    assert batch["history_features"].shape == (1, 10, 93)
    assert batch["prev_state_features"] is batch["history_features"]
    assert batch["canon_root_pos"].shape == (1, 1, 3)
    assert batch["canon_root_quat"].shape == (1, 1, 4)
    assert batch["mask"]["valid"].shape == (1, 120)
    torch.testing.assert_close(
        batch["canon_root_quat"].norm(dim=-1),
        torch.ones(1, 1),
        atol=1.0e-6,
        rtol=0.0,
    )
    assert all(
        torch.isfinite(value).all()
        for value in (
            prev_qpos,
            batch["history_features"],
            batch["canon_root_pos"],
            batch["canon_root_quat"],
        )
    )


def test_default_history_is_static_and_uses_existing_grounding() -> None:
    model = _StandaloneModel()
    representation = model.representation
    prev_qpos = representation.get_default_prev_qpos(1, torch.device("cpu"), torch.float32)
    assert torch.max(torch.abs(prev_qpos[:, 1:] - prev_qpos[:, :-1])).item() < 1.0e-7

    fk = representation.kinematics.forward_kinematics(prev_qpos)
    assert torch.isfinite(fk["body_pos_w"]).all()
    assert torch.isfinite(fk["body_quat_w"]).all()
    sole_points, sole_radii = representation.kinematics.get_sole_proxy_points(
        fk["body_pos_w"], fk["body_quat_w"]
    )
    sole_bottom = sole_points[..., 2] - sole_radii.view(1, 1, -1)
    assert float(sole_bottom.min()) >= -2.0e-5
    assert abs(float(sole_bottom.min())) <= 2.0e-5


def test_standalone_history_never_calls_dataset(monkeypatch) -> None:
    model = _StandaloneModel()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("standalone history attempted to instantiate a dataset")

    monkeypatch.setattr(generate_cli, "_val_dataset", fail_if_called)
    dataset, batch, history_meta, history_val_index = generate_cli._initialize_history(
        model,
        _cfg(),
        _args(),
        num_frames=60,
    )
    assert dataset is None
    assert history_meta == {}
    assert history_val_index is None
    assert batch["history_features"].shape == (1, 10, 93)


def test_dataset_history_mode_preserves_validation_history_path(monkeypatch) -> None:
    sentinel_dataset = object()
    sentinel_batch = {"meta": [{"window_start": np.int64(17)}]}
    calls = []

    def fake_val_dataset(cfg, *, num_frames):
        calls.append((cfg, num_frames))
        return sentinel_dataset

    monkeypatch.setattr(generate_cli, "_val_dataset", fake_val_dataset)
    monkeypatch.setattr(
        generate_cli,
        "_history_batch",
        lambda dataset, index: sentinel_batch if (dataset is sentinel_dataset and index == 7) else None,
    )
    cfg = _cfg()
    dataset, batch, history_meta, history_val_index = generate_cli._initialize_history(
        _StandaloneModel(),
        cfg,
        _args(history_source="dataset", history_val_index=7),
        num_frames=120,
    )
    assert calls == [(cfg, 120)]
    assert dataset is sentinel_dataset
    assert batch is sentinel_batch
    assert history_meta == {"window_start": 17}
    assert history_val_index == 7


def test_text_standalone_sets_text_and_leaves_optional_conditions_null(monkeypatch) -> None:
    monkeypatch.setattr(
        generate_cli,
        "_val_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("dataset accessed")),
    )
    args = _args(text="walk forward")
    _, batch, history_meta, _ = generate_cli._initialize_history(
        _StandaloneModel(), _cfg(), args, num_frames=60
    )
    prepared = generate_cli._prepare_condition_batch(
        args,
        _cfg(),
        batch=batch,
        dataset=None,
        history_meta=history_meta,
        text="walk forward",
        num_frames=60,
    )
    assert prepared["text"] == "walk forward"
    assert prepared["batch"]["caption"] == ["walk forward"]
    assert prepared["batch"]["has_text"].tolist() == [True]
    assert prepared["music_features"] is None
    assert prepared["human_motion"] is None
    assert "audio_features" not in prepared["batch"]
    assert "human_motion" not in prepared["batch"]


def test_raw_wav_standalone_starts_at_zero_and_never_accesses_dataset(tmp_path, monkeypatch) -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate // 4, dtype=np.float32) / float(sample_rate)
    waveform = (0.25 * np.sin(2.0 * np.pi * 220.0 * time)).astype(np.float32)
    wav_path = tmp_path / "test.wav"
    wavfile.write(wav_path, sample_rate, waveform)

    monkeypatch.setattr(
        generate_cli,
        "_val_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("dataset accessed")),
    )
    args = _args(music=str(wav_path), music_mod="raw", music_feature_type="current35")
    _, batch, history_meta, _ = generate_cli._initialize_history(
        _StandaloneModel(), _cfg(), args, num_frames=12
    )
    prepared = generate_cli._prepare_condition_batch(
        args,
        _cfg(),
        batch=batch,
        dataset=None,
        history_meta=history_meta,
        text="",
        num_frames=12,
    )
    assert prepared["music_start_frame"] == 0
    assert prepared["music_features"].shape == (12, 35)
    assert prepared["has_audio"].shape == (12,)
    assert prepared["has_audio"].any()
    assert prepared["batch"]["audio_features"].shape == (1, 12, 35)
    assert prepared["batch"]["mask"]["has_audio"].shape == (1, 12)
    assert prepared["human_motion"] is None


def test_humanref_standalone_starts_at_zero_and_never_accesses_dataset(tmp_path, monkeypatch) -> None:
    human_path = tmp_path / "human.npy"
    human = np.arange(8 * 66, dtype=np.float32).reshape(8, 66)
    np.save(human_path, human)
    monkeypatch.setattr(
        generate_cli,
        "_val_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("dataset accessed")),
    )
    args = _args(human_motion=str(human_path))
    _, batch, history_meta, _ = generate_cli._initialize_history(
        _StandaloneModel(), _cfg(), args, num_frames=12
    )
    prepared = generate_cli._prepare_condition_batch(
        args,
        _cfg(),
        batch=batch,
        dataset=None,
        history_meta=history_meta,
        text="",
        num_frames=12,
    )
    assert prepared["human_motion"].shape == (12, 66)
    assert prepared["has_human_motion"].tolist() == [True] * 8 + [False] * 4
    np.testing.assert_array_equal(prepared["human_motion"][:8].numpy(), human)
    assert prepared["batch"]["human_motion"].shape == (1, 12, 66)
    assert prepared["batch"]["mask"]["has_human_motion"].shape == (1, 12)
    assert prepared["music_features"] is None


@pytest.mark.parametrize(
    ("attribute", "flag"),
    [
        ("aligned_gt_comparison", "--aligned_gt_comparison"),
        ("save_gt_motion", "--save_gt_motion"),
        ("render_comparison_video", "--render_comparison_video"),
        ("overlay_gt_on_generated", "--overlay_gt_on_generated"),
    ],
)
def test_gt_options_require_dataset_history(attribute: str, flag: str) -> None:
    args = _args(**{attribute: True})
    with pytest.raises(ValueError, match=rf"{flag} requires --history_source dataset"):
        generate_cli._validate_history_source_args(args)


def test_checkpoint_loader_keeps_strict_state_dict_contract(monkeypatch) -> None:
    class RecordingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))
            self.strict = None

        def load_state_dict(self, state_dict, strict=True):
            self.strict = strict
            assert state_dict == {"weight": "sentinel"}
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    model = RecordingModel()
    monkeypatch.setattr(generate_cli, "instantiate", lambda cfg: model)
    monkeypatch.setattr(
        generate_cli.torch,
        "load",
        lambda path, map_location: {"state_dict": {"weight": "sentinel"}},
    )
    loaded = generate_cli._load_model(SimpleNamespace(model=object()), "checkpoint.ckpt")
    assert loaded is model
    assert model.strict is True
