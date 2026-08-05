from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
import torch.nn as nn


G1_TO_BUMI_ADAPTER = "g1_to_bumi"

# These tensors encode the source robot's feature layout.  They must retain the
# target model's normal PyTorch initialization even when their shapes happen to
# match by accident.
_ROBOT_SPECIFIC_PREFIXES: tuple[tuple[str, str], ...] = (
    ("representation.", "representation_buffer"),
    ("history_projector.0.", "history_feature_normalization"),
    ("history_projector.1.", "history_motion_input_projection"),
    ("denoiser.input_proj.", "motion_input_projection"),
    ("denoiser.output.", "motion_output_projection"),
    ("motion_head.", "robot_specific_head"),
    ("robot_head.", "robot_specific_head"),
)


@dataclass(frozen=True)
class SkippedCheckpointTensor:
    key: str
    reason: str
    checkpoint_shape: tuple[int, ...] | None
    model_shape: tuple[int, ...] | None


@dataclass(frozen=True)
class CheckpointAdaptationReport:
    adapter: str
    source_robot: str
    target_robot: str
    loaded: tuple[str, ...]
    skipped: tuple[SkippedCheckpointTensor, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["counts"] = {
            "loaded": len(self.loaded),
            "skipped": len(self.skipped),
            "missing": len(self.missing),
            "unexpected": len(self.unexpected),
        }
        return payload


def _robot_specific_reason(key: str) -> str | None:
    for prefix, reason in _ROBOT_SPECIFIC_PREFIXES:
        if key.startswith(prefix):
            return reason
    return None


def adapt_g1_checkpoint_to_bumi(
    model: nn.Module,
    source_state_dict: Mapping[str, torch.Tensor],
) -> CheckpointAdaptationReport:
    """Load robot-independent G1 weights into a BUMI model explicitly.

    Shared condition encoders, timestep embeddings, transformer blocks and
    self/cross-attention parameters are loaded only when their names and shapes
    match exactly.  Robot-specific feature projections and normalization remain
    freshly initialized.  Any unclassified shape mismatch is rejected instead
    of being cropped or silently ignored.
    """

    target_robot = str(getattr(getattr(model, "representation", None), "robot_name", "unknown"))
    if target_robot != "bumi":
        raise ValueError(
            f"{G1_TO_BUMI_ADAPTER!r} requires a BUMI target representation, got {target_robot!r}"
        )
    target_state = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped: list[SkippedCheckpointTensor] = []
    unexpected: list[str] = []

    for key in sorted(source_state_dict):
        value = source_state_dict[key]
        target_value = target_state.get(key)
        reason = _robot_specific_reason(key)
        if reason is not None:
            skipped.append(
                SkippedCheckpointTensor(
                    key=key,
                    reason=reason,
                    checkpoint_shape=tuple(value.shape),
                    model_shape=None if target_value is None else tuple(target_value.shape),
                )
            )
            continue
        if target_value is None:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(target_value.shape):
            raise RuntimeError(
                "Unclassified checkpoint shape mismatch. Add an explicit robot-specific "
                f"adapter rule before loading {key!r}: checkpoint={tuple(value.shape)}, "
                f"model={tuple(target_value.shape)}"
            )
        filtered[key] = value

    incompatible = model.load_state_dict(filtered, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Adapter constructed an invalid state dict with unexpected keys: "
            f"{sorted(incompatible.unexpected_keys)}"
        )
    loaded = tuple(sorted(filtered))
    missing = tuple(sorted(incompatible.missing_keys))
    expected_missing = set(target_state) - set(filtered)
    if set(missing) != expected_missing:
        raise RuntimeError(
            "Checkpoint adapter missing-key accounting mismatch: "
            f"loader={missing}, expected={tuple(sorted(expected_missing))}"
        )
    return CheckpointAdaptationReport(
        adapter=G1_TO_BUMI_ADAPTER,
        source_robot="g1",
        target_robot=target_robot,
        loaded=loaded,
        skipped=tuple(skipped),
        missing=missing,
        unexpected=tuple(sorted(unexpected)),
    )


def adapt_checkpoint(
    adapter: str,
    model: nn.Module,
    source_state_dict: Mapping[str, torch.Tensor],
) -> CheckpointAdaptationReport:
    adapter = str(adapter)
    if adapter == G1_TO_BUMI_ADAPTER:
        return adapt_g1_checkpoint_to_bumi(model, source_state_dict)
    raise ValueError(
        f"Unsupported checkpoint adapter {adapter!r}; expected {G1_TO_BUMI_ADAPTER!r}"
    )
