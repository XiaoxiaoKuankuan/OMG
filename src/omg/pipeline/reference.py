from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MotionReference:
    path: Path
    qpos: np.ndarray
    fps: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def state_dim(self) -> int:
        return int(self.qpos.shape[1])

    @property
    def qpos_36(self) -> np.ndarray:
        """Compatibility alias for existing G1-only callers."""

        return self.qpos


def _coerce_qpos(value: Any, source: Path, *, expected_state_dim: int) -> np.ndarray:
    qpos = np.asarray(value, dtype=np.float32)
    if qpos.ndim == 3:
        if qpos.shape[0] != 1:
            raise ValueError(f"Expected batch dimension 1 in {source}, got {qpos.shape}")
        qpos = qpos[0]
    if qpos.ndim != 2 or qpos.shape[1] != int(expected_state_dim):
        raise ValueError(
            f"Expected qpos shape (T,{int(expected_state_dim)}) in {source}, got {qpos.shape}"
        )
    if qpos.shape[0] <= 0:
        raise ValueError(f"Reference motion is empty: {source}")
    if not np.isfinite(qpos).all():
        raise ValueError(f"Reference motion contains non-finite values: {source}")
    quat_norm = np.linalg.norm(qpos[:, 3:7], axis=1)
    if np.any(quat_norm <= 1.0e-8):
        raise ValueError(f"Reference motion contains an invalid root quaternion: {source}")
    qpos = qpos.copy()
    qpos[:, 3:7] /= quat_norm[:, None]
    qpos[qpos[:, 3] < 0.0, 3:7] *= -1.0
    return qpos.astype(np.float32, copy=False)


def _coerce_fps(value: Any, source: Path) -> float:
    arr = np.asarray(value, dtype=np.float32)
    if arr.size != 1:
        raise ValueError(f"Expected scalar fps in {source}, got shape {arr.shape}")
    fps = float(arr.reshape(-1)[0])
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"Expected positive finite fps in {source}, got {fps}")
    return fps


def _metadata_from_npz(data: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in data.files:
        if key in {"qpos", "qpos_36", "pred_qpos_36", "fps", "motion_features"}:
            continue
        arr = np.asarray(data[key])
        if arr.shape == ():
            metadata[key] = arr.item()
        elif arr.size == 1 and arr.dtype.kind in {"U", "S", "O"}:
            metadata[key] = arr.reshape(-1)[0].item()
    return metadata


def _load_pt(path: Path) -> tuple[Any, dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError(f"Loading {path.suffix} references requires torch") from exc
    payload = torch.load(path, map_location="cpu")
    metadata: dict[str, Any] = {}
    if isinstance(payload, dict):
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"qpos", "qpos_36", "pred_qpos_36", "motion_features"}
        }
        for key in ("qpos", "qpos_36", "pred_qpos_36"):
            if key in payload:
                value = payload[key]
                break
        else:
            raise KeyError(f"No qpos, qpos_36, or pred_qpos_36 key found in {path}")
    else:
        value = payload
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return value, metadata


def load_motion_reference(
    path: str | Path,
    *,
    expected_state_dim: int,
    fps: float | None = None,
) -> MotionReference:
    ref_path = Path(path).expanduser().resolve()
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference motion not found: {ref_path}")

    metadata: dict[str, Any] = {}
    loaded_fps: float | None = None
    if ref_path.suffix == ".npy":
        qpos_value = np.load(ref_path)
    elif ref_path.suffix == ".npz":
        with np.load(ref_path, allow_pickle=False) as data:
            for key in ("qpos", "qpos_36", "pred_qpos_36"):
                if key in data:
                    qpos_value = np.asarray(data[key])
                    break
            else:
                raise KeyError(f"No qpos, qpos_36, or pred_qpos_36 key found in {ref_path}")
            if "fps" in data:
                loaded_fps = _coerce_fps(data["fps"], ref_path)
            metadata = _metadata_from_npz(data)
    elif ref_path.suffix in {".pt", ".pth"}:
        qpos_value, metadata = _load_pt(ref_path)
        if "fps" in metadata:
            loaded_fps = _coerce_fps(metadata["fps"], ref_path)
    else:
        raise ValueError(f"Unsupported reference motion extension: {ref_path.suffix}")

    resolved_fps = _coerce_fps(fps, ref_path) if fps is not None else loaded_fps
    if resolved_fps is None:
        raise ValueError(f"Reference fps is required for {ref_path}")
    qpos = _coerce_qpos(qpos_value, ref_path, expected_state_dim=int(expected_state_dim))
    metadata_state_dim = metadata.get("state_dim")
    if metadata_state_dim is not None and int(metadata_state_dim) != int(expected_state_dim):
        raise ValueError(
            f"Reference metadata state_dim={metadata_state_dim} does not match expected {expected_state_dim}"
        )
    return MotionReference(path=ref_path, qpos=qpos, fps=float(resolved_fps), metadata=metadata)
