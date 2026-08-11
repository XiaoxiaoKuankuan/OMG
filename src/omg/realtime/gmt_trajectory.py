from __future__ import annotations

import hashlib
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from omg.realtime.bumi_motion_mux import (
    BUMI_QPOS_DIM,
    coerce_bumi_qpos_motion,
)
from omg.robots.bumi.kinematics import BumiKinematics
from omg.tracking.holomotion.reference import (
    body_angvel_from_quats,
    finite_difference,
    normalize_quat_wxyz,
    quat_rotate_inv_wxyz,
)


TRAJECTORY_MAGIC = b"OMGBT001"
TRAJECTORY_VERSION = 1
TRAJECTORY_HEADER_FORMAT = "<8sHHIQQQqqfHHHH32sI"
TRAJECTORY_HEADER_SIZE = struct.calcsize(TRAJECTORY_HEADER_FORMAT)
TRAJECTORY_JOINT_COUNT = 21
TRAJECTORY_FRAME_DIM = 55
TRAJECTORY_HISTORY_FRAMES = 10
TRAJECTORY_PLAN_FRAMES = 100
TRAJECTORY_FRAME_COUNT = TRAJECTORY_HISTORY_FRAMES + TRAJECTORY_PLAN_FRAMES
TRAJECTORY_CURRENT_INDEX = TRAJECTORY_HISTORY_FRAMES

FLAG_FIXED_IDLE = 1 << 0
FLAG_TRANSITION = 1 << 1
FLAG_TEXT = 1 << 2
FLAG_AUDIO = 1 << 3
FLAG_ERROR = 1 << 4


def joint_order_sha256(joint_names: Iterable[str]) -> bytes:
    names = tuple(str(name).strip() for name in joint_names)
    if len(names) != TRAJECTORY_JOINT_COUNT or any(not name for name in names):
        raise ValueError(
            f"GMT joint order must contain {TRAJECTORY_JOINT_COUNT} non-empty names"
        )
    if len(set(names)) != len(names):
        raise ValueError("GMT joint order contains duplicate names")
    return hashlib.sha256("\n".join(names).encode("utf-8")).digest()


@dataclass(frozen=True)
class GmtPolicyContract:
    path: Path
    joint_names: tuple[str, ...]
    default_joint_pos: np.ndarray
    joint_order_hash: bytes

    @classmethod
    def from_onnx(cls, path: str | Path) -> "GmtPolicyContract":
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"GMT policy ONNX does not exist: {resolved}")
        import onnx

        model = onnx.load(str(resolved), load_external_data=False)
        metadata = {item.key: item.value for item in model.metadata_props}
        if "joint_names" not in metadata:
            raise ValueError(f"GMT policy is missing joint_names metadata: {resolved}")
        if "default_joint_pos" not in metadata:
            raise ValueError(f"GMT policy is missing default_joint_pos metadata: {resolved}")
        names = tuple(part.strip() for part in metadata["joint_names"].split(","))
        defaults = np.asarray(
            [float(part) for part in metadata["default_joint_pos"].split(",")],
            dtype=np.float32,
        )
        if len(names) != TRAJECTORY_JOINT_COUNT:
            raise ValueError(
                f"GMT policy must declare 21 joint names, got {len(names)} in {resolved}"
            )
        if defaults.shape != (TRAJECTORY_JOINT_COUNT,):
            raise ValueError(
                f"GMT policy default_joint_pos must have shape (21,), got {defaults.shape}"
            )
        if not np.isfinite(defaults).all():
            raise ValueError("GMT policy default_joint_pos contains non-finite values")
        return cls(
            path=resolved,
            joint_names=names,
            default_joint_pos=defaults,
            joint_order_hash=joint_order_sha256(names),
        )

    def native_to_gmt_indices(self, native_joint_names: Iterable[str]) -> np.ndarray:
        native = tuple(str(name).strip() for name in native_joint_names)
        if len(native) != TRAJECTORY_JOINT_COUNT or len(set(native)) != len(native):
            raise ValueError("BUMI native joint order must contain 21 unique names")
        native_index = {name: index for index, name in enumerate(native)}
        missing = sorted(set(self.joint_names) - set(native))
        extra = sorted(set(native) - set(self.joint_names))
        if missing or extra:
            raise ValueError(
                "BUMI/GMT joint-name mismatch: "
                f"missing_from_bumi={missing}, missing_from_gmt={extra}"
            )
        return np.asarray([native_index[name] for name in self.joint_names], dtype=np.int64)

    def default_in_native_order(self, native_joint_names: Iterable[str]) -> np.ndarray:
        native = tuple(str(name).strip() for name in native_joint_names)
        permutation = self.native_to_gmt_indices(native)
        # permutation maps each GMT element to a native source index.  Invert
        # it to place GMT metadata values into native BUMI qpos order.
        result = np.empty((TRAJECTORY_JOINT_COUNT,), dtype=np.float32)
        result[permutation] = self.default_joint_pos
        return result


def build_policy_default_idle_qpos(
    contract: GmtPolicyContract,
    *,
    kinematics_path: str | Path = "assets/robots/bumi/bumi_kinematics.json",
) -> tuple[np.ndarray, BumiKinematics, np.ndarray]:
    """Build and ground the fixed BUMI reference used when no command exists."""

    import torch

    kinematics = BumiKinematics(kinematics_path)
    native_joints = contract.default_in_native_order(kinematics.joint_order)
    lower = kinematics.joint_lower_limits.detach().cpu().numpy()
    upper = kinematics.joint_upper_limits.detach().cpu().numpy()
    if np.any(native_joints < lower - 1e-6) or np.any(native_joints > upper + 1e-6):
        raise ValueError("GMT policy default_joint_pos violates BUMI kinematic joint limits")
    qpos = np.zeros((BUMI_QPOS_DIM,), dtype=np.float32)
    qpos[3] = 1.0
    qpos[7:] = native_joints
    tensor = torch.from_numpy(qpos).view(1, 1, -1)
    with torch.no_grad():
        fk = kinematics.forward_kinematics(tensor)
        points, radii = kinematics.get_sole_proxy_points(
            fk["body_pos_w"], fk["body_quat_w"]
        )
        sole_bottom = points[..., 2] - radii.view(1, 1, -1)
        qpos[2] -= float(sole_bottom.min().item())
    return qpos, kinematics, contract.native_to_gmt_indices(kinematics.joint_order)


def qpos_timeline_to_gmt_frames(
    qpos: np.ndarray,
    *,
    fps: float,
    native_to_gmt: np.ndarray,
) -> np.ndarray:
    timeline = coerce_bumi_qpos_motion(qpos)
    rate = float(fps)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError(f"fps must be positive and finite, got {fps}")
    permutation = np.asarray(native_to_gmt, dtype=np.int64)
    if permutation.shape != (TRAJECTORY_JOINT_COUNT,) or set(permutation.tolist()) != set(
        range(TRAJECTORY_JOINT_COUNT)
    ):
        raise ValueError("native_to_gmt must be a permutation of 0..20")
    root_pos = timeline[:, :3]
    root_quat = np.stack(
        [normalize_quat_wxyz(value) for value in timeline[:, 3:7]], axis=0
    )
    root_lin_vel_w = finite_difference(root_pos, rate)
    root_lin_vel_b = np.stack(
        [quat_rotate_inv_wxyz(root_quat[i], root_lin_vel_w[i]) for i in range(len(timeline))],
        axis=0,
    )
    root_ang_vel_b = body_angvel_from_quats(root_quat, rate)
    native_joint_pos = timeline[:, 7:]
    native_joint_vel = finite_difference(native_joint_pos, rate)
    joint_pos = native_joint_pos[:, permutation]
    joint_vel = native_joint_vel[:, permutation]
    result = np.concatenate(
        (
            root_pos,
            root_quat,
            root_lin_vel_b,
            root_ang_vel_b,
            joint_pos,
            joint_vel,
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    if result.shape != (timeline.shape[0], TRAJECTORY_FRAME_DIM):
        raise AssertionError(f"Unexpected GMT frame shape: {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("GMT trajectory contains non-finite derived features")
    return result


@dataclass(frozen=True)
class GmtTrajectoryPacket:
    stream_id: int
    sequence: int
    published_unix_ns: int
    command_revision: int
    plan_id: int
    fps: float
    current_index: int
    flags: int
    joint_order_hash: bytes
    frames: np.ndarray

    def encode(self) -> bytes:
        frames = np.asarray(self.frames, dtype="<f4")
        if frames.ndim != 2 or frames.shape[1] != TRAJECTORY_FRAME_DIM:
            raise ValueError(
                f"frames must have shape (T,{TRAJECTORY_FRAME_DIM}), got {frames.shape}"
            )
        if frames.shape[0] <= 0 or not 0 <= int(self.current_index) < frames.shape[0]:
            raise ValueError("current_index must identify a frame in the packet")
        if not np.isfinite(frames).all():
            raise ValueError("frames contain non-finite values")
        hashes = bytes(self.joint_order_hash)
        if len(hashes) != 32:
            raise ValueError("joint_order_hash must contain 32 SHA256 bytes")
        payload = frames.tobytes(order="C")
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        header = struct.pack(
            TRAJECTORY_HEADER_FORMAT,
            TRAJECTORY_MAGIC,
            TRAJECTORY_VERSION,
            TRAJECTORY_HEADER_SIZE,
            int(self.flags),
            int(self.stream_id),
            int(self.sequence),
            int(self.published_unix_ns),
            int(self.command_revision),
            int(self.plan_id),
            float(self.fps),
            int(frames.shape[0]),
            int(self.current_index),
            TRAJECTORY_JOINT_COUNT,
            TRAJECTORY_FRAME_DIM,
            hashes,
            crc,
        )
        return header + payload

    @classmethod
    def decode(cls, blob: bytes | bytearray | memoryview) -> "GmtTrajectoryPacket":
        value = bytes(blob)
        if len(value) < TRAJECTORY_HEADER_SIZE:
            raise ValueError("trajectory packet is shorter than its fixed header")
        unpacked = struct.unpack(TRAJECTORY_HEADER_FORMAT, value[:TRAJECTORY_HEADER_SIZE])
        (
            magic,
            version,
            header_size,
            flags,
            stream_id,
            sequence,
            published_unix_ns,
            command_revision,
            plan_id,
            fps,
            frame_count,
            current_index,
            joint_count,
            frame_dim,
            order_hash,
            payload_crc,
        ) = unpacked
        if magic != TRAJECTORY_MAGIC or version != TRAJECTORY_VERSION:
            raise ValueError("unsupported GMT trajectory magic/version")
        if header_size != TRAJECTORY_HEADER_SIZE:
            raise ValueError(f"unsupported trajectory header size {header_size}")
        if joint_count != TRAJECTORY_JOINT_COUNT or frame_dim != TRAJECTORY_FRAME_DIM:
            raise ValueError("unexpected trajectory joint count or frame dimension")
        if (
            frame_count != TRAJECTORY_FRAME_COUNT
            or current_index != TRAJECTORY_CURRENT_INDEX
        ):
            raise ValueError(
                "unexpected trajectory frame count/current index: "
                f"{frame_count}/{current_index}"
            )
        if not np.isfinite(float(fps)) or float(fps) <= 0.0:
            raise ValueError(f"trajectory fps must be positive and finite, got {fps}")
        expected_size = header_size + int(frame_count) * int(frame_dim) * 4
        if len(value) != expected_size:
            raise ValueError(
                f"trajectory byte length mismatch: expected {expected_size}, got {len(value)}"
            )
        payload = value[header_size:]
        if (zlib.crc32(payload) & 0xFFFFFFFF) != payload_crc:
            raise ValueError("trajectory payload CRC32 mismatch")
        frames = np.frombuffer(payload, dtype="<f4").reshape(frame_count, frame_dim).copy()
        if not np.isfinite(frames).all():
            raise ValueError("trajectory packet contains non-finite values")
        return cls(
            stream_id=int(stream_id),
            sequence=int(sequence),
            published_unix_ns=int(published_unix_ns),
            command_revision=int(command_revision),
            plan_id=int(plan_id),
            fps=float(fps),
            current_index=int(current_index),
            flags=int(flags),
            joint_order_hash=bytes(order_hash),
            frames=frames,
        )


def make_trajectory_packet(
    qpos: np.ndarray,
    *,
    fps: float,
    native_to_gmt: np.ndarray,
    joint_order_hash: bytes,
    stream_id: int,
    sequence: int,
    command_revision: int,
    plan_id: int | None,
    flags: int,
    published_unix_ns: int | None = None,
) -> GmtTrajectoryPacket:
    timeline = coerce_bumi_qpos_motion(qpos)
    if timeline.shape != (TRAJECTORY_FRAME_COUNT, BUMI_QPOS_DIM):
        raise ValueError(
            f"trajectory timeline must have shape ({TRAJECTORY_FRAME_COUNT},28), "
            f"got {timeline.shape}"
        )
    return GmtTrajectoryPacket(
        stream_id=int(stream_id),
        sequence=int(sequence),
        published_unix_ns=(time.time_ns() if published_unix_ns is None else int(published_unix_ns)),
        command_revision=int(command_revision),
        plan_id=-1 if plan_id is None else int(plan_id),
        fps=float(fps),
        current_index=TRAJECTORY_CURRENT_INDEX,
        flags=int(flags),
        joint_order_hash=bytes(joint_order_hash),
        frames=qpos_timeline_to_gmt_frames(
            timeline, fps=float(fps), native_to_gmt=native_to_gmt
        ),
    )
