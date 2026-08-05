from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from omg.motion.feature_codec import G1MotionFeatureCodec, MotionComponents, MotionFeatureCodec
from omg.robots.base import RobotKinematics
from omg.robots.bumi.kinematics import BumiKinematics
from omg.robots.g1.kinematics import G1Kinematics
from omg.utils.rotation_conversions import standardize_quaternion


def _resolve_repo_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    return Path(__file__).resolve().parents[3] / value


class RobotMotionRepresentation(nn.Module):
    """Robot-parameterized motion representation with feature normalization."""

    def __init__(
        self,
        *,
        kinematics: RobotKinematics,
        stats_path: str | Path,
        num_prev_states: int = 2,
        canonical_frame_idx: int | None = None,
        feat_dim: int,
        sequence_length: int = 64,
        clip_std_min: float = 1e-6,
        rotation_representation: str = "quat",
        rot6d_gradient_mode: str = "vanilla",
        representation_name: str | None = None,
        codec_class: type[MotionFeatureCodec] = MotionFeatureCodec,
    ):
        super().__init__()
        self.kinematics = kinematics
        self.codec = codec_class(
            self.kinematics,
            num_prev_states=num_prev_states,
            canonical_frame_idx=canonical_frame_idx,
            rotation_representation=rotation_representation,
            rot6d_gradient_mode=rot6d_gradient_mode,
        )
        self.feat_dim = int(feat_dim)
        if self.feat_dim != self.codec.feature_dim:
            raise ValueError(
                "Configured feat_dim must exactly match the robot codec: "
                f"configured={self.feat_dim}, codec={self.codec.feature_dim}, "
                f"robot={self.kinematics.robot_name}"
            )
        self.robot_name = str(self.kinematics.robot_name)
        self.state_dim = int(getattr(self.kinematics, "qpos_dim", 7 + self.kinematics.num_joints))
        self.joint_names = tuple(self.kinematics.joint_order)
        self.representation_name = str(
            representation_name
            or f"{self.robot_name}_{self.codec.rotation_representation}_{self.feat_dim}d"
        )
        self.stats_path = str(_resolve_repo_path(stats_path))
        self.num_prev_states = int(num_prev_states)
        self.rotation_representation = self.codec.rotation_representation
        self.rot6d_gradient_mode = self.codec.rot6d_gradient_mode
        self.canonical_frame_idx = self.codec.canonical_frame_idx
        self.sequence_length = int(sequence_length)
        self.is_motion_representation = True

        stats = json.loads(Path(self.stats_path).read_text(encoding="utf-8"))
        mean = torch.tensor(stats["mean"], dtype=torch.float32)
        std = torch.tensor(stats["std"], dtype=torch.float32)
        default_root_pos = torch.tensor(stats["default_root_pos"], dtype=torch.float32)
        default_root_quat = torch.tensor(stats["default_root_quat"], dtype=torch.float32)
        default_joint_dof = torch.tensor(stats["default_joint_dof"], dtype=torch.float32)
        if mean.numel() != self.feat_dim or std.numel() != self.feat_dim:
            raise ValueError(
                f"Stats dim mismatch: expected {self.feat_dim}, got mean={mean.numel()}, std={std.numel()}"
            )
        if default_root_pos.shape != (3,) or default_root_quat.shape != (4,):
            raise ValueError("Stats default root values must have shapes [3] and [4]")
        if default_joint_dof.shape != (self.kinematics.num_joints,):
            raise ValueError(
                f"Stats default_joint_dof must have shape [{self.kinematics.num_joints}], "
                f"got {tuple(default_joint_dof.shape)}"
            )
        grounded_default_root_pos = self._ground_default_root_pos(
            default_root_pos,
            default_root_quat,
            default_joint_dof,
        )
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std.clamp_min(float(clip_std_min)), persistent=False)
        self.register_buffer("default_root_pos_stats", default_root_pos, persistent=False)
        self.register_buffer("default_root_pos", grounded_default_root_pos, persistent=False)
        self.register_buffer("default_root_quat", default_root_quat, persistent=False)
        self.register_buffer("default_joint_dof", default_joint_dof, persistent=False)
        self.obs_indices_dict = dict(self.codec.feature_slices)

    def _ground_default_root_pos(
        self,
        default_root_pos: torch.Tensor,
        default_root_quat: torch.Tensor,
        default_joint_dof: torch.Tensor,
    ) -> torch.Tensor:
        qpos = torch.cat(
            (
                default_root_pos.view(1, 1, 3),
                default_root_quat.view(1, 1, 4),
                default_joint_dof.view(1, 1, -1),
            ),
            dim=-1,
        )
        body_state = self.kinematics.forward_kinematics(qpos)
        sole_points, sole_radii = self.kinematics.get_sole_proxy_points(
            body_state["body_pos_w"], body_state["body_quat_w"]
        )
        sole_bottom = sole_points[..., 2] - sole_radii.view(1, 1, -1)
        grounded = default_root_pos.clone()
        grounded[2] -= sole_bottom.min()
        return grounded

    def build_obs_indices_dict(self) -> None:
        self.obs_indices_dict = dict(self.codec.feature_slices)

    def normalize_features(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.mean.to(value)) / self.std.to(value)

    def denormalize_features(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.std.to(value) + self.mean.to(value)

    def encode(self, batch: dict) -> torch.Tensor:
        if "motion_features" not in batch:
            raise KeyError("motion_features is required for RobotMotionRepresentation.encode")
        return self.normalize_features(batch["motion_features"])

    def decode(self, normalized: torch.Tensor) -> dict[str, torch.Tensor]:
        components = self.codec.split_features(self.denormalize_features(normalized))
        return {
            "root_pos_local": components.root_pos_local,
            "root_rot_local": components.root_rot_local_quat,
            "root_rot_local_quat": components.root_rot_local_quat,
            "joint_dof": components.joint_dof,
            "body_link_pos_local": components.body_link_pos_local,
        }

    def compose_qpos(
        self,
        decode_dict: dict[str, torch.Tensor],
        canon_root_pos: torch.Tensor,
        canon_root_quat: torch.Tensor,
    ) -> torch.Tensor:
        components = MotionComponents(
            root_pos_local=decode_dict["root_pos_local"],
            root_rot_local_quat=decode_dict["root_rot_local_quat"],
            joint_dof=decode_dict["joint_dof"],
            body_link_pos_local=decode_dict["body_link_pos_local"],
        )
        return self.codec.decode_to_world_qpos(
            components,
            anchor_root_pos=canon_root_pos,
            anchor_root_quat=canon_root_quat,
        )

    def get_default_prev_qpos(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        root_pos = self.default_root_pos.to(device=device, dtype=dtype).view(1, 1, 3)
        root_quat = self.default_root_quat.to(device=device, dtype=dtype).view(1, 1, 4)
        joints = self.default_joint_dof.to(device=device, dtype=dtype).view(1, 1, -1)
        root_pos = root_pos.expand(batch_size, self.num_prev_states, -1)
        root_quat = standardize_quaternion(
            F.normalize(root_quat.expand(batch_size, self.num_prev_states, -1), dim=-1)
        )
        joints = joints.expand(batch_size, self.num_prev_states, -1)
        return torch.cat((root_pos, root_quat, joints), dim=-1)

    def get_motion_dim(self) -> int:
        return self.feat_dim

    def get_obs_indices(self, obs: str):
        return self.obs_indices_dict[obs]


class G1MotionRepresentation(RobotMotionRepresentation):
    def __init__(
        self,
        stats_path: str | Path = "assets/stats/g1_125d_stats.json",
        kinematics_path: str | Path = "assets/robots/g1/g1_kinematics.json",
        num_prev_states: int = 2,
        canonical_frame_idx: int | None = None,
        feat_dim: int = 123,
        sequence_length: int = 64,
        clip_std_min: float = 1e-6,
        rotation_representation: str = "quat",
        rot6d_gradient_mode: str = "vanilla",
    ):
        super().__init__(
            kinematics=G1Kinematics(kinematics_path=kinematics_path),
            stats_path=stats_path,
            num_prev_states=num_prev_states,
            canonical_frame_idx=canonical_frame_idx,
            feat_dim=feat_dim,
            sequence_length=sequence_length,
            clip_std_min=clip_std_min,
            rotation_representation=rotation_representation,
            rot6d_gradient_mode=rot6d_gradient_mode,
            representation_name=f"g1_{rotation_representation}_{feat_dim}d",
            codec_class=G1MotionFeatureCodec,
        )

    def compose_qpos_36(
        self,
        decode_dict: dict[str, torch.Tensor],
        canon_root_pos: torch.Tensor,
        canon_root_quat: torch.Tensor,
    ) -> torch.Tensor:
        return self.compose_qpos(decode_dict, canon_root_pos, canon_root_quat)

    def get_default_prev_qpos36(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.get_default_prev_qpos(batch_size, device, dtype)


class BumiMotionRepresentation(RobotMotionRepresentation):
    def __init__(
        self,
        stats_path: str | Path = "assets/stats/bumi_93d_stats.json",
        kinematics_path: str | Path = "assets/robots/bumi/bumi_kinematics.json",
        num_prev_states: int = 2,
        canonical_frame_idx: int | None = None,
        feat_dim: int = 93,
        sequence_length: int = 64,
        clip_std_min: float = 1e-6,
        rotation_representation: str = "rot6d",
        rot6d_gradient_mode: str = "vanilla",
    ):
        super().__init__(
            kinematics=BumiKinematics(kinematics_path=kinematics_path),
            stats_path=stats_path,
            num_prev_states=num_prev_states,
            canonical_frame_idx=canonical_frame_idx,
            feat_dim=feat_dim,
            sequence_length=sequence_length,
            clip_std_min=clip_std_min,
            rotation_representation=rotation_representation,
            rot6d_gradient_mode=rot6d_gradient_mode,
            representation_name=f"bumi_{rotation_representation}_{feat_dim}d",
        )
