from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from omg.utils.rotation_conversions import (
    axis_angle_to_matrix,
    euler_angles_to_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
    standardize_quaternion,
)


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    return Path(__file__).resolve().parents[3] / value


def _view_for_batch(value: torch.Tensor, batch_ndim: int) -> torch.Tensor:
    return value.view(*([1] * batch_ndim), *value.shape)


class GenericKinematics(nn.Module):
    """Differentiable FK driven by an exported MuJoCo kinematics spec.

    Each feature body is the child of one actuated one-DoF joint.  Exported
    parent transforms are relative to the nearest upstream feature body, so
    any fixed MuJoCo bodies between them are folded into ``joint_origin_*``.
    """

    def __init__(self, kinematics_path: str | Path):
        super().__init__()
        path = resolve_repo_path(kinematics_path).resolve()
        spec = json.loads(path.read_text(encoding="utf-8"))
        if spec.get("quat_order") != "wxyz":
            raise ValueError(f"Kinematics spec must use wxyz quaternions: {path}")
        self.kinematics_path = str(path)
        self.robot_name = str(spec["robot_name"])
        self.root_link = str(spec["root_link"])
        self.body_order = tuple(str(name) for name in spec["body_order"])
        self.feature_body_names = tuple(
            str(name) for name in spec.get("feature_body_names", self.body_order[1:])
        )
        self.joint_order = tuple(str(name) for name in spec["joint_order"])
        self.body_name_to_index = {str(key): int(value) for key, value in spec["body_name_to_index"].items()}
        self.joint_name_to_qpos_index = {
            str(key): int(value) for key, value in spec["joint_name_to_qpos_index"].items()
        }
        self.qpos_dim = int(spec.get("qpos_dim", 7 + len(self.joint_order)))
        if self.qpos_dim != 7 + len(self.joint_order):
            raise ValueError(
                f"Expected free-root qpos_dim=7+num_joints, got {self.qpos_dim} and {len(self.joint_order)}"
            )
        if len(self.feature_body_names) != len(self.joint_order):
            raise ValueError("feature_body_names must contain one child body per actuated joint")
        if self.body_order != (self.root_link, *self.feature_body_names):
            raise ValueError("body_order must be [root_link, *feature_body_names]")

        parent = [int(value) for value in spec["parent_body_indices"]]
        child = [int(value) for value in spec["child_body_indices"]]
        if len(parent) != self.num_joints or len(child) != self.num_joints:
            raise ValueError("parent/child arrays must match joint count")
        self._parent_body_indices_py = tuple(parent)
        self._child_body_indices_py = tuple(child)
        self.register_buffer("parent_body_indices", torch.tensor(parent, dtype=torch.long), persistent=False)
        self.register_buffer("child_body_indices", torch.tensor(child, dtype=torch.long), persistent=False)
        self.register_buffer(
            "joint_axes", torch.tensor(spec["joint_axes"], dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "joint_origin_xyz",
            torch.tensor(spec["joint_origin_xyz"], dtype=torch.float32),
            persistent=False,
        )
        if "joint_origin_quat_wxyz" in spec:
            origin_quat = torch.tensor(spec["joint_origin_quat_wxyz"], dtype=torch.float32)
            origin_rot = quaternion_to_matrix(F.normalize(origin_quat, dim=-1))
        else:
            origin_rpy = torch.tensor(spec["joint_origin_rpy"], dtype=torch.float32)
            origin_rot = euler_angles_to_matrix(origin_rpy[..., [2, 1, 0]], convention="ZYX")
        self.register_buffer("joint_origin_rot", origin_rot, persistent=False)
        self.register_buffer(
            "joint_anchor_xyz",
            torch.tensor(spec.get("joint_anchor_xyz", [[0.0, 0.0, 0.0]] * self.num_joints), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "joint_lower_limits",
            torch.tensor(spec["joint_lower_limits"], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "joint_upper_limits",
            torch.tensor(spec["joint_upper_limits"], dtype=torch.float32),
            persistent=False,
        )

        sole_proxies = list(spec.get("sole_proxies", []))
        self.register_buffer(
            "sole_proxy_body_indices",
            torch.tensor([int(item["feature_body_index"]) for item in sole_proxies], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "sole_proxy_local_positions",
            torch.tensor([item["local_position"] for item in sole_proxies], dtype=torch.float32).reshape(-1, 3),
            persistent=False,
        )
        self.register_buffer(
            "sole_proxy_radii",
            torch.tensor([float(item.get("radius", 0.0)) for item in sole_proxies], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "sole_proxy_foot_ids",
            torch.tensor([int(item.get("foot_id", 0)) for item in sole_proxies], dtype=torch.long),
            persistent=False,
        )
        self.evaluation_proxies = tuple(spec.get("evaluation_proxies", ()))

    @property
    def num_joints(self) -> int:
        return len(self.joint_order)

    @property
    def num_bodies(self) -> int:
        return len(self.body_order)

    @property
    def num_feature_bodies(self) -> int:
        return len(self.feature_body_names)

    @staticmethod
    def body_quat_wxyz_to_matrix(body_quat_wxyz: torch.Tensor) -> torch.Tensor:
        return quaternion_to_matrix(F.normalize(body_quat_wxyz, dim=-1))

    @staticmethod
    def matrix_to_body_quat_wxyz(matrix: torch.Tensor) -> torch.Tensor:
        return standardize_quaternion(F.normalize(matrix_to_quaternion(matrix), dim=-1))

    def clamp_joint_positions(self, joint_pos: torch.Tensor) -> torch.Tensor:
        return joint_pos.clamp(
            min=self.joint_lower_limits.to(joint_pos),
            max=self.joint_upper_limits.to(joint_pos),
        )

    def _split_qpos(
        self, qpos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if qpos.shape[-1] != self.qpos_dim:
            raise ValueError(f"Expected qpos last dim {self.qpos_dim}, got {qpos.shape}")
        root_pos = qpos[..., :3]
        root_quat = standardize_quaternion(F.normalize(qpos[..., 3:7], dim=-1))
        return root_pos, root_quat, quaternion_to_matrix(root_quat), qpos[..., 7:]

    def _forward_body_pos_rot(self, qpos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        root_pos, _root_quat, root_rot, joints = self._split_qpos(qpos)
        batch_ndim = qpos.ndim - 1
        axes = self.joint_axes.to(qpos)
        origins = self.joint_origin_xyz.to(qpos)
        origin_rotations = self.joint_origin_rot.to(qpos)
        anchors = self.joint_anchor_xyz.to(qpos)
        body_positions: list[torch.Tensor | None] = [None] * self.num_bodies
        body_rotations: list[torch.Tensor | None] = [None] * self.num_bodies
        body_positions[0] = root_pos
        body_rotations[0] = root_rot
        for joint_index in range(self.num_joints):
            parent_index = self._parent_body_indices_py[joint_index]
            child_index = self._child_body_indices_py[joint_index]
            parent_pos = body_positions[parent_index]
            parent_rot = body_rotations[parent_index]
            if parent_pos is None or parent_rot is None:
                raise RuntimeError(
                    f"Kinematics are not topologically ordered at joint {self.joint_order[joint_index]}"
                )
            angle = joints[..., joint_index : joint_index + 1]
            axis = _view_for_batch(axes[joint_index], batch_ndim)
            joint_rot = axis_angle_to_matrix(axis * angle)
            origin_rot = _view_for_batch(origin_rotations[joint_index], batch_ndim)
            origin = _view_for_batch(origins[joint_index], batch_ndim)
            anchor = _view_for_batch(anchors[joint_index], batch_ndim)
            rotated_anchor = (joint_rot @ anchor.unsqueeze(-1)).squeeze(-1)
            local_position = origin + (origin_rot @ (anchor - rotated_anchor).unsqueeze(-1)).squeeze(-1)
            child_pos = parent_pos + (parent_rot @ local_position.unsqueeze(-1)).squeeze(-1)
            child_rot = parent_rot @ origin_rot @ joint_rot
            body_positions[child_index] = child_pos
            body_rotations[child_index] = child_rot
        if any(value is None for value in body_positions) or any(value is None for value in body_rotations):
            raise RuntimeError("Forward kinematics did not resolve every feature body")
        return (
            torch.stack([value for value in body_positions if value is not None], dim=-2),
            torch.stack([value for value in body_rotations if value is not None], dim=-3),
        )

    def forward_body_positions(self, qpos: torch.Tensor) -> torch.Tensor:
        return self._forward_body_pos_rot(qpos)[0]

    def forward_kinematics(self, qpos: torch.Tensor) -> dict[str, torch.Tensor]:
        positions, rotations = self._forward_body_pos_rot(qpos)
        return {
            "body_pos_w": positions,
            "body_quat_w": self.matrix_to_body_quat_wxyz(rotations),
        }

    def forward_kinematics_full(self, qpos: torch.Tensor) -> dict[str, torch.Tensor]:
        positions, rotations = self._forward_body_pos_rot(qpos)
        root_pos, _root_quat, root_rot, joints = self._split_qpos(qpos)
        del root_pos, root_rot, joints
        batch_ndim = qpos.ndim - 1
        axes = self.joint_axes.to(qpos)
        origins = self.joint_origin_xyz.to(qpos)
        origin_rotations = self.joint_origin_rot.to(qpos)
        joint_positions = []
        joint_axes = []
        for joint_index, parent_index in enumerate(self._parent_body_indices_py):
            parent_pos = positions[..., parent_index, :]
            parent_rot = rotations[..., parent_index, :, :]
            origin = _view_for_batch(origins[joint_index], batch_ndim)
            origin_rot = _view_for_batch(origin_rotations[joint_index], batch_ndim)
            anchor = _view_for_batch(self.joint_anchor_xyz.to(qpos)[joint_index], batch_ndim)
            axis = _view_for_batch(axes[joint_index], batch_ndim)
            joint_positions.append(
                parent_pos
                + (parent_rot @ (origin + (origin_rot @ anchor.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)).squeeze(-1)
            )
            joint_axes.append(
                F.normalize((parent_rot @ origin_rot @ axis.unsqueeze(-1)).squeeze(-1), dim=-1)
            )
        return {
            "body_pos_w": positions,
            "body_rot_w": rotations,
            "body_quat_w": self.matrix_to_body_quat_wxyz(rotations),
            "joint_pos_w": torch.stack(joint_positions, dim=-2),
            "joint_axis_w": torch.stack(joint_axes, dim=-2),
        }

    def get_sole_proxy_points(
        self,
        body_pos_w: torch.Tensor,
        body_quat_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.sole_proxy_body_indices.numel() == 0:
            raise ValueError(f"No sole proxies configured for {self.robot_name}")
        body_dim = body_pos_w.ndim - 2
        indices = self.sole_proxy_body_indices.to(device=body_pos_w.device)
        offsets = self.sole_proxy_local_positions.to(body_pos_w)
        positions = body_pos_w.index_select(body_dim, indices)
        rotations = quaternion_to_matrix(body_quat_w.index_select(body_dim, indices))
        offsets = offsets.view(*([1] * (body_pos_w.ndim - 2)), *offsets.shape)
        points = positions + (rotations @ offsets.unsqueeze(-1)).squeeze(-1)
        return points, self.sole_proxy_radii.to(body_pos_w)
