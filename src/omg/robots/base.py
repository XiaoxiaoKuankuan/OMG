from __future__ import annotations

from typing import Protocol

import torch


class RobotKinematics(Protocol):
    """Structural interface consumed by the generic motion representation."""

    robot_name: str
    qpos_dim: int
    joint_order: tuple[str, ...]
    body_order: tuple[str, ...]
    feature_body_names: tuple[str, ...]
    joint_name_to_qpos_index: dict[str, int]

    @property
    def num_joints(self) -> int: ...

    @property
    def num_bodies(self) -> int: ...

    @property
    def num_feature_bodies(self) -> int: ...

    def forward_kinematics(self, qpos: torch.Tensor) -> dict[str, torch.Tensor]: ...

    def forward_body_positions(self, qpos: torch.Tensor) -> torch.Tensor: ...

    def clamp_joint_positions(self, joint_pos: torch.Tensor) -> torch.Tensor: ...
