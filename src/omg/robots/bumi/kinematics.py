from __future__ import annotations

from pathlib import Path

from omg.robots.generic_kinematics import GenericKinematics


class BumiKinematics(GenericKinematics):
    def __init__(
        self,
        kinematics_path: str | Path = "assets/robots/bumi/bumi_kinematics.json",
    ):
        super().__init__(kinematics_path=kinematics_path)
        if self.robot_name != "bumi":
            raise ValueError(f"BUMI kinematics spec must declare robot_name='bumi', got {self.robot_name!r}")
        if self.qpos_dim != 28 or self.num_joints != 21 or self.num_feature_bodies != 21:
            raise ValueError(
                "BUMI kinematics must define qpos_dim=28, 21 joints and 21 feature bodies; "
                f"got {self.qpos_dim}, {self.num_joints}, {self.num_feature_bodies}"
            )
