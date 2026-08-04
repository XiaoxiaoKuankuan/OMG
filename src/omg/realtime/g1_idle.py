from __future__ import annotations

import math

import numpy as np


G1_QPOS_DIM = 36
G1_MODEL_ROOT_HEIGHT_METERS = 0.793
G1_NEUTRAL_SHOULDER_OPEN_RADIANS = 0.12

# qpos36 is root position xyz + root quaternion wxyz + these 29 joints.
G1_JOINT_ORDER = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
G1_JOINT_QPOS_INDEX = {
    name: 7 + index for index, name in enumerate(G1_JOINT_ORDER)
}


def neutral_g1_idle_qpos(
    *,
    shoulder_open_radians: float = G1_NEUTRAL_SHOULDER_OPEN_RADIANS,
    yaw_degrees: float = 0.0,
) -> np.ndarray:
    """Build the explicit OMG→GMR simulation idle in G1 qpos36 order.

    G1's elbow zero does not mean an arm hanging down: its forearm extends
    along the elbow link's local +X axis.  A +pi/2 elbow angle rotates that
    forearm downward.  The BUMI3 semantic elbow calibration then maps it to
    approximately zero, which is BUMI3's hanging-arm pose.

    This is a deterministic simulation/reference pose.  Hardware operation
    must still use a robot-owner-verified ``--idle-motion``.
    """

    shoulder_open = float(shoulder_open_radians)
    if not math.isfinite(shoulder_open):
        raise ValueError("shoulder_open_radians must be finite")
    if not 0.0 <= shoulder_open <= 0.5:
        raise ValueError("shoulder_open_radians must be in [0, 0.5]")
    yaw = math.radians(float(yaw_degrees))
    if not math.isfinite(yaw):
        raise ValueError("yaw_degrees must be finite")

    qpos = np.zeros(G1_QPOS_DIM, dtype=np.float32)
    qpos[2] = np.float32(G1_MODEL_ROOT_HEIGHT_METERS)
    qpos[3] = np.float32(math.cos(0.5 * yaw))
    qpos[6] = np.float32(math.sin(0.5 * yaw))

    # Legs and waist intentionally remain at zero.  Equal and opposite
    # shoulder roll opens both arms slightly; +pi/2 makes both forearms hang.
    qpos[G1_JOINT_QPOS_INDEX["left_shoulder_roll_joint"]] = np.float32(
        shoulder_open
    )
    qpos[G1_JOINT_QPOS_INDEX["right_shoulder_roll_joint"]] = np.float32(
        -shoulder_open
    )
    qpos[G1_JOINT_QPOS_INDEX["left_elbow_joint"]] = np.float32(math.pi / 2.0)
    qpos[G1_JOINT_QPOS_INDEX["right_elbow_joint"]] = np.float32(math.pi / 2.0)
    return qpos
