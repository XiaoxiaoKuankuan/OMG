from __future__ import annotations

import numpy as np
import pytest

from omg.realtime.bumi_motion_mux import BumiTrajectoryHistory
from omg.realtime.gmt_trajectory import (
    TRAJECTORY_CURRENT_INDEX,
    TRAJECTORY_FRAME_COUNT,
    GmtPolicyContract,
    GmtTrajectoryAck,
    GmtTrajectoryPacket,
    build_policy_default_idle_qpos,
    joint_order_sha256,
    make_trajectory_packet,
)
from omg.robots.bumi.kinematics import BumiKinematics


def _qpos(x: float = 0.0) -> np.ndarray:
    value = np.zeros((28,), dtype=np.float32)
    value[0] = x
    value[2] = 0.5
    value[3] = 1.0
    return value


def test_trajectory_ack_v1_roundtrip_and_rejects_wrong_size() -> None:
    ack = GmtTrajectoryAck(
        stream_id=12,
        sequence=34,
        command_revision=5,
        plan_id=6,
        received_unix_ns=7,
    )
    decoded = GmtTrajectoryAck.decode(ack.encode())
    assert decoded == ack
    with pytest.raises(ValueError, match="must contain"):
        GmtTrajectoryAck.decode(ack.encode()[:-1])


def test_trajectory_v1_roundtrip_crc_and_exact_temporal_layout() -> None:
    history = BumiTrajectoryHistory(_qpos())
    prefix = None
    for index in range(1, 16):
        prefix = history.packet_prefix(_qpos(float(index)))
    assert prefix is not None
    future = np.stack([_qpos(float(16 + i)) for i in range(99)])
    timeline = np.concatenate((prefix, future))
    names = tuple(f"joint_{index}" for index in range(21))
    packet = make_trajectory_packet(
        timeline,
        fps=50.0,
        native_to_gmt=np.arange(21),
        joint_order_hash=joint_order_sha256(names),
        stream_id=12,
        sequence=34,
        command_revision=5,
        plan_id=6,
        flags=7,
        published_unix_ns=8,
    )
    decoded = GmtTrajectoryPacket.decode(packet.encode())
    assert decoded.frames.shape == (TRAJECTORY_FRAME_COUNT, 55)
    assert decoded.current_index == TRAJECTORY_CURRENT_INDEX
    np.testing.assert_allclose(decoded.frames[:21, 0], np.arange(5.0, 26.0))
    assert np.unique(decoded.frames[:21, 0]).size == 21
    assert np.all(np.diff(decoded.frames[:21, 0]) > 0.0)
    corrupt = bytearray(packet.encode())
    corrupt[-1] ^= 1
    with pytest.raises(ValueError, match="CRC32"):
        GmtTrajectoryPacket.decode(corrupt)


def test_policy_idle_comes_from_metadata_and_is_grounded() -> None:
    import torch

    kinematics = BumiKinematics("assets/robots/bumi/bumi_kinematics.json")
    names = tuple(kinematics.joint_order)
    defaults = np.zeros((21,), dtype=np.float32)
    defaults[names.index("l_arm_roll_joint")] = 0.3
    defaults[names.index("r_arm_roll_joint")] = -0.3
    contract = GmtPolicyContract(
        path=None,  # type: ignore[arg-type]
        joint_names=names,
        default_joint_pos=defaults,
        joint_order_hash=joint_order_sha256(names),
    )
    qpos, resolved, permutation = build_policy_default_idle_qpos(contract)
    assert qpos.shape == (28,)
    assert permutation.tolist() == list(range(21))
    assert qpos[7 + names.index("l_arm_roll_joint")] == pytest.approx(0.3)
    assert qpos[7 + names.index("r_arm_roll_joint")] == pytest.approx(-0.3)
    with torch.no_grad():
        fk = resolved.forward_kinematics(torch.from_numpy(qpos).view(1, 1, -1))
        points, radii = resolved.get_sole_proxy_points(
            fk["body_pos_w"], fk["body_quat_w"]
        )
    lowest = (points[..., 2] - radii.view(1, 1, -1)).min().item()
    assert lowest == pytest.approx(0.0, abs=2e-5)
