import torch

from omg.motion.representation import BumiMotionRepresentation


def test_bumi_encode_decode_preserves_qpos_and_fk() -> None:
    representation = BumiMotionRepresentation(num_prev_states=2, canonical_frame_idx=1)
    qpos = representation.get_default_prev_qpos(1, torch.device("cpu"), torch.float32)
    qpos = qpos[:, :1].expand(-1, 5, -1).clone()
    qpos[:, :, 0] += torch.linspace(0.0, 0.1, 5)
    qpos[:, :, 7] += torch.linspace(0.0, 0.2, 5)
    fk = representation.kinematics.forward_kinematics(qpos)
    prev_qpos = qpos[:, :2]
    prev_fk = representation.kinematics.forward_kinematics(prev_qpos)
    fps = torch.tensor([30.0])
    history, anchor_pos, anchor_quat = representation.codec.prev_state_features_from_history(
        prev_qpos, prev_fk["body_pos_w"], prev_fk["body_quat_w"], fps=fps
    )
    components = representation.codec.canonicalize(
        qpos,
        fk["body_pos_w"],
        fk["body_quat_w"],
        anchor_root_pos=anchor_pos,
        anchor_root_quat=anchor_quat,
        fps=fps,
        valid_mask=torch.ones(1, 5, dtype=torch.bool),
    )
    features = representation.codec.assemble_features(components)
    decoded = representation.decode(representation.normalize_features(features))
    reconstructed = representation.compose_qpos(decoded, anchor_pos, anchor_quat)
    reconstructed_fk = representation.kinematics.forward_kinematics(reconstructed)
    torch.testing.assert_close(reconstructed[..., :3], qpos[..., :3], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(reconstructed[..., 7:], qpos[..., 7:], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        reconstructed_fk["body_pos_w"], fk["body_pos_w"], atol=2e-5, rtol=2e-5
    )
    assert history.shape == (1, 2, 93)
