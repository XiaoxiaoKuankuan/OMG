import torch

from omg.motion.representation import BumiMotionRepresentation


def test_bumi_representation_dimensions_and_default_state() -> None:
    representation = BumiMotionRepresentation(num_prev_states=2, canonical_frame_idx=1)
    assert representation.robot_name == "bumi"
    assert representation.state_dim == 28
    assert representation.feat_dim == 93
    assert len(representation.joint_names) == 21
    default = representation.get_default_prev_qpos(3, torch.device("cpu"), torch.float32)
    assert default.shape == (3, 2, 28)
    torch.testing.assert_close(default[..., 3:7].norm(dim=-1), torch.ones(3, 2))
