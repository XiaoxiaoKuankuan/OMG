import torch

from omg.data.datamodule import motion_collate_fn
from omg.generation.denoisers.transformer import MotionTransformerDenoiser
from omg.generation.diffusion.guided import GuidedDiffusion
from omg.generation.losses.motion import MotionLoss
from omg.generation.models.motion_generator import MotionGenerator
from omg.motion.representation import BumiMotionRepresentation
from tests.bumi_test_utils import make_bumi_lerobot_dataset, open_bumi_dataset


def _zero_motion_loss() -> MotionLoss:
    return MotionLoss(
        simple_root_pos=0.0,
        simple_root_rot=0.0,
        simple_joint_dof=0.0,
        simple_body_pos=0.0,
        body_pos_consistency=0.0,
        vel_root_pos=0.0,
        vel_root_rot=0.0,
        vel_joint_dof=0.0,
        fk_body_pos=0.0,
        fk_body_rot=0.0,
        body_vel=0.0,
        terrain_penetration=0.0,
        contact_height=0.0,
        contact_velocity=0.0,
        seam_root_pos=0.0,
        seam_root_rot=0.0,
        seam_joint_dof=0.0,
        seam_body_pos=0.0,
    )


def _small_bumi_model() -> MotionGenerator:
    representation = BumiMotionRepresentation(
        num_prev_states=2,
        canonical_frame_idx=1,
        sequence_length=4,
    )
    return MotionGenerator(
        representation=representation,
        denoiser=MotionTransformerDenoiser(
            input_dim=93,
            hidden_dim=32,
            num_layers=1,
            num_heads=4,
            text_dim=16,
            dropout=0.0,
        ),
        diffusion=GuidedDiffusion(timesteps=10, test_timestep_respacing="2"),
        loss=_zero_motion_loss(),
        text_encoder=None,
        condition_dim=32,
        history_mask_prob=0.0,
        text_mask_prob=0.0,
        log_loss_term_grad_norms=False,
    )


def test_bumi_single_train_step_and_generation_are_finite(tmp_path) -> None:
    bundle = make_bumi_lerobot_dataset(tmp_path / "dataset")
    dataset = open_bumi_dataset(bundle, omnimodal=False)
    batch = motion_collate_fn([dataset[0], dataset[1]])
    model = _small_bumi_model()
    model.train()
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    model.eval()
    generated = model.generate(batch, num_frames=4, cfg_scale=1.0)
    assert generated["qpos"].shape == (2, 4, 28)
    assert torch.isfinite(generated["qpos"]).all()
    assert "qpos_36" not in generated
