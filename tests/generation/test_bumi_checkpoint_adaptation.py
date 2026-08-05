import torch

from omg.generation.checkpoint_adapter import adapt_g1_checkpoint_to_bumi
from omg.generation.denoisers.transformer import MotionTransformerDenoiser
from omg.generation.models.motion_generator import MotionGenerator
from omg.motion.representation import BumiMotionRepresentation, G1MotionRepresentation


def _model(representation, feature_dim: int) -> MotionGenerator:
    return MotionGenerator(
        representation=representation,
        denoiser=MotionTransformerDenoiser(
            input_dim=feature_dim,
            hidden_dim=32,
            num_layers=1,
            num_heads=4,
            text_dim=16,
            dropout=0.0,
        ),
        diffusion=object(),
        loss=object(),
        text_encoder=None,
        condition_dim=32,
        use_audio=True,
        use_human_motion=True,
        log_loss_term_grad_norms=False,
    )


def test_g1_to_bumi_checkpoint_adapter_is_explicit_and_shape_safe() -> None:
    source = _model(
        G1MotionRepresentation(
            num_prev_states=2,
            canonical_frame_idx=1,
            feat_dim=125,
            rotation_representation="rot6d",
        ),
        125,
    )
    target = _model(BumiMotionRepresentation(num_prev_states=2, canonical_frame_idx=1), 93)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(0.125)
    input_before = target.denoiser.input_proj.weight.detach().clone()
    output_before = target.denoiser.output.weight.detach().clone()
    history_before = target.history_projector[1].weight.detach().clone()
    report = adapt_g1_checkpoint_to_bumi(target, source.state_dict())
    assert "denoiser.layers.0.self_attn.qkv.weight" in report.loaded
    torch.testing.assert_close(
        target.denoiser.layers[0].self_attn.qkv.weight,
        torch.full_like(target.denoiser.layers[0].self_attn.qkv.weight, 0.125),
    )
    torch.testing.assert_close(target.denoiser.input_proj.weight, input_before)
    torch.testing.assert_close(target.denoiser.output.weight, output_before)
    torch.testing.assert_close(target.history_projector[1].weight, history_before)
    skipped = {item.key: item.reason for item in report.skipped}
    assert skipped["denoiser.input_proj.weight"] == "motion_input_projection"
    assert skipped["denoiser.output.weight"] == "motion_output_projection"
    assert skipped["history_projector.1.weight"] == "history_motion_input_projection"
    assert report.unexpected == ()
