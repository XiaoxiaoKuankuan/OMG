from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


def test_compose_transformer():
    config_dir = str(Path(__file__).resolve().parents[2] / "configs" / "generation")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="train", overrides=["exp=base", "logger=none", "trainer=1gpu"])
    assert cfg.model._target_.endswith("MotionGenerator")
    assert cfg.denoiser._target_.endswith("MotionTransformerDenoiser")
    assert cfg.model.text_encoder.model_name == f"{cfg.paths.repo_root}/models/t5-base-local"
    assert cfg.model.scheduler.type == "linear_warmup_cosine"
    assert cfg.model.scheduler.warmup_steps == 2000


def test_compose_300m_diffusion_only():
    config_dir = str(Path(__file__).resolve().parents[2] / "configs" / "generation")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="train", overrides=["exp=300m", "logger=none", "trainer=8gpu"])
    assert cfg.denoiser._target_.endswith("MotionTransformerDenoiser")
    assert cfg.loss.simple_root_pos == 0.0
    assert cfg.loss.seam_body_pos == 0.0
    assert cfg.model.use_audio is True
    assert cfg.model.use_human_motion is True
    assert cfg.trainer.devices == 8
    assert cfg.trainer.strategy == "ddp_find_unused_parameters_true"


def test_compose_100m_omnimodal():
    config_dir = str(Path(__file__).resolve().parents[2] / "configs" / "generation")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(
            config_name="train",
            overrides=["exp=100m_omnimodal", "data=omg_data_lerobot_omnimodal", "logger=none", "trainer=4gpu"],
        )
    dataset = cfg.data.dataset_opts.train.omg_lerobot_omnimodal_train
    assert cfg.model.use_audio is True
    assert cfg.model.use_human_motion is True
    assert dataset.use_text is True
    assert dataset.use_audio is True
    assert dataset.use_human_motion is True
    assert dataset.revision == "6e0dfbc1c5298bff14d4e2b1459ad678af0a38e7"


@pytest.mark.parametrize("experiment", ["50m_bumi", "100m_bumi", "300m_bumi"])
def test_bumi_experiments_use_full_motion_loss(experiment):
    config_dir = str(Path(__file__).resolve().parents[2] / "configs" / "generation")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(
            config_name="train",
            overrides=[f"exp={experiment}", "logger=none", "trainer=1gpu"],
        )
    assert cfg.loss.simple_root_pos > 0.0
    assert cfg.loss.body_pos_consistency > 0.0
    assert cfg.loss.terrain_penetration > 0.0
    assert cfg.loss.contact_velocity > 0.0
    assert cfg.loss.seam_body_pos > 0.0
