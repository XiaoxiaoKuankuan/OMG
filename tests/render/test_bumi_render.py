from pathlib import Path

from omg.render.mujoco import render_qpos_video
from tests.bumi_test_utils import standing_qpos


def test_bumi_mujoco_render(tmp_path) -> None:
    output = render_qpos_video(
        standing_qpos(2),
        tmp_path / "bumi.mp4",
        fps=30,
        width=320,
        height=240,
        robot_name="bumi",
        kinematics_path="assets/robots/bumi/bumi_kinematics.json",
        mjcf_path="assets/robots/bumi/bumi3.xml",
        title="BUMI Test",
    )
    assert Path(output).is_file()
    assert Path(output).stat().st_size > 0
    assert Path(output).with_suffix(".json").is_file()
