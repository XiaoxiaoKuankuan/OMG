__all__ = [
    "GenerationDataModule",
    "LeRobotG1MotionDataset",
    "LeRobotMotionDataset",
    "LeRobotBumiMotionDataset",
    "motion_collate_fn",
]


def __getattr__(name: str):
    if name in {"GenerationDataModule", "motion_collate_fn"}:
        from omg.data.datamodule import GenerationDataModule, motion_collate_fn

        return {"GenerationDataModule": GenerationDataModule, "motion_collate_fn": motion_collate_fn}[name]
    if name in {"LeRobotMotionDataset", "LeRobotG1MotionDataset", "LeRobotBumiMotionDataset"}:
        from omg.data.lerobot_dataset import (
            LeRobotBumiMotionDataset,
            LeRobotG1MotionDataset,
            LeRobotMotionDataset,
        )

        return {
            "LeRobotMotionDataset": LeRobotMotionDataset,
            "LeRobotG1MotionDataset": LeRobotG1MotionDataset,
            "LeRobotBumiMotionDataset": LeRobotBumiMotionDataset,
        }[name]
    raise AttributeError(name)
