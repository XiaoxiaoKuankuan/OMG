#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from omg.robots.generic_kinematics import GenericKinematics


def validate(mjcf_path: Path, spec_path: Path, samples: int, seed: int) -> dict:
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    kinematics = GenericKinematics(spec_path).double()
    rng = np.random.default_rng(seed)
    qpos = np.repeat(np.asarray(model.qpos0, dtype=np.float64)[None], samples, axis=0)
    qpos[:, :3] += rng.uniform(-1.0, 1.0, size=(samples, 3))
    quaternion = rng.normal(size=(samples, 4))
    quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True)
    qpos[:, 3:7] = quaternion
    lower = np.asarray(kinematics.joint_lower_limits, dtype=np.float64)
    upper = np.asarray(kinematics.joint_upper_limits, dtype=np.float64)
    qpos[:, 7:] = rng.uniform(lower, upper, size=(samples, model.nq - 7))
    with torch.no_grad():
        generic = kinematics.forward_kinematics_full(torch.from_numpy(qpos))
    generic_position = generic["body_pos_w"].numpy()
    generic_rotation = generic["body_rot_w"].numpy()
    body_ids = [
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        for name in spec["body_order"]
    ]
    mujoco_position = np.zeros_like(generic_position)
    mujoco_rotation = np.zeros_like(generic_rotation)
    data = mujoco.MjData(model)
    for index in range(samples):
        data.qpos[:] = qpos[index]
        mujoco.mj_forward(model, data)
        mujoco_position[index] = data.xpos[body_ids]
        mujoco_rotation[index] = data.xmat[body_ids].reshape(-1, 3, 3)
    position_error = np.abs(generic_position - mujoco_position)
    rotation_error = np.abs(generic_rotation - mujoco_rotation)
    report = {
        "samples": samples,
        "seed": seed,
        "max_position_error_m": float(position_error.max(initial=0.0)),
        "max_rotation_matrix_error": float(rotation_error.max(initial=0.0)),
        "mean_position_error_m": float(position_error.mean()),
        "mean_rotation_matrix_error": float(rotation_error.mean()),
        "position_threshold_m": 1.0e-5,
        "rotation_matrix_threshold": 1.0e-4,
    }
    report["valid"] = bool(
        report["max_position_error_m"] < report["position_threshold_m"]
        and report["max_rotation_matrix_error"] < report["rotation_matrix_threshold"]
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generic PyTorch FK against MuJoCo.")
    parser.add_argument("--mjcf", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.mjcf.resolve(), args.spec.resolve(), args.samples, args.seed)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
