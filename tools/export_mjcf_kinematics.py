#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


def _name(model: mujoco.MjModel, object_type, object_id: int) -> str:
    value = mujoco.mj_id2name(model, object_type, int(object_id))
    if not value:
        raise ValueError(f"MuJoCo object {object_type}/{object_id} has no name")
    return str(value)


def _actuated_joints(model: mujoco.MjModel) -> list[int]:
    joint_ids = []
    for actuator_id in range(model.nu):
        if int(model.actuator_trntype[actuator_id]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            raise ValueError(f"Actuator {actuator_id} is not a joint transmission")
        joint_ids.append(int(model.actuator_trnid[actuator_id, 0]))
    if len(set(joint_ids)) != len(joint_ids):
        raise ValueError("Actuators must map one-to-one to joints")
    return sorted(joint_ids, key=lambda joint_id: int(model.jnt_qposadr[joint_id]))


def _nearest_feature_ancestor(
    model: mujoco.MjModel,
    body_id: int,
    feature_body_ids: set[int],
    root_body_id: int,
) -> int:
    parent = int(model.body_parentid[body_id])
    while parent not in feature_body_ids and parent != root_body_id:
        if parent == 0:
            raise ValueError(f"Body {_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)} is outside root subtree")
        parent = int(model.body_parentid[parent])
    return parent


def _relative_pose(data: mujoco.MjData, parent: int, child: int) -> tuple[np.ndarray, np.ndarray]:
    parent_rot = np.asarray(data.xmat[parent]).reshape(3, 3)
    child_rot = np.asarray(data.xmat[child]).reshape(3, 3)
    position = parent_rot.T @ (np.asarray(data.xpos[child]) - np.asarray(data.xpos[parent]))
    rotation = parent_rot.T @ child_rot
    quaternion = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return position, quaternion


def export_spec(mjcf_path: Path, *, robot_name: str | None = None) -> dict:
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    if model.nq < 8:
        raise ValueError("Expected a free-root articulated model")
    root_joint_ids = [
        joint_id
        for joint_id in range(model.njnt)
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    if len(root_joint_ids) != 1:
        raise ValueError(f"Expected one free joint, got {root_joint_ids}")
    root_body_id = int(model.jnt_bodyid[root_joint_ids[0]])
    joint_ids = _actuated_joints(model)
    if model.nq != 7 + len(joint_ids):
        raise ValueError(f"Expected nq=7+nu, got nq={model.nq} actuated={len(joint_ids)}")
    child_body_ids = [int(model.jnt_bodyid[joint_id]) for joint_id in joint_ids]
    if len(set(child_body_ids)) != len(child_body_ids):
        raise ValueError("Each feature joint must have a distinct child body")
    feature_body_ids = set(child_body_ids)
    body_ids = [root_body_id, *child_body_ids]
    body_index = {body_id: index for index, body_id in enumerate(body_ids)}

    qpos = np.asarray(model.qpos0).copy()
    qpos[7:] = 0.0
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    parent_indices = []
    child_indices = []
    origins = []
    origin_quaternions = []
    for joint_index, (joint_id, child_body_id) in enumerate(zip(joint_ids, child_body_ids, strict=True)):
        parent_body_id = _nearest_feature_ancestor(
            model, child_body_id, feature_body_ids, root_body_id
        )
        if parent_body_id not in body_index:
            raise ValueError("Failed to fold fixed-body parent transform")
        position, quaternion = _relative_pose(data, parent_body_id, child_body_id)
        parent_indices.append(body_index[parent_body_id])
        child_indices.append(joint_index + 1)
        origins.append(position.tolist())
        origin_quaternions.append(quaternion.tolist())

    proxy_names = (
        "head_sphere",
        "left_hand",
        "right_hand",
        "left_foot_end_link",
        "left_toe_link",
        "right_foot_end_link",
        "right_toe_link",
    )
    evaluation_proxies = []
    sole_proxies = []
    for proxy_name in proxy_names:
        proxy_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, proxy_name))
        if proxy_id < 0:
            continue
        feature_parent = _nearest_feature_ancestor(model, proxy_id, feature_body_ids, root_body_id)
        position, quaternion = _relative_pose(data, feature_parent, proxy_id)
        item = {
            "name": proxy_name,
            "feature_body_name": _name(model, mujoco.mjtObj.mjOBJ_BODY, feature_parent),
            "feature_body_index": body_index[feature_parent],
            "local_position": position.tolist(),
            "local_quat_wxyz": quaternion.tolist(),
        }
        evaluation_proxies.append(item)
        if "foot_end" in proxy_name or "toe" in proxy_name:
            geom_ids = np.flatnonzero(np.asarray(model.geom_bodyid) == proxy_id)
            radius = float(model.geom_size[geom_ids[0], 0]) if geom_ids.size else 0.0
            sole_proxies.append(
                {
                    **item,
                    "radius": radius,
                    "foot_id": 0 if proxy_name.startswith("left") else 1,
                }
            )

    joint_names = [_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) for joint_id in joint_ids]
    feature_names = [_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) for body_id in child_body_ids]
    body_order = [_name(model, mujoco.mjtObj.mjOBJ_BODY, root_body_id), *feature_names]
    return {
        "format": "omg.mjcf_kinematics.v1",
        "robot_name": str(robot_name or mjcf_path.stem),
        "source_mjcf": mjcf_path.name,
        "root_link": body_order[0],
        "quat_order": "wxyz",
        "qpos_dim": int(model.nq),
        "qpos_layout": f"root_pos_xyz + root_quat_wxyz + joint_pos_{len(joint_ids)}",
        "body_order": body_order,
        "feature_body_names": feature_names,
        "body_name_to_index": {name: index for index, name in enumerate(body_order)},
        "joint_order": joint_names,
        "joint_name_to_qpos_index": {name: index for index, name in enumerate(joint_names)},
        "parent_body_indices": parent_indices,
        "child_body_indices": child_indices,
        "joint_axes": [np.asarray(model.jnt_axis[joint_id]).tolist() for joint_id in joint_ids],
        "joint_origin_xyz": origins,
        "joint_origin_quat_wxyz": origin_quaternions,
        "joint_anchor_xyz": [np.asarray(model.jnt_pos[joint_id]).tolist() for joint_id in joint_ids],
        "joint_lower_limits": [float(model.jnt_range[joint_id, 0]) for joint_id in joint_ids],
        "joint_upper_limits": [float(model.jnt_range[joint_id, 1]) for joint_id in joint_ids],
        "default_qpos": np.asarray(model.qpos0).tolist(),
        "evaluation_proxies": evaluation_proxies,
        "sole_proxies": sole_proxies,
    }


def copy_render_assets(source_mjcf: Path, output_mjcf: Path, *, copy_meshes: bool) -> None:
    tree = ET.parse(source_mjcf)
    root = tree.getroot()
    compiler = root.find("compiler")
    source_mesh_dir = source_mjcf.parent
    if compiler is not None and compiler.get("meshdir"):
        source_mesh_dir = (source_mjcf.parent / compiler.get("meshdir")).resolve()
        compiler.set("meshdir", "meshes")
    output_mjcf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_mjcf, encoding="unicode")
    if copy_meshes:
        mesh_output = output_mjcf.parent / "meshes"
        mesh_output.mkdir(parents=True, exist_ok=True)
        for mesh in root.findall("./asset/mesh"):
            source = source_mesh_dir / str(mesh.get("file"))
            if not source.is_file():
                raise FileNotFoundError(f"Referenced mesh not found: {source}")
            shutil.copy2(source, mesh_output / source.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export differentiable kinematics from MuJoCo MJCF.")
    parser.add_argument("--mjcf", required=True, type=Path)
    parser.add_argument("--output-mjcf", required=True, type=Path)
    parser.add_argument("--output-spec", required=True, type=Path)
    parser.add_argument(
        "--robot-name",
        default=None,
        help="Canonical OMG robot identifier (defaults to the source MJCF stem).",
    )
    parser.add_argument("--copy-meshes", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    source = args.mjcf.expanduser().resolve()
    spec = export_spec(source, robot_name=args.robot_name)
    args.output_spec.parent.mkdir(parents=True, exist_ok=True)
    args.output_spec.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    copy_render_assets(source, args.output_mjcf, copy_meshes=args.copy_meshes)
    print(json.dumps({"nq": spec["qpos_dim"], "joints": len(spec["joint_order"]), "bodies": len(spec["feature_body_names"])}, indent=2))


if __name__ == "__main__":
    main()
