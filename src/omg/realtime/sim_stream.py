from __future__ import annotations

import json
import os
import platform
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SimStreamConfig:
    bind: str
    fps: float = 20.0
    width: int = 640
    height: int = 360
    camera_view: str = "iso"
    follow_mode: str = "xy"
    camera_distance: float = 4.5
    camera_elevation: float = -18.0
    robot_name: str = "g1"
    state_dim: int | None = None
    kinematics_path: str | Path | None = None
    urdf_path: str | Path | None = None
    mjcf_path: str | Path | None = None
    scene_preset: str = "studio"


def parse_host_port(bind: str) -> tuple[str, int]:
    text = str(bind).strip()
    if text.startswith("http://"):
        text = text[len("http://") :]
    if "/" in text:
        text = text.split("/", 1)[0]
    if text.count(":") != 1:
        raise ValueError(f"Expected HOST:PORT bind address, got {bind!r}")
    host, port_text = text.rsplit(":", 1)
    if host == "":
        host = "127.0.0.1"
    port = int(port_text)
    if port <= 0 or port > 65535:
        raise ValueError(f"Invalid port in bind address: {bind!r}")
    return host, port


class MujocoSimStream:
    def __init__(self, config: SimStreamConfig) -> None:
        os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL") or ("glfw" if platform.system() == "Darwin" else "egl")
        self.config = config
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._jpeg: bytes | None = None
        self._qpos: np.ndarray | None = None
        self._overlay_lines: list[str] = []
        self._frame_index = -1
        self._render_count = 0
        self._last_render = 0.0
        self._last_update_wall_time = 0.0
        self._closed = False

        import cv2
        import mujoco
        import torch

        from omg.render.mujoco import _make_camera, _set_data_qpos, build_mjcf, parse_urdf
        from omg.tracking.holomotion.video import camera_azimuth, draw_overlay, yaw_degrees_from_wxyz

        self._cv2 = cv2
        self._mujoco = mujoco
        self._torch = torch
        self._make_camera = _make_camera
        self._set_data_qpos = _set_data_qpos
        self._camera_azimuth = camera_azimuth
        self._draw_overlay = draw_overlay
        self._yaw_degrees_from_wxyz = yaw_degrees_from_wxyz

        robot_name = str(config.robot_name).strip().lower()
        if robot_name == "g1":
            from omg.robots.g1.kinematics import G1Kinematics

            kinematics_path = config.kinematics_path or "assets/robots/g1/g1_kinematics.json"
            urdf_path = config.urdf_path or "assets/robots/g1/g1_29dof.urdf"
            self._kinematics = G1Kinematics(kinematics_path)
            urdf_data = parse_urdf(urdf_path)
            xml_string = build_mjcf(
                urdf_data,
                offscreen_width=int(config.width),
                offscreen_height=int(config.height),
                scene_preset=str(config.scene_preset),
            )
            self._model = mujoco.MjModel.from_xml_string(xml_string)
            default_state_dim = 36
        elif robot_name == "bumi":
            from omg.robots.bumi.kinematics import BumiKinematics

            kinematics_path = config.kinematics_path or "assets/robots/bumi/bumi_kinematics.json"
            mjcf_path = Path(
                config.mjcf_path or "assets/robots/bumi/bumi3.xml"
            ).expanduser().resolve()
            if not mjcf_path.is_file():
                raise FileNotFoundError(f"BUMI MuJoCo model does not exist: {mjcf_path}")
            self._kinematics = BumiKinematics(kinematics_path)
            self._model = mujoco.MjModel.from_xml_path(str(mjcf_path))
            # MuJoCo permits updating the offscreen dimensions before the GL
            # context is created.  This avoids modifying the source MJCF and
            # removes its default 640-pixel framebuffer limit.
            self._model.vis.global_.offwidth = max(
                int(self._model.vis.global_.offwidth), int(config.width)
            )
            self._model.vis.global_.offheight = max(
                int(self._model.vis.global_.offheight), int(config.height)
            )
            default_state_dim = 28
        else:
            raise ValueError(f"Unsupported sim stream robot_name={config.robot_name!r}")
        self._robot_name = robot_name
        self._state_dim = default_state_dim if config.state_dim is None else int(config.state_dim)
        if self._state_dim != int(self._kinematics.qpos_dim):
            raise ValueError(
                f"sim stream state_dim={self._state_dim} does not match "
                f"{robot_name} qpos_dim={self._kinematics.qpos_dim}"
            )
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(self._model, height=int(config.height), width=int(config.width))
        self._joint_qposadr = {}
        for joint_name in self._kinematics.joint_order:
            joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Joint {joint_name} missing in MuJoCo model")
            self._joint_qposadr[joint_name] = int(self._model.jnt_qposadr[joint_id])

    def update(self, qpos: np.ndarray, *, frame_index: int, overlay_lines: Sequence[str] | None = None) -> None:
        min_interval = 1.0 / max(float(self.config.fps), 1e-6)
        now = time.perf_counter()
        state_dim = int(getattr(self, "_state_dim", 36))
        robot_name = str(getattr(self, "_robot_name", "g1"))
        with self._lock:
            if self._closed:
                return
            self._qpos = np.asarray(qpos, dtype=np.float32).reshape(state_dim).copy()
            # Keep the historical private field available to callers/tests
            # which monkeypatch the old G1-only renderer internals.
            if robot_name == "g1":
                self._qpos_36 = self._qpos
            self._overlay_lines = [str(line) for line in overlay_lines or ()]
            self._frame_index = int(frame_index)
            self._last_update_wall_time = time.time()
            if now - self._last_render < min_interval:
                return
            self._last_render = now
        frame = self._render_frame(qpos, overlay_lines=overlay_lines)
        ok, encoded = self._cv2.imencode(".jpg", frame[..., ::-1], [int(self._cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            raise RuntimeError("Failed to encode MuJoCo frame as JPEG")
        with self._condition:
            self._jpeg = encoded.tobytes()
            self._frame_index = int(frame_index)
            self._render_count += 1
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._renderer.close()

    def wait_jpeg(self, previous_render_count: int, timeout: float = 2.0) -> tuple[bytes | None, int, int]:
        deadline = time.perf_counter() + float(timeout)
        with self._condition:
            while not self._closed and self._render_count <= previous_render_count:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                self._condition.wait(timeout=remaining)
            return self._jpeg, self._render_count, self._frame_index

    def status(self) -> dict[str, object]:
        with self._lock:
            qpos = getattr(self, "_qpos", getattr(self, "_qpos_36", None))
            return {
                "frame_index": int(self._frame_index),
                "render_count": int(self._render_count),
                "fps": float(self.config.fps),
                "width": int(self.config.width),
                "height": int(self.config.height),
                "updated_time": float(self._last_update_wall_time),
                "robot_name": str(getattr(self, "_robot_name", "g1")),
                "state_dim": int(getattr(self, "_state_dim", 36)),
                "has_qpos": qpos is not None,
            }

    def state(self) -> dict[str, object]:
        with self._lock:
            latest = getattr(self, "_qpos", getattr(self, "_qpos_36", None))
            qpos = None if latest is None else latest.astype(float).tolist()
            robot_name = str(getattr(self, "_robot_name", "g1"))
            state_dim = int(getattr(self, "_state_dim", 36))
            result = {
                "frame_index": int(self._frame_index),
                "render_count": int(self._render_count),
                "fps": float(self.config.fps),
                "width": int(self.config.width),
                "height": int(self.config.height),
                "updated_time": float(self._last_update_wall_time),
                "overlay_lines": list(self._overlay_lines),
                "robot_name": robot_name,
                "state_dim": state_dim,
                "qpos": qpos,
            }
            if robot_name == "g1":
                result["qpos_36"] = qpos
            return result

    def _render_frame(self, qpos_value: np.ndarray, *, overlay_lines: Sequence[str] | None) -> np.ndarray:
        qpos = np.asarray(qpos_value, dtype=np.float32).reshape(self._state_dim)
        self._set_data_qpos(self._data, qpos, self._joint_qposadr, self._kinematics.joint_name_to_qpos_index)
        self._mujoco.mj_forward(self._model, self._data)
        root = qpos[:3].astype(np.float64, copy=True)
        lookat = root.copy()
        if str(self.config.follow_mode) in {"xy", "heading"}:
            lookat[2] += 0.35
        elif str(self.config.follow_mode) == "xyz":
            lookat[2] += 0.35
        elif str(self.config.follow_mode) != "none":
            raise ValueError(f"Unsupported follow_mode: {self.config.follow_mode}")
        azimuth = self._camera_azimuth(str(self.config.camera_view))
        if str(self.config.follow_mode) == "heading":
            azimuth = self._yaw_degrees_from_wxyz(qpos[3:7]) + azimuth
        camera = self._make_camera(
            self._model,
            lookat=lookat,
            distance=float(self.config.camera_distance),
            azimuth=float(azimuth),
            elevation=float(self.config.camera_elevation),
        )
        self._renderer.update_scene(self._data, camera=camera)
        return self._draw_overlay(self._renderer.render(), overlay_lines)


class SimStreamServer:
    def __init__(self, config: SimStreamConfig) -> None:
        self.stream = MujocoSimStream(config)
        self.config = config
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        host, port = parse_host_port(self.config.bind)
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/" or self.path.startswith("/index.html"):
                    self._send_index()
                    return
                if self.path.startswith("/video.mjpg"):
                    self._send_mjpeg()
                    return
                if self.path.startswith("/status"):
                    # Include overlay_lines so the web UI exposes the exact
                    # command/planner/buffer/audio state displayed in-video.
                    self._send_json(outer.stream.state())
                    return
                if self.path.startswith("/state.json"):
                    self._send_json(outer.stream.state())
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")

            def _send_index(self) -> None:
                body = (
                    "<!doctype html><html><head><meta charset='utf-8'>"
                    "<title>OMG BUMI realtime reference</title>"
                    "<style>html,body{margin:0;background:#111;color:#eee;font-family:sans-serif;}"
                    "main{display:flex;flex-direction:column;align-items:center;gap:8px;padding:10px;}"
                    "img{max-width:100%;height:auto;border:1px solid #444;}"
                    "code{color:#9fd;}</style></head><body><main>"
                    "<h2>OMG realtime reference — "
                    + str(outer.config.robot_name).upper()
                    + "</h2><img src='/video.mjpg' alt='realtime MuJoCo stream'>"
                    "<p>Gazebo displays the executed robot. This page displays the exact reference sent to GMT.</p>"
                    "<p>Status: <code id='status'>loading</code></p>"
                    "<script>setInterval(async()=>{try{let r=await fetch('/status');"
                    "document.getElementById('status').textContent=JSON.stringify(await r.json());}catch(e){}},1000);</script>"
                    "</main></body></html>"
                ).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args: object) -> None:
                print(f"[sim-stream] {self.address_string()} - {fmt % args}", flush=True)

            def _send_json(self, payload: dict[str, object]) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _send_mjpeg(self) -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=omg")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                render_count = -1
                while True:
                    jpeg, render_count, _frame_index = outer.stream.wait_jpeg(render_count)
                    if jpeg is None:
                        continue
                    try:
                        self.wfile.write(b"--omg\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return

        self._server = ThreadingHTTPServer((host, port), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"[sim-stream] serving http://{host}:{port}/", flush=True)

    def update(self, qpos: np.ndarray, *, frame_index: int, overlay_lines: Sequence[str] | None = None) -> None:
        self.stream.update(qpos, frame_index=frame_index, overlay_lines=overlay_lines)

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.stream.close()
