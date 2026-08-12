from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _planner_connect(bind: str, explicit: str | None) -> str:
    if explicit:
        return str(explicit)
    parsed = urlparse(str(bind))
    if parsed.scheme != "tcp" or parsed.port is None:
        raise ValueError(
            "--planner-connect is required when --planner-bind is not a tcp URI"
        )
    host = parsed.hostname or "127.0.0.1"
    if host in {"0.0.0.0", "*", "::"}:
        host = "127.0.0.1"
    return f"tcp://{host}:{parsed.port}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the persistent BUMI ONNX planner and native BUMI→GMT bridge "
            "as one supervised runtime."
        )
    )
    parser.add_argument("--diffusion-onnx", required=True)
    parser.add_argument("--representation-stats-path", required=True)
    parser.add_argument(
        "--kinematics-path", default="assets/robots/bumi/bumi_kinematics.json"
    )
    parser.add_argument("--text-encoder-model", required=True)
    parser.add_argument("--gmt-policy-onnx", required=True)
    parser.add_argument("--providers", default="CUDAExecutionProvider,CPUExecutionProvider")
    parser.add_argument("--planner-bind", default="tcp://127.0.0.1:5571")
    parser.add_argument("--planner-connect", default=None)
    parser.add_argument("--command-bind", default="tcp://127.0.0.1:5581")
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--cfg-text-scale", type=float, default=2.5)
    parser.add_argument("--cfg-audio-scale", type=float, default=2.5)
    parser.add_argument("--torch-device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tensorrt-fp16", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--dit-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--redis-host", default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--redis-key", default="gmt_online_frame_bumi")
    parser.add_argument(
        "--redis-ack-key",
        default=None,
        help="GMT trajectory ACK key (default: <redis-key>_ack)",
    )
    parser.add_argument("--redis-ack-poll-ms", type=float, default=5.0)
    parser.add_argument("--audio-ack-timeout-ms", type=float, default=2000.0)
    parser.add_argument(
        "--redis-lowstate-key",
        default=None,
        help="GMT LowState feedback key (default: <redis-key>_lowstate)",
    )
    parser.add_argument(
        "--redis-lowstate-poll-ms",
        type=float,
        default=5.0,
        help="Background LowState Redis polling interval",
    )
    parser.add_argument(
        "--lowstate-max-age-ms",
        type=float,
        default=200.0,
        help="Pause new replans when measured feedback is older than this",
    )
    parser.add_argument("--redis-ttl-ms", type=_positive_int, default=500)
    parser.add_argument("--tracker-fps", type=float, default=50.0)
    parser.add_argument("--history-fps", type=float, default=30.0)
    parser.add_argument("--history-frames", type=_positive_int, default=10)
    parser.add_argument(
        "--history-source",
        choices=["reference", "lowstate", "hybrid"],
        default="reference",
        help=(
            "Planner history source: reference uses emitted motion, lowstate uses "
            "measured IMU/joints, and hybrid progressively blends all history "
            "frames toward measurements; root xyz always remains reference"
        ),
    )
    parser.add_argument(
        "--hybrid-history-beta",
        default="0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80,1.00",
        help=(
            "Comma-separated oldest-to-newest Hybrid weights; count must match "
            "--history-frames and the final value must be 1.0"
        ),
    )
    parser.add_argument(
        "--hybrid-max-rotation-error-deg",
        type=float,
        default=25.0,
        help="Geodesic cap for each reference-to-measured root rotation error",
    )
    parser.add_argument(
        "--hybrid-max-tilt-error-deg",
        type=float,
        default=20.0,
        help="Pause replanning above this reference/measured body-up angle",
    )
    parser.add_argument(
        "--hybrid-hard-root-tilt-deg",
        type=float,
        default=45.0,
        help="Pause replanning when the measured root exceeds this world tilt",
    )
    parser.add_argument("--planner-frames", type=_positive_int, default=60)
    parser.add_argument("--replan-remaining-frames", type=int, default=60)
    parser.add_argument(
        "--audio-tail-silence-dbfs", type=float, default=-50.0
    )
    parser.add_argument(
        "--audio-tail-silence-min-seconds", type=float, default=0.5
    )
    parser.add_argument(
        "--audio-tail-analysis-window-ms", type=float, default=20.0
    )
    parser.add_argument("--condition-audio-step-frames", type=int, default=None)
    parser.add_argument("--play-audio", action="store_true")
    parser.add_argument("--ffplay", default="/usr/bin/ffplay")
    parser.add_argument("--sim-stream-bind", default=None)
    parser.add_argument("--sim-stream-fps", type=float, default=20.0)
    parser.add_argument("--sim-stream-width", type=_positive_int, default=1280)
    parser.add_argument("--sim-stream-height", type=_positive_int, default=720)
    parser.add_argument("--sim-camera-view", default="iso")
    parser.add_argument("--sim-follow-mode", default="xy")
    parser.add_argument("--status-jsonl", default=None)
    parser.add_argument("--planner-log-jsonl", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--num-frames", type=int, default=0)
    parser.add_argument("--startup-grace-seconds", type=float, default=1.0)
    return parser.parse_args()


def _append_option(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend((flag, str(value)))


def _planner_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "omg.cli.realtime.planner_server",
        "--bind",
        str(args.planner_bind),
        "--diffusion-onnx",
        str(args.diffusion_onnx),
        "--num-frames",
        str(args.planner_frames),
        "--providers",
        str(args.providers),
        "--text-encoder-model",
        str(args.text_encoder_model),
        "--representation-stats-path",
        str(args.representation_stats_path),
        "--kinematics-path",
        str(args.kinematics_path),
        "--torch-device",
        str(args.torch_device),
        "--seed",
        str(args.seed),
        "--tensorrt-fp16" if args.tensorrt_fp16 else "--no-tensorrt-fp16",
        "--dit-cache" if args.dit_cache else "--no-dit-cache",
    ]
    _append_option(command, "--cfg-scale", args.cfg_scale)
    _append_option(command, "--cfg-text-scale", args.cfg_text_scale)
    _append_option(command, "--cfg-audio-scale", args.cfg_audio_scale)
    _append_option(command, "--log-jsonl", args.planner_log_jsonl)
    return command


def _bridge_command(args: argparse.Namespace, connect: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "omg.cli.realtime.omg_bumi_gmt_bridge",
        "--connect",
        connect,
        "--command-bind",
        str(args.command_bind),
        "--gmt-policy-onnx",
        str(args.gmt_policy_onnx),
        "--kinematics-path",
        str(args.kinematics_path),
        "--tracker-fps",
        str(args.tracker_fps),
        "--history-fps",
        str(args.history_fps),
        "--history-frames",
        str(args.history_frames),
        "--history-source",
        str(args.history_source),
        "--hybrid-history-beta",
        str(args.hybrid_history_beta),
        "--hybrid-max-rotation-error-deg",
        str(args.hybrid_max_rotation_error_deg),
        "--hybrid-max-tilt-error-deg",
        str(args.hybrid_max_tilt_error_deg),
        "--hybrid-hard-root-tilt-deg",
        str(args.hybrid_hard_root_tilt_deg),
        "--planner-frames",
        str(args.planner_frames),
        "--replan-remaining-frames",
        str(args.replan_remaining_frames),
        "--audio-tail-silence-dbfs",
        str(args.audio_tail_silence_dbfs),
        "--audio-tail-silence-min-seconds",
        str(args.audio_tail_silence_min_seconds),
        "--audio-tail-analysis-window-ms",
        str(args.audio_tail_analysis_window_ms),
        "--redis-host",
        str(args.redis_host),
        "--redis-port",
        str(args.redis_port),
        "--redis-db",
        str(args.redis_db),
        "--redis-key",
        str(args.redis_key),
        "--redis-ack-poll-ms",
        str(args.redis_ack_poll_ms),
        "--audio-ack-timeout-ms",
        str(args.audio_ack_timeout_ms),
        "--redis-lowstate-poll-ms",
        str(args.redis_lowstate_poll_ms),
        "--lowstate-max-age-ms",
        str(args.lowstate_max_age_ms),
        "--redis-ttl-ms",
        str(args.redis_ttl_ms),
        "--ffplay",
        str(args.ffplay),
    ]
    _append_option(
        command,
        "--condition-audio-step-frames",
        args.condition_audio_step_frames,
    )
    _append_option(command, "--redis-ack-key", args.redis_ack_key)
    _append_option(command, "--redis-lowstate-key", args.redis_lowstate_key)
    _append_option(command, "--status-jsonl", args.status_jsonl)
    _append_option(command, "--output", args.output)
    if args.play_audio:
        command.append("--play-audio")
    if args.sim_stream_bind is not None:
        command.extend(
            (
                "--sim-stream-bind",
                str(args.sim_stream_bind),
                "--sim-stream-fps",
                str(args.sim_stream_fps),
                "--sim-stream-width",
                str(args.sim_stream_width),
                "--sim-stream-height",
                str(args.sim_stream_height),
                "--sim-camera-view",
                str(args.sim_camera_view),
                "--sim-follow-mode",
                str(args.sim_follow_mode),
            )
        )
    if args.continuous:
        command.append("--continuous")
    else:
        command.extend(("--num-frames", str(args.num_frames)))
    return command


def _stop_process(process: subprocess.Popen[bytes] | None, *, name: str) -> None:
    if process is None or process.poll() is not None:
        return
    print(f"[BUMI GMT runtime] stopping {name} pid={process.pid}", flush=True)
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=2.0)


def main() -> None:
    args = _parse_args()
    if not args.continuous and int(args.num_frames) <= 0:
        raise ValueError("Use --continuous or provide a positive --num-frames")
    for label, value in (
        ("diffusion ONNX", args.diffusion_onnx),
        ("BUMI stats", args.representation_stats_path),
        ("BUMI kinematics", args.kinematics_path),
        ("text encoder", args.text_encoder_model),
        ("GMT policy ONNX", args.gmt_policy_onnx),
    ):
        if not Path(value).expanduser().exists():
            raise FileNotFoundError(f"{label} does not exist: {value}")
    connect = _planner_connect(args.planner_bind, args.planner_connect)
    planner: subprocess.Popen[bytes] | None = None
    bridge: subprocess.Popen[bytes] | None = None
    try:
        planner_cmd = _planner_command(args)
        print("[BUMI GMT runtime] starting persistent ONNX/T5 Planner", flush=True)
        planner = subprocess.Popen(planner_cmd, start_new_session=True)
        deadline = time.monotonic() + max(0.0, float(args.startup_grace_seconds))
        while time.monotonic() < deadline:
            if planner.poll() is not None:
                raise RuntimeError(
                    f"Planner exited during startup with code {planner.returncode}"
                )
            time.sleep(0.05)
        bridge_cmd = _bridge_command(args, connect)
        print(
            "[BUMI GMT runtime] starting native Bridge; idle/stand never request Planner",
            flush=True,
        )
        bridge = subprocess.Popen(bridge_cmd, start_new_session=True)
        while True:
            planner_code = planner.poll()
            bridge_code = bridge.poll()
            if planner_code is not None or bridge_code is not None:
                if bridge_code == 0 and planner_code is None:
                    return
                raise RuntimeError(
                    "BUMI GMT child exited: "
                    f"planner={planner_code}, bridge={bridge_code}"
                )
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("[BUMI GMT runtime] interrupted", flush=True)
    finally:
        _stop_process(bridge, name="Bridge")
        _stop_process(planner, name="Planner")


if __name__ == "__main__":
    main()
