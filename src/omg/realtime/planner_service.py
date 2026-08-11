from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from omg.pipeline.condition_sequence import (
    PipelineConditionSequence,
    condition_sequence_audio,
    condition_sequence_for_plan,
    condition_sequence_human,
    condition_sequence_text,
    describe_condition_sequence,
    load_pipeline_condition_sequence,
)
from omg.pipeline import OnnxDiffusionPlanner
from omg.realtime.protocol import MotionPlanChunk, RobotStateRequest


@dataclass(frozen=True)
class RealtimePlannerConfig:
    diffusion_onnx: str | Path
    num_frames: int | None = None
    cfg_scale: float | None = None
    cfg_text_scale: float | None = None
    cfg_audio_scale: float | None = None
    cfg_human_scale: float | None = None
    providers: Sequence[str] | str | None = None
    text_encoder_model: str | None = None
    representation_stats_path: str | Path | None = None
    kinematics_path: str | Path | None = None
    torch_device: str = "auto"
    seed: int = 0
    tensorrt_fp16: bool = True
    tensorrt_engine_cache_path: str | Path | None = None
    dit_cache: bool = True
    dit_cache_threshold: float = 0.995
    dit_cache_warmup_steps: int = 4
    dit_cache_max_consecutive: int = 2
    compile_history_encoder: bool | None = None
    include_motion_features: bool = False
    audio_fps: float = 30.0
    tracker_fps: float = 50.0
    audio_feature_type: str = "current35"
    condition_audio_step_frames: int | None = None
    audio_type: str = "audio"


class RealtimeDiffusionPlannerService:
    def __init__(self, config: RealtimePlannerConfig) -> None:
        self.config = config
        self.planner = OnnxDiffusionPlanner(
            config.diffusion_onnx,
            providers=config.providers,
            text_encoder_model=config.text_encoder_model,
            representation_stats_path=config.representation_stats_path,
            kinematics_path=config.kinematics_path,
            torch_device=config.torch_device,
            seed=config.seed,
            tensorrt_fp16=config.tensorrt_fp16,
            tensorrt_engine_cache_path=config.tensorrt_engine_cache_path,
            dit_cache=config.dit_cache,
            dit_cache_threshold=config.dit_cache_threshold,
            dit_cache_warmup_steps=config.dit_cache_warmup_steps,
            dit_cache_max_consecutive=config.dit_cache_max_consecutive,
            compile_history_encoder=config.compile_history_encoder,
        )
        self._condition_sequence_cache: dict[
            tuple[str, str, float, str, int | None, str],
            tuple[PipelineConditionSequence, dict[str, int]],
        ] = {}
        self._next_plan_id = 0

    @property
    def plan_frames(self) -> int:
        return int(self.config.num_frames or self.planner.sequence_length)

    def _condition_sequence_for_request(
        self,
        request: RobotStateRequest,
    ) -> tuple[PipelineConditionSequence, dict[str, int], int, dict]:
        raw_sequence = request.metadata.get("condition_sequence")
        if raw_sequence is None or str(raw_sequence).strip() == "":
            raise ValueError("Realtime planner requests must include metadata['condition_sequence']")
        raw_sequence = str(raw_sequence)
        session_id = str(request.metadata.get("condition_session_id", "default"))
        condition_index = int(request.metadata.get("condition_index", 0))
        audio_fps = float(request.metadata.get("audio_fps", self.config.audio_fps))
        tracker_fps = float(request.metadata.get("tracker_fps", self.config.tracker_fps))
        audio_feature_type = str(request.metadata.get("audio_feature_type", self.config.audio_feature_type))
        audio_type = str(request.metadata.get("audio_type", self.config.audio_type))
        audio_step_raw = request.metadata.get("condition_audio_step_frames", self.config.condition_audio_step_frames)
        audio_step_frames = None if audio_step_raw is None else int(audio_step_raw)
        key = (session_id, raw_sequence, audio_fps, audio_feature_type, audio_step_frames, audio_type)
        cached = self._condition_sequence_cache.get(key)
        if cached is None:
            cached = (
                load_pipeline_condition_sequence(
                    raw_sequence,
                    planner=self.planner,
                    target_fps=audio_fps,
                    audio_feature_type=audio_feature_type,
                    num_frames_per_chunk=self.plan_frames,
                    audio_step_frames=audio_step_frames,
                    audio_type=audio_type,
                ),
                {},
            )
            self._condition_sequence_cache[key] = cached
        condition_sequence, timeline_starts = cached
        if len(condition_sequence) <= 0:
            raise ValueError("Realtime condition sequence is empty")
        metadata = {
            "raw": raw_sequence,
            "session_id": session_id,
            "audio_fps": audio_fps,
            "tracker_fps": tracker_fps,
            "audio_feature_type": audio_feature_type,
            "audio_type": audio_type,
            "condition_audio_step_frames": audio_step_frames,
        }
        return condition_sequence, timeline_starts, min(max(condition_index, 0), len(condition_sequence) - 1), metadata

    def _condition_inputs_for_request(
        self,
        request: RobotStateRequest,
    ) -> tuple[
        str | None,
        tuple[np.ndarray, np.ndarray] | None,
        tuple[np.ndarray, np.ndarray] | None,
        dict | None,
    ]:
        if request.metadata.get("condition_sequence") is None:
            return None, None, None, None
        condition_sequence, timeline_starts, condition_index, sequence_metadata = self._condition_sequence_for_request(
            request
        )
        chunk = condition_sequence_for_plan(condition_sequence, condition_index)
        text = condition_sequence_text(condition_sequence, "", condition_index)
        explicit_audio_elapsed = request.metadata.get(
            "condition_elapsed_tracker_frames"
        )
        if explicit_audio_elapsed is not None:
            explicit_audio_elapsed = max(0, int(explicit_audio_elapsed))
        audio_inputs = condition_sequence_audio(
            condition_sequence,
            condition_index,
            request_tracker_frame=int(request.tracker_frame),
            target_fps=float(sequence_metadata["audio_fps"]),
            tracker_fps=float(sequence_metadata["tracker_fps"]),
            num_frames=self.plan_frames,
            timeline_starts=timeline_starts,
            elapsed_tracker_frames=explicit_audio_elapsed,
        )
        human_inputs = condition_sequence_human(
            condition_sequence,
            condition_index,
            num_frames=self.plan_frames,
        )
        audio_metadata = None
        if audio_inputs is not None and chunk is not None:
            audio_features, audio_mask = audio_inputs
            audio_start_frame = int(chunk.audio_start_frame)
            if (
                chunk.audio_timeline_key is not None
                and explicit_audio_elapsed is not None
            ):
                audio_start_frame = int(
                    np.floor(
                        float(explicit_audio_elapsed)
                        * float(sequence_metadata["audio_fps"])
                        / float(sequence_metadata["tracker_fps"])
                        + 1e-9
                    )
                )
            elif chunk.audio_timeline_key is not None:
                segment_start = int(
                    timeline_starts.get(
                        chunk.audio_timeline_key,
                        int(request.tracker_frame),
                    )
                )
                elapsed_tracker_frames = max(0, int(request.tracker_frame) - segment_start)
                audio_start_frame = int(
                    np.floor(
                        float(elapsed_tracker_frames)
                        * float(sequence_metadata["audio_fps"])
                        / float(sequence_metadata["tracker_fps"])
                        + 1e-9
                    )
                )
            audio_metadata = {
                "start_frame": int(audio_start_frame),
                "end_frame": int(audio_start_frame + self.plan_frames),
                "frames": int(self.plan_frames),
                "feature_shape": list(np.asarray(audio_features).shape),
                "valid_frames": int(np.asarray(audio_mask, dtype=bool).sum()),
                "audio_fps": float(sequence_metadata["audio_fps"]),
                "tracker_fps": float(sequence_metadata["tracker_fps"]),
                "request_tracker_frame": int(request.tracker_frame),
                "condition_elapsed_tracker_frames": explicit_audio_elapsed,
            }
        human_metadata = None
        if human_inputs is not None and chunk is not None:
            human_features, human_mask = human_inputs
            human_metadata = {
                "start_frame": int(chunk.human_start_frame),
                "end_frame": int(chunk.human_start_frame + self.plan_frames),
                "frames": int(self.plan_frames),
                "feature_shape": list(np.asarray(human_features).shape),
                "valid_frames": int(np.asarray(human_mask, dtype=bool).sum()),
            }
        return text, audio_inputs, human_inputs, {
            "index": int(condition_index),
            "requested_index": int(request.metadata.get("condition_index", condition_index)),
            "held_last_chunk": bool(condition_index != int(request.metadata.get("condition_index", condition_index))),
            "chunk": None if chunk is None else chunk.describe(),
            "sequence": describe_condition_sequence(condition_sequence),
            "source": "request",
            "config": sequence_metadata,
            "audio": audio_metadata,
            "human_motion": human_metadata,
        }

    def plan(self, request: RobotStateRequest) -> MotionPlanChunk:
        if request.robot_name != self.planner.robot_name or request.state_dim != self.planner.state_dim:
            raise ValueError(
                "Realtime request robot identity does not match the ONNX planner: "
                f"request={request.robot_name}/{request.state_dim}, "
                f"planner={self.planner.robot_name}/{self.planner.state_dim}"
            )
        plan_id = self._next_plan_id
        condition_text, condition_audio, condition_human, condition_metadata = self._condition_inputs_for_request(
            request
        )
        if condition_metadata is None:
            raise ValueError("Realtime planner requests must include metadata['condition_sequence']")
        text = condition_text
        if text:
            self.planner.cache_text_conditions(text)
        audio_inputs = condition_audio
        audio_metadata = condition_metadata.get("audio") if condition_metadata is not None else None
        human_inputs = condition_human
        plan_start_wall_time = time.time()
        started = time.perf_counter()
        plan = self.planner.plan(
            seed_qpos=request.qpos_history,
            text=text,
            fps=request.history_fps,
            num_frames=self.plan_frames,
            cfg_scale=self.config.cfg_scale,
            cfg_text_scale=self.config.cfg_text_scale,
            cfg_audio_scale=self.config.cfg_audio_scale,
            cfg_human_scale=self.config.cfg_human_scale,
            audio_features=audio_inputs,
            human_motion=human_inputs,
        )
        latency = time.perf_counter() - started
        plan_end_wall_time = time.time()
        self._next_plan_id += 1
        metadata = dict(plan.metadata)
        transport = dict(request.metadata.get("realtime_transport", {}))
        transport.update(
            {
                "server_plan_start_wall_time": plan_start_wall_time,
                "server_plan_end_wall_time": plan_end_wall_time,
                "server_plan_ms": float(latency) * 1000.0,
            }
        )
        metadata.update(
            {
                "realtime_request": {
                    "request_id": request.request_id,
                    "tracker_frame": int(request.tracker_frame),
                    "buffer_remaining_frames": int(request.buffer_remaining_frames),
                    "last_plan_id": request.last_plan_id,
                    "history_fps": float(request.history_fps),
                    "history_frames": int(request.qpos_history.shape[0]),
                    "robot_name": request.robot_name,
                    "state_dim": request.state_dim,
                    "metadata": request.metadata,
                },
                "planning_latency_seconds": float(latency),
                "realtime_transport": transport,
                "realtime_audio": audio_metadata,
                "realtime_condition": condition_metadata,
            }
        )
        return MotionPlanChunk(
            qpos_36=plan.qpos,
            motion_features=plan.motion_features if self.config.include_motion_features else None,
            fps=plan.fps,
            request_id=str(request.request_id),
            plan_id=plan_id,
            request_tracker_frame=int(request.tracker_frame),
            planning_latency_seconds=latency,
            prompt=text,
            metadata=metadata,
        )

    def append_jsonl(self, path: str | Path, request: RobotStateRequest, response: MotionPlanChunk) -> None:
        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "request_id": request.request_id,
            "plan_id": response.plan_id,
            "request_tracker_frame": request.tracker_frame,
            "buffer_remaining_frames": request.buffer_remaining_frames,
            "fps": response.fps,
            "frames": int(response.qpos.shape[0]),
            "robot_name": response.robot_name,
            "state_dim": response.state_dim,
            "planning_latency_seconds": response.planning_latency_seconds,
            "prompt": response.prompt,
            "timing_ms": response.metadata.get("timing_ms", {}),
            "metadata": response.metadata,
        }
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")
