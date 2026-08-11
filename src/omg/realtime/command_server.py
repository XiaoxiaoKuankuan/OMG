from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from omg.realtime.dynamic_condition import DynamicConditionController


def _zmq() -> Any:
    try:
        import zmq
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("Dynamic command server requires pyzmq. Install omg[realtime].") from exc
    return zmq


@dataclass(frozen=True)
class CommandServerConfig:
    bind: str
    poll_timeout_ms: int = 100
    linger_ms: int = 0


class DynamicCommandServer:
    def __init__(
        self,
        config: CommandServerConfig,
        controller: DynamicConditionController,
        status_callback: Callable[[dict[str, Any]], None] | None = None,
        status_provider: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        if not str(config.bind).strip():
            raise ValueError("Command server bind URI must be non-empty")
        if int(config.poll_timeout_ms) <= 0:
            raise ValueError("Command server poll_timeout_ms must be positive")
        self.config = config
        self.controller = controller
        self.status_callback = status_callback
        self.status_provider = status_provider
        self._stop = threading.Event()
        self._started = threading.Event()
        self._thread: threading.Thread | None = None
        self._startup_error: BaseException | None = None

    @property
    def is_alive(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Dynamic command server has already been started")
        self._stop.clear()
        self._started.clear()
        self._startup_error = None
        self._thread = threading.Thread(
            target=self._run,
            name="omg-dynamic-command-server",
            daemon=True,
        )
        self._thread.start()
        if not self._started.wait(timeout=5.0):
            self.close()
            raise RuntimeError("Timed out starting dynamic command server")
        if self._startup_error is not None:
            error = self._startup_error
            self.close()
            raise RuntimeError(
                f"Failed to bind dynamic command server at {self.config.bind}: {error}"
            ) from error

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)
            if thread.is_alive():
                raise RuntimeError("Dynamic command server thread did not stop")
        self._thread = None

    def _notify(self, event: dict[str, Any]) -> None:
        if self.status_callback is None:
            return
        try:
            self.status_callback(dict(event))
        except Exception as exc:  # pragma: no cover - logging must not stop control
            print(f"[dynamic-condition] status callback failed: {exc}", flush=True)

    @staticmethod
    def _error_response(exc: BaseException) -> dict[str, Any]:
        return {
            "ok": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }

    def _handle(self, request: object) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ValueError("Command request must be a JSON object")
        command_type = request.get("type")
        if isinstance(command_type, str) and command_type.strip().lower() == "status":
            response = {"ok": True, "active": self.controller.active_status()}
            if self.status_provider is not None:
                response["runtime"] = dict(self.status_provider())
            return response
        command = self.controller.accept_command(request)
        active = self.controller.active_status()
        event = {
            "kind": "command_accepted",
            "command_id": command.command_id,
            "command_type": command.command_type,
            "command_revision": int(command.revision),
            "condition_session_id": active["condition_session_id"],
            "audio_duration_seconds": active["audio_duration_seconds"],
            "audio_source_duration_seconds": active[
                "audio_source_duration_seconds"
            ],
            "audio_effective_duration_seconds": active[
                "audio_effective_duration_seconds"
            ],
            "audio_trailing_silence_seconds": active[
                "audio_trailing_silence_seconds"
            ],
            "audio_start_tracker_frame": active["audio_start_tracker_frame"],
            "audio_end_tracker_frame": active["audio_end_tracker_frame"],
            "stale_command": self.controller.last_requested_revision != command.revision,
            "active": active,
        }
        self._notify(event)
        return {
            "ok": True,
            "accepted": True,
            "switch_policy": "next_replan",
            "active": active,
        }

    def _run(self) -> None:
        zmq = _zmq()
        socket = None
        try:
            context = zmq.Context.instance()
            socket = context.socket(zmq.REP)
            socket.setsockopt(zmq.LINGER, int(self.config.linger_ms))
            socket.bind(str(self.config.bind))
        except BaseException as exc:
            self._startup_error = exc
            self._started.set()
            if socket is not None:
                socket.close(0)
            return
        self._started.set()
        try:
            while not self._stop.is_set():
                try:
                    if (int(socket.poll(int(self.config.poll_timeout_ms))) & zmq.POLLIN) == 0:
                        continue
                    raw = socket.recv()
                    try:
                        request = json.loads(raw.decode("utf-8"))
                        response = self._handle(request)
                    except BaseException as exc:
                        response = self._error_response(exc)
                    socket.send_json(response)
                except zmq.ZMQError as exc:
                    if self._stop.is_set():
                        break
                    print(f"[dynamic-condition] command server ZMQ error: {exc}", flush=True)
        finally:
            socket.close(0)
