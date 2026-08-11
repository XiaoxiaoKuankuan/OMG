from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable


class SynchronizedAudioPlayer:
    """Own exactly one ffplay process and start it on a motion execution tick."""

    def __init__(
        self,
        *,
        enabled: bool,
        executable: str = "ffplay",
        popen_factory: Callable[..., Any] = subprocess.Popen,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.enabled = bool(enabled)
        self.executable = str(executable)
        self._popen_factory = popen_factory
        self._clock = clock
        self._process: Any | None = None
        self._command_id: str | None = None
        self._path: str | None = None
        self._started_tracker_frame: int | None = None
        self._started_monotonic: float | None = None
        self._last_stop_reason: str | None = None
        if self.enabled:
            resolved = shutil.which(self.executable)
            if resolved is None and not Path(self.executable).is_file():
                raise FileNotFoundError(
                    f"--play-audio requires ffplay; executable not found: {self.executable}"
                )
            if resolved is not None:
                self.executable = resolved

    @property
    def is_playing(self) -> bool:
        return bool(self._process is not None and self._process.poll() is None)

    @property
    def command_id(self) -> str | None:
        return self._command_id

    def start(
        self,
        audio_path: str | Path,
        *,
        command_id: str,
        tracker_frame: int,
    ) -> bool:
        if not self.enabled:
            return False
        path = Path(audio_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio playback file does not exist: {path}")
        if self._command_id == str(command_id) and self.is_playing:
            return False
        self.stop(reason="replaced")
        self._process = self._popen_factory(
            [
                self.executable,
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "error",
                str(path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self._command_id = str(command_id)
        self._path = str(path)
        self._started_tracker_frame = int(tracker_frame)
        self._started_monotonic = float(self._clock())
        self._last_stop_reason = None
        print(
            f"[audio-playback] started command_id={command_id} frame={tracker_frame} "
            f"path={path}",
            flush=True,
        )
        return True

    def stop(self, *, reason: str) -> bool:
        process, self._process = self._process, None
        was_active = process is not None and process.poll() is None
        if was_active:
            try:
                os.killpg(int(process.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    process.terminate()
                except Exception:
                    pass
            try:
                process.wait(timeout=1.0)
            except Exception:
                try:
                    os.killpg(int(process.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    try:
                        process.kill()
                    except Exception:
                        pass
                try:
                    process.wait(timeout=1.0)
                except Exception:
                    pass
        if process is not None:
            print(
                f"[audio-playback] stopped command_id={self._command_id} reason={reason}",
                flush=True,
            )
        self._command_id = None
        self._path = None
        self._started_tracker_frame = None
        self._started_monotonic = None
        self._last_stop_reason = str(reason)
        return bool(was_active)

    def status(self) -> dict[str, Any]:
        playing = self.is_playing
        if self._process is not None and not playing:
            self._last_stop_reason = "process_exited"
        elapsed = (
            None
            if self._started_monotonic is None
            else max(0.0, float(self._clock()) - self._started_monotonic)
        )
        return {
            "enabled": self.enabled,
            "playing": playing,
            "command_id": self._command_id,
            "audio_path": self._path,
            "started_tracker_frame": self._started_tracker_frame,
            "elapsed_seconds": elapsed,
            "last_stop_reason": self._last_stop_reason,
        }

    def close(self) -> None:
        self.stop(reason="bridge_closed")
