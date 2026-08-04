from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

from omg.realtime.g1_motion_mux import coerce_g1_qpos_frame


@dataclass(frozen=True)
class RedisQposPublisherConfig:
    host: str = "127.0.0.1"
    port: int = 6379
    db: int = 0
    key: str = "omg_online_frame_g1"
    ttl_ms: int = 250
    connect_timeout_seconds: float = 2.0
    socket_timeout_seconds: float = 2.0
    reconnect_interval_seconds: float = 0.25
    verbose: bool = True

    def __post_init__(self) -> None:
        if not str(self.host).strip():
            raise ValueError("Redis host must be non-empty")
        if int(self.port) <= 0 or int(self.port) > 65535:
            raise ValueError(f"Redis port must be in [1,65535], got {self.port}")
        if int(self.db) < 0:
            raise ValueError(f"Redis db must be non-negative, got {self.db}")
        if not str(self.key):
            raise ValueError("Redis key must be non-empty")
        if int(self.ttl_ms) < 0:
            raise ValueError(f"Redis ttl_ms must be non-negative, got {self.ttl_ms}")
        for name, value in (
            ("connect_timeout_seconds", self.connect_timeout_seconds),
            ("socket_timeout_seconds", self.socket_timeout_seconds),
        ):
            number = float(value)
            if not np.isfinite(number) or number <= 0.0:
                raise ValueError(f"{name} must be positive and finite, got {value}")
        reconnect = float(self.reconnect_interval_seconds)
        if not np.isfinite(reconnect) or reconnect < 0.0:
            raise ValueError(
                "reconnect_interval_seconds must be non-negative and finite, "
                f"got {self.reconnect_interval_seconds}"
            )


def g1_qpos_payload(qpos_36: Any, *, timestamp: float) -> dict[str, Any]:
    frame = coerce_g1_qpos_frame(qpos_36)
    stamp = float(timestamp)
    if not np.isfinite(stamp):
        raise ValueError(f"timestamp must be finite, got {timestamp}")
    return {
        "timestamp": stamp,
        "root_pos": frame[:3].astype(float).tolist(),
        "root_quat": frame[3:7].astype(float).tolist(),
        "joints": frame[7:].astype(float).tolist(),
    }


def encode_g1_qpos_json(qpos_36: Any, *, timestamp: float) -> str:
    return json.dumps(
        g1_qpos_payload(qpos_36, timestamp=timestamp),
        separators=(",", ":"),
        allow_nan=False,
    )


class _RedisConnectionLike(Protocol):
    def set_json(self, key: str, payload: str, ttl_ms: int) -> None: ...

    def close(self) -> None: ...


class RespRedisConnection:
    """Small RESP2 connection used to avoid a mandatory redis-py dependency."""

    def __init__(self, config: RedisQposPublisherConfig) -> None:
        self._socket = socket.create_connection(
            (str(config.host), int(config.port)),
            timeout=float(config.connect_timeout_seconds),
        )
        self._socket.settimeout(float(config.socket_timeout_seconds))
        self._file = self._socket.makefile("rb")
        try:
            if int(config.db) != 0:
                self.command("SELECT", str(int(config.db)))
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _encode(parts: tuple[str, ...]) -> bytes:
        payload = [f"*{len(parts)}\r\n".encode("ascii")]
        for part in parts:
            value = str(part).encode("utf-8")
            payload.extend(
                (
                    f"${len(value)}\r\n".encode("ascii"),
                    value,
                    b"\r\n",
                )
            )
        return b"".join(payload)

    def command(self, *parts: str) -> bytes:
        self._socket.sendall(self._encode(tuple(str(part) for part in parts)))
        prefix = self._file.read(1)
        if prefix == b"":
            raise ConnectionError("Redis closed the connection")
        if prefix in {b"+", b"-", b":"}:
            response = self._file.readline().rstrip(b"\r\n")
            if prefix == b"-":
                raise RuntimeError(
                    f"Redis error: {response.decode('utf-8', errors='replace')}"
                )
            return response
        if prefix == b"$":
            length_line = self._file.readline()
            if not length_line:
                raise ConnectionError("Redis closed the bulk response")
            length = int(length_line)
            if length < 0:
                return b""
            response = self._file.read(length)
            trailer = self._file.read(2)
            if len(response) != length or trailer != b"\r\n":
                raise ConnectionError("Redis returned a truncated bulk response")
            return response
        raise RuntimeError(f"Unexpected Redis response prefix: {prefix!r}")

    def set_json(self, key: str, payload: str, ttl_ms: int) -> None:
        if int(ttl_ms) > 0:
            self.command("SET", str(key), str(payload), "PX", str(int(ttl_ms)))
        else:
            self.command("SET", str(key), str(payload))

    def close(self) -> None:
        try:
            self._file.close()
        finally:
            self._socket.close()


class RedisQposPublisher:
    """Best-effort G1 qpos publisher with validation and reconnect support."""

    def __init__(
        self,
        config: RedisQposPublisherConfig | None = None,
        *,
        connection_factory: Callable[[RedisQposPublisherConfig], _RedisConnectionLike] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or RedisQposPublisherConfig()
        self._factory = connection_factory or RespRedisConnection
        self._clock = clock
        self._connection: _RedisConnectionLike | None = None
        self._next_connect_time = float("-inf")
        self._last_warning_time = float("-inf")
        self._published_frames = 0
        self._publish_errors = 0
        self._connect_attempts = 0
        self._last_error: str | None = None
        self._closed = False

    def _warn(self, message: str) -> None:
        if not self.config.verbose:
            return
        now = float(self._clock())
        if now - self._last_warning_time >= 1.0:
            print(f"[OMG G1 Redis] {message}", flush=True)
            self._last_warning_time = now

    def _disconnect(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

    def _ensure_connection(self) -> bool:
        if self._closed:
            return False
        if self._connection is not None:
            return True
        now = float(self._clock())
        if now < self._next_connect_time:
            return False
        self._connect_attempts += 1
        try:
            self._connection = self._factory(self.config)
            self._last_error = None
            if self.config.verbose:
                print(
                    f"[OMG G1 Redis] connected {self.config.host}:{self.config.port}/"
                    f"{self.config.db} key={self.config.key}",
                    flush=True,
                )
            return True
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._next_connect_time = now + float(self.config.reconnect_interval_seconds)
            self._warn(f"connect failed: {self._last_error}; will retry")
            return False

    def publish(self, qpos_36: Any, *, timestamp: float | None = None) -> bool:
        stamp = float(self._clock()) if timestamp is None else float(timestamp)
        payload = encode_g1_qpos_json(qpos_36, timestamp=stamp)
        if not self._ensure_connection():
            return False
        try:
            assert self._connection is not None
            self._connection.set_json(
                str(self.config.key),
                payload,
                int(self.config.ttl_ms),
            )
            self._published_frames += 1
            self._last_error = None
            return True
        except Exception as exc:
            self._publish_errors += 1
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._warn(f"publish failed: {self._last_error}; reconnecting")
            self._disconnect()
            self._next_connect_time = float(self._clock()) + float(
                self.config.reconnect_interval_seconds
            )
            return False

    def status(self) -> dict[str, Any]:
        return {
            "connected": self._connection is not None,
            "host": str(self.config.host),
            "port": int(self.config.port),
            "db": int(self.config.db),
            "key": str(self.config.key),
            "ttl_ms": int(self.config.ttl_ms),
            "published_frames": int(self._published_frames),
            "publish_errors": int(self._publish_errors),
            "connect_attempts": int(self._connect_attempts),
            "last_error": self._last_error,
        }

    def close(self) -> None:
        self._closed = True
        self._disconnect()


class AsyncRedisQposPublisher:
    """Non-blocking latest-frame adapter around :class:`RedisQposPublisher`.

    Redis connect/reconnect and socket writes happen only in the worker thread,
    so a missing Redis server cannot stop the tracker-rate motion loop.
    """

    def __init__(self, publisher: RedisQposPublisher) -> None:
        self.publisher = publisher
        self._condition = threading.Condition()
        self._latest: tuple[int, np.ndarray, float] | None = None
        self._next_sequence = 0
        self._consumed_sequence = -1
        self._queued_frames = 0
        self._dropped_frames = 0
        self._stop = False
        self._thread: threading.Thread | None = None

    @property
    def is_alive(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Async Redis publisher has already been started")
        self._stop = False
        self._thread = threading.Thread(
            target=self._run,
            name="omg-g1-redis-publisher",
            daemon=True,
        )
        self._thread.start()

    def publish(self, qpos_36: Any, *, timestamp: float | None = None) -> bool:
        frame = coerce_g1_qpos_frame(qpos_36)
        stamp = time.monotonic() if timestamp is None else float(timestamp)
        if not np.isfinite(stamp):
            raise ValueError(f"timestamp must be finite, got {timestamp}")
        with self._condition:
            if self._stop:
                return False
            sequence = self._next_sequence
            self._next_sequence += 1
            if self._latest is not None and self._latest[0] > self._consumed_sequence:
                self._dropped_frames += 1
            self._latest = (sequence, frame, stamp)
            self._queued_frames += 1
            self._condition.notify()
        return True

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stop
                    or (
                        self._latest is not None
                        and self._latest[0] > self._consumed_sequence
                    )
                )
                if self._stop:
                    break
                assert self._latest is not None
                sequence, frame, stamp = self._latest
                self._consumed_sequence = sequence
            self.publisher.publish(frame, timestamp=stamp)

    def status(self) -> dict[str, Any]:
        with self._condition:
            queued = int(self._queued_frames)
            dropped = int(self._dropped_frames)
            pending = bool(
                self._latest is not None
                and self._latest[0] > self._consumed_sequence
            )
        return {
            **self.publisher.status(),
            "async_thread_alive": self.is_alive,
            "queued_frames": queued,
            "dropped_frames": dropped,
            "latest_pending": pending,
        }

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            timeout = (
                float(self.publisher.config.connect_timeout_seconds)
                + float(self.publisher.config.socket_timeout_seconds)
                + 1.0
            )
            thread.join(timeout=timeout)
            if thread.is_alive():
                raise RuntimeError("Async Redis publisher thread did not stop")
        self._thread = None
        self.publisher.close()
