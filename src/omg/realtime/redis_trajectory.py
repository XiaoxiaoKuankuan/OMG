from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

from omg.realtime.gmt_trajectory import GmtTrajectoryAck, GmtTrajectoryPacket


@dataclass(frozen=True)
class RedisTrajectoryPublisherConfig:
    host: str = "127.0.0.1"
    port: int = 6379
    db: int = 0
    key: str = "gmt_online_frame_bumi"
    ttl_ms: int = 500
    connect_timeout_seconds: float = 0.5
    socket_timeout_seconds: float = 0.5
    reconnect_interval_seconds: float = 0.25
    verbose: bool = True

    def __post_init__(self) -> None:
        if not str(self.host).strip():
            raise ValueError("Redis host must be non-empty")
        if not 1 <= int(self.port) <= 65535:
            raise ValueError("Redis port must be in [1,65535]")
        if int(self.db) < 0:
            raise ValueError("Redis db must be non-negative")
        if not str(self.key):
            raise ValueError("Redis key must be non-empty")
        if int(self.ttl_ms) <= 0:
            raise ValueError("Redis trajectory ttl_ms must be positive")
        for name, value in (
            ("connect_timeout_seconds", self.connect_timeout_seconds),
            ("socket_timeout_seconds", self.socket_timeout_seconds),
        ):
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if (
            not np.isfinite(float(self.reconnect_interval_seconds))
            or float(self.reconnect_interval_seconds) < 0.0
        ):
            raise ValueError("reconnect_interval_seconds must be non-negative and finite")


class _BinaryRedisConnection(Protocol):
    def set_bytes(self, key: str, payload: bytes, ttl_ms: int) -> None: ...

    def get_bytes(self, key: str) -> bytes | None: ...

    def close(self) -> None: ...


class RespBinaryRedisConnection:
    def __init__(self, config: RedisTrajectoryPublisherConfig) -> None:
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
    def _encode(parts: tuple[str | bytes, ...]) -> bytes:
        output = [f"*{len(parts)}\r\n".encode("ascii")]
        for part in parts:
            value = part if isinstance(part, bytes) else str(part).encode("utf-8")
            output.extend((f"${len(value)}\r\n".encode("ascii"), value, b"\r\n"))
        return b"".join(output)

    def command(self, *parts: str | bytes) -> bytes:
        self._socket.sendall(self._encode(tuple(parts)))
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
            length = int(self._file.readline())
            if length < 0:
                return b""
            response = self._file.read(length)
            trailer = self._file.read(2)
            if len(response) != length or trailer != b"\r\n":
                raise ConnectionError("Redis returned a truncated response")
            return response
        raise RuntimeError(f"Unexpected Redis response prefix: {prefix!r}")

    def set_bytes(self, key: str, payload: bytes, ttl_ms: int) -> None:
        self.command("SET", str(key), bytes(payload), "PX", str(int(ttl_ms)))

    def get_bytes(self, key: str) -> bytes | None:
        response = self.command("GET", str(key))
        return None if response == b"" else bytes(response)

    def close(self) -> None:
        try:
            self._file.close()
        finally:
            self._socket.close()


class RedisTrajectoryPublisher:
    def __init__(
        self,
        config: RedisTrajectoryPublisherConfig | None = None,
        *,
        connection_factory: Callable[
            [RedisTrajectoryPublisherConfig], _BinaryRedisConnection
        ]
        | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or RedisTrajectoryPublisherConfig()
        self._factory = connection_factory or RespBinaryRedisConnection
        self._clock = clock
        self._connection: _BinaryRedisConnection | None = None
        self._next_connect_time = float("-inf")
        self._last_warning_time = float("-inf")
        self._published_packets = 0
        self._publish_errors = 0
        self._connect_attempts = 0
        self._last_error: str | None = None
        self._closed = False

    def _warn(self, message: str) -> None:
        now = float(self._clock())
        if self.config.verbose and now - self._last_warning_time >= 1.0:
            print(f"[OMG BUMI Redis] {message}", flush=True)
            self._last_warning_time = now

    def _disconnect(self) -> None:
        connection, self._connection = self._connection, None
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
                    f"[OMG BUMI Redis] connected {self.config.host}:{self.config.port}/"
                    f"{self.config.db} key={self.config.key} protocol=trajectory_v1",
                    flush=True,
                )
            return True
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._next_connect_time = now + float(self.config.reconnect_interval_seconds)
            self._warn(f"connect failed: {self._last_error}; will retry")
            return False

    def publish(self, packet: GmtTrajectoryPacket | bytes) -> bool:
        payload = packet.encode() if isinstance(packet, GmtTrajectoryPacket) else bytes(packet)
        if not self._ensure_connection():
            return False
        try:
            assert self._connection is not None
            self._connection.set_bytes(
                str(self.config.key), payload, int(self.config.ttl_ms)
            )
            self._published_packets += 1
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
            "host": self.config.host,
            "port": int(self.config.port),
            "db": int(self.config.db),
            "key": self.config.key,
            "ttl_ms": int(self.config.ttl_ms),
            "protocol": "trajectory_v1",
            "published_packets": int(self._published_packets),
            "publish_errors": int(self._publish_errors),
            "connect_attempts": int(self._connect_attempts),
            "last_error": self._last_error,
        }

    def close(self) -> None:
        self._closed = True
        self._disconnect()


class AsyncRedisTrajectoryPublisher:
    """Latest-only non-blocking binary trajectory publisher."""

    def __init__(self, publisher: RedisTrajectoryPublisher) -> None:
        self.publisher = publisher
        self._condition = threading.Condition()
        self._latest: tuple[int, bytes] | None = None
        self._next_sequence = 0
        self._consumed_sequence = -1
        self._queued_packets = 0
        self._dropped_packets = 0
        self._stop = False
        self._thread: threading.Thread | None = None

    @property
    def is_alive(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Async Redis trajectory publisher already started")
        self._thread = threading.Thread(
            target=self._run, name="omg-bumi-redis-publisher", daemon=True
        )
        self._thread.start()

    def publish(self, packet: GmtTrajectoryPacket | bytes) -> bool:
        payload = packet.encode() if isinstance(packet, GmtTrajectoryPacket) else bytes(packet)
        with self._condition:
            if self._stop:
                return False
            sequence = self._next_sequence
            self._next_sequence += 1
            if self._latest is not None and self._latest[0] > self._consumed_sequence:
                self._dropped_packets += 1
            self._latest = (sequence, payload)
            self._queued_packets += 1
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
                    return
                assert self._latest is not None
                sequence, payload = self._latest
                self._consumed_sequence = sequence
            self.publisher.publish(payload)

    def status(self) -> dict[str, Any]:
        with self._condition:
            pending = bool(
                self._latest is not None
                and self._latest[0] > self._consumed_sequence
            )
            queued = self._queued_packets
            dropped = self._dropped_packets
        return {
            **self.publisher.status(),
            "async_thread_alive": self.is_alive,
            "queued_packets": int(queued),
            "dropped_packets": int(dropped),
            "latest_pending": pending,
        }

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        if self._thread is not None and self._thread is not threading.current_thread():
            timeout = (
                float(self.publisher.config.connect_timeout_seconds)
                + float(self.publisher.config.socket_timeout_seconds)
                + 1.0
            )
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                raise RuntimeError("Async Redis trajectory publisher thread did not stop")
        self._thread = None
        self.publisher.close()


@dataclass(frozen=True)
class RedisTrajectoryAckReaderConfig:
    host: str = "127.0.0.1"
    port: int = 6379
    db: int = 0
    key: str = "gmt_online_frame_bumi_ack"
    poll_interval_seconds: float = 0.005
    connect_timeout_seconds: float = 0.5
    socket_timeout_seconds: float = 0.5
    reconnect_interval_seconds: float = 0.25
    verbose: bool = True

    def __post_init__(self) -> None:
        if not str(self.host).strip():
            raise ValueError("Redis ACK host must be non-empty")
        if not 1 <= int(self.port) <= 65535:
            raise ValueError("Redis ACK port must be in [1,65535]")
        if int(self.db) < 0:
            raise ValueError("Redis ACK db must be non-negative")
        if not str(self.key):
            raise ValueError("Redis ACK key must be non-empty")
        for name, value in (
            ("poll_interval_seconds", self.poll_interval_seconds),
            ("connect_timeout_seconds", self.connect_timeout_seconds),
            ("socket_timeout_seconds", self.socket_timeout_seconds),
        ):
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if (
            not np.isfinite(float(self.reconnect_interval_seconds))
            or float(self.reconnect_interval_seconds) < 0.0
        ):
            raise ValueError("reconnect_interval_seconds must be non-negative and finite")


class RedisTrajectoryAckReader:
    """Poll the separate GMT acknowledgement key without blocking the 50 Hz loop."""

    def __init__(
        self,
        config: RedisTrajectoryAckReaderConfig | None = None,
        *,
        connection_factory: Callable[
            [RedisTrajectoryPublisherConfig], _BinaryRedisConnection
        ]
        | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or RedisTrajectoryAckReaderConfig()
        self._factory = connection_factory or RespBinaryRedisConnection
        self._clock = clock
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: GmtTrajectoryAck | None = None
        self._connection: _BinaryRedisConnection | None = None
        self._next_connect_time = float("-inf")
        self._connect_attempts = 0
        self._valid_acks = 0
        self._decode_errors = 0
        self._read_errors = 0
        self._last_error: str | None = None

    @property
    def is_alive(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Redis trajectory ACK reader already started")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="omg-bumi-redis-ack-reader",
            daemon=True,
        )
        self._thread.start()

    def latest_ack(self) -> GmtTrajectoryAck | None:
        with self._condition:
            return self._latest

    def _connection_config(self) -> RedisTrajectoryPublisherConfig:
        return RedisTrajectoryPublisherConfig(
            host=self.config.host,
            port=self.config.port,
            db=self.config.db,
            key=self.config.key,
            ttl_ms=1000,
            connect_timeout_seconds=self.config.connect_timeout_seconds,
            socket_timeout_seconds=self.config.socket_timeout_seconds,
            reconnect_interval_seconds=self.config.reconnect_interval_seconds,
            verbose=False,
        )

    def _disconnect(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

    def _ensure_connection(self) -> bool:
        if self._connection is not None:
            return True
        now = float(self._clock())
        if now < self._next_connect_time:
            return False
        self._connect_attempts += 1
        try:
            self._connection = self._factory(self._connection_config())
            self._last_error = None
            if self.config.verbose:
                print(
                    f"[OMG BUMI Redis ACK] connected {self.config.host}:"
                    f"{self.config.port}/{self.config.db} key={self.config.key}",
                    flush=True,
                )
            return True
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._next_connect_time = now + float(
                self.config.reconnect_interval_seconds
            )
            return False

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if not self._ensure_connection():
                self._stop_event.wait(float(self.config.poll_interval_seconds))
                continue
            try:
                assert self._connection is not None
                payload = self._connection.get_bytes(str(self.config.key))
                if payload is not None:
                    try:
                        ack = GmtTrajectoryAck.decode(payload)
                    except Exception as exc:
                        self._decode_errors += 1
                        self._last_error = f"{type(exc).__name__}: {exc}"
                    else:
                        with self._condition:
                            previous = self._latest
                            if (
                                previous is None
                                or ack.stream_id != previous.stream_id
                                or ack.sequence > previous.sequence
                            ):
                                self._latest = ack
                                self._valid_acks += 1
                                self._last_error = None
                                self._condition.notify_all()
            except Exception as exc:
                self._read_errors += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._disconnect()
                self._next_connect_time = float(self._clock()) + float(
                    self.config.reconnect_interval_seconds
                )
            self._stop_event.wait(float(self.config.poll_interval_seconds))

    def status(self) -> dict[str, Any]:
        with self._condition:
            ack = self._latest
        return {
            "connected": self._connection is not None,
            "thread_alive": self.is_alive,
            "key": self.config.key,
            "connect_attempts": int(self._connect_attempts),
            "valid_acks": int(self._valid_acks),
            "decode_errors": int(self._decode_errors),
            "read_errors": int(self._read_errors),
            "last_error": self._last_error,
            "latest_stream_id": None if ack is None else int(ack.stream_id),
            "latest_sequence": None if ack is None else int(ack.sequence),
            "latest_command_revision": (
                None if ack is None else int(ack.command_revision)
            ),
        }

    def close(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            timeout = float(self.config.socket_timeout_seconds) + 1.0
            thread.join(timeout=timeout)
            if thread.is_alive():
                raise RuntimeError("Redis trajectory ACK reader thread did not stop")
        self._thread = None
        self._disconnect()
