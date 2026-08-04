from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest

from omg.realtime.redis_qpos import (
    AsyncRedisQposPublisher,
    RedisQposPublisher,
    RedisQposPublisherConfig,
    encode_g1_qpos_json,
    g1_qpos_payload,
)


def _frame() -> np.ndarray:
    frame = np.arange(36, dtype=np.float32) / 10.0
    frame[3:7] = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return frame


class _FakeConnection:
    def __init__(self) -> None:
        self.values: list[tuple[str, str, int]] = []
        self.closed = False

    def set_json(self, key: str, payload: str, ttl_ms: int) -> None:
        self.values.append((key, payload, ttl_ms))

    def close(self) -> None:
        self.closed = True


def test_payload_matches_gmr_json_protocol_and_normalizes_quaternion() -> None:
    payload = g1_qpos_payload(_frame(), timestamp=12.5)

    assert payload.keys() == {"timestamp", "root_pos", "root_quat", "joints"}
    assert len(payload["root_pos"]) == 3
    assert len(payload["root_quat"]) == 4
    assert len(payload["joints"]) == 29
    assert payload["root_quat"] == [1.0, 0.0, 0.0, 0.0]
    assert json.loads(encode_g1_qpos_json(_frame(), timestamp=12.5)) == payload


def test_invalid_payload_is_rejected_before_network_io() -> None:
    bad = _frame()
    bad[7] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        encode_g1_qpos_json(bad, timestamp=0.0)
    with pytest.raises(ValueError, match="timestamp"):
        encode_g1_qpos_json(_frame(), timestamp=float("inf"))


def test_publisher_sets_expected_key_and_ttl() -> None:
    connection = _FakeConnection()
    publisher = RedisQposPublisher(
        RedisQposPublisherConfig(key="omg_online_frame_g1", ttl_ms=250, verbose=False),
        connection_factory=lambda _config: connection,
    )

    assert publisher.publish(_frame(), timestamp=4.0)

    assert len(connection.values) == 1
    key, payload, ttl = connection.values[0]
    assert key == "omg_online_frame_g1"
    assert ttl == 250
    assert json.loads(payload)["timestamp"] == 4.0
    assert publisher.status()["published_frames"] == 1
    publisher.close()
    assert connection.closed


def test_publisher_reconnects_after_connect_failure() -> None:
    now = [0.0]
    connection = _FakeConnection()
    attempts = [0]

    def factory(_config: RedisQposPublisherConfig) -> _FakeConnection:
        attempts[0] += 1
        if attempts[0] == 1:
            raise ConnectionError("not ready")
        return connection

    publisher = RedisQposPublisher(
        RedisQposPublisherConfig(reconnect_interval_seconds=0.25, verbose=False),
        connection_factory=factory,
        clock=lambda: now[0],
    )

    assert not publisher.publish(_frame(), timestamp=0.0)
    now[0] = 0.1
    assert not publisher.publish(_frame(), timestamp=0.1)
    now[0] = 0.3
    assert publisher.publish(_frame(), timestamp=0.3)
    assert attempts[0] == 2
    assert len(connection.values) == 1


def test_async_publisher_keeps_socket_work_off_calling_thread() -> None:
    caller_thread = threading.get_ident()
    worker_threads: list[int] = []
    connection = _FakeConnection()

    class RecordingConnection(_FakeConnection):
        def set_json(self, key: str, payload: str, ttl_ms: int) -> None:
            worker_threads.append(threading.get_ident())
            super().set_json(key, payload, ttl_ms)

    connection = RecordingConnection()
    sync = RedisQposPublisher(
        RedisQposPublisherConfig(verbose=False),
        connection_factory=lambda _config: connection,
    )
    publisher = AsyncRedisQposPublisher(sync)
    publisher.start()
    try:
        assert publisher.publish(_frame(), timestamp=1.0)
        deadline = time.monotonic() + 1.0
        while not connection.values and time.monotonic() < deadline:
            time.sleep(0.005)
        assert connection.values
        assert worker_threads[0] != caller_thread
        assert publisher.is_alive
    finally:
        publisher.close()
    assert not publisher.is_alive
