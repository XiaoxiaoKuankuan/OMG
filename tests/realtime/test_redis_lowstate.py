from __future__ import annotations

import time

import numpy as np

from omg.realtime.gmt_trajectory import GmtLowStateFeedback, joint_order_sha256
from omg.realtime.redis_trajectory import (
    RedisLowStateFeedbackReader,
    RedisLowStateFeedbackReaderConfig,
)


class _Connection:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.closed = False

    def get_bytes(self, _key: str) -> bytes | None:
        return self.payload

    def close(self) -> None:
        self.closed = True


def _sample(sequence: int, captured_unix_ns: int) -> GmtLowStateFeedback:
    return GmtLowStateFeedback(
        sequence=sequence,
        captured_unix_ns=captured_unix_ns,
        joint_order_hash=joint_order_sha256([f"j{i}" for i in range(21)]),
        root_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        joint_pos=np.arange(21, dtype=np.float32),
    )


def test_lowstate_reader_does_not_refresh_age_for_repeated_redis_value() -> None:
    now = [10.0]
    connection = _Connection(_sample(1, 100).encode())
    reader = RedisLowStateFeedbackReader(
        RedisLowStateFeedbackReaderConfig(
            key="motion_lowstate",
            poll_interval_seconds=0.001,
            verbose=False,
        ),
        connection_factory=lambda _config: connection,
        clock=lambda: now[0],
    )
    reader.start()
    deadline = time.monotonic() + 1.0
    while reader.latest_feedback() is None and time.monotonic() < deadline:
        time.sleep(0.001)
    assert reader.latest_feedback(max_age_seconds=0.2) is not None

    now[0] = 10.25
    time.sleep(0.01)
    assert reader.latest_feedback(max_age_seconds=0.2) is None

    connection.payload = _sample(2, 200).encode()
    deadline = time.monotonic() + 1.0
    while reader.status()["valid_samples"] < 2 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert reader.latest_feedback(max_age_seconds=0.2).sequence == 2
    reader.close()
    assert not reader.is_alive
    assert connection.closed
