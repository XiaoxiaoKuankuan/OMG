from __future__ import annotations

import time

from omg.realtime.gmt_trajectory import GmtTrajectoryAck
from omg.realtime.redis_trajectory import (
    RedisTrajectoryAckReader,
    RedisTrajectoryAckReaderConfig,
)


class _Connection:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.closed = False

    def get_bytes(self, _key: str) -> bytes | None:
        return self.payload

    def close(self) -> None:
        self.closed = True


def test_ack_reader_decodes_on_background_thread_and_closes() -> None:
    expected = GmtTrajectoryAck(
        stream_id=1,
        sequence=2,
        command_revision=3,
        plan_id=4,
        received_unix_ns=5,
    )
    connection = _Connection(expected.encode())
    reader = RedisTrajectoryAckReader(
        RedisTrajectoryAckReaderConfig(
            key="motion_ack",
            poll_interval_seconds=0.001,
            verbose=False,
        ),
        connection_factory=lambda _config: connection,
    )
    reader.start()
    deadline = time.monotonic() + 1.0
    while reader.latest_ack() is None and time.monotonic() < deadline:
        time.sleep(0.001)
    assert reader.latest_ack() == expected
    assert reader.status()["valid_acks"] >= 1
    reader.close()
    assert not reader.is_alive
    assert connection.closed
