from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from omg.realtime.command_server import CommandServerConfig, DynamicCommandServer
from omg.realtime.dynamic_condition import DynamicConditionController

zmq = pytest.importorskip("zmq")


def _endpoint() -> str:
    return f"inproc://omg-command-test-{uuid.uuid4().hex}"


def _request(socket: object, payload: dict[str, object]) -> dict[str, object]:
    socket.send_json(payload)
    return socket.recv_json()


@pytest.fixture
def command_server() -> tuple[DynamicCommandServer, object, DynamicConditionController]:
    endpoint = _endpoint()
    controller = DynamicConditionController(tracker_fps=50.0)
    server = DynamicCommandServer(CommandServerConfig(bind=endpoint), controller)
    server.start()
    socket = zmq.Context.instance().socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(endpoint)
    try:
        yield server, socket, controller
    finally:
        socket.close(0)
        server.close()


def test_text_json_command(command_server: tuple[DynamicCommandServer, object, DynamicConditionController]) -> None:
    _server, socket, _controller = command_server

    response = _request(socket, {"type": "text", "text": "turn left", "command_id": "left"})

    assert response["ok"] is True
    assert response["accepted"] is True
    assert response["switch_policy"] == "next_replan"
    assert response["active"]["command_id"] == "left"
    assert response["active"]["condition_sequence"] == "text: turn left"


def test_audio_json_command(
    command_server: tuple[DynamicCommandServer, object, DynamicConditionController],
    tmp_path: Path,
) -> None:
    _server, socket, _controller = command_server
    path = tmp_path / "music.wav"
    wavfile.write(path, 8000, np.zeros((4000,), dtype=np.int16))

    response = _request(
        socket,
        {
            "type": "audio",
            "audio_path": str(path),
            "audio_type": "audio",
            "on_end": "stand",
        },
    )

    assert response["ok"] is True
    assert response["active"]["type"] == "audio"
    assert response["active"]["audio_duration_seconds"] == pytest.approx(0.5)


def test_stand_and_status_commands(
    command_server: tuple[DynamicCommandServer, object, DynamicConditionController],
) -> None:
    _server, socket, _controller = command_server
    _request(socket, {"type": "text", "text": "walk"})

    stand = _request(socket, {"type": "stand", "command_id": "stop"})
    status = _request(socket, {"type": "status"})

    assert stand["active"]["type"] == "stand"
    assert stand["active"]["condition_sequence"] == "text: stand still"
    assert status["ok"] is True
    assert status["active"]["command_id"] == "stop"


def test_invalid_json_returns_structured_error(
    command_server: tuple[DynamicCommandServer, object, DynamicConditionController],
) -> None:
    _server, socket, _controller = command_server
    socket.send(b"{not valid json")

    response = socket.recv_json()

    assert response["ok"] is False
    assert response["error"]["type"] == "JSONDecodeError"
    assert response["error"]["message"]


def test_invalid_command_does_not_stop_server_or_replace_active(
    command_server: tuple[DynamicCommandServer, object, DynamicConditionController],
) -> None:
    server, socket, controller = command_server
    before = controller.snapshot()

    response = _request(socket, {"type": "dance-loop"})
    status = _request(socket, {"type": "status"})

    assert response["ok"] is False
    assert response["error"]["type"] == "ValueError"
    assert controller.snapshot() == before
    assert status["ok"] is True
    assert server.is_alive


def test_port_already_bound_is_reported() -> None:
    endpoint = _endpoint()
    first = DynamicCommandServer(
        CommandServerConfig(bind=endpoint),
        DynamicConditionController(tracker_fps=50.0),
    )
    second = DynamicCommandServer(
        CommandServerConfig(bind=endpoint),
        DynamicConditionController(tracker_fps=50.0),
    )
    first.start()
    try:
        with pytest.raises(RuntimeError, match="Failed to bind"):
            second.start()
    finally:
        second.close()
        first.close()


def test_server_close_stops_background_thread() -> None:
    server = DynamicCommandServer(
        CommandServerConfig(bind=_endpoint(), poll_timeout_ms=10),
        DynamicConditionController(tracker_fps=50.0),
    )
    server.start()
    assert server.is_alive

    server.close()

    assert not server.is_alive
