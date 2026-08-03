from __future__ import annotations

import argparse
import json
import shlex
from typing import Any


def _zmq() -> Any:
    try:
        import zmq
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("Realtime command client requires pyzmq. Install omg[realtime].") from exc
    return zmq


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send runtime text/audio commands to an OMG bridge.")
    parser.add_argument("--connect", default="tcp://127.0.0.1:5581")
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("command", nargs="?", choices=["text", "audio", "stand", "status"])
    parser.add_argument("value", nargs="*")
    return parser.parse_args()


def _request_from_parts(command: str, values: list[str]) -> dict[str, Any]:
    if command == "text":
        text = " ".join(values).strip()
        if not text:
            raise ValueError("text command requires a prompt")
        return {"type": "text", "text": text}
    if command == "audio":
        path = " ".join(values).strip()
        if not path:
            raise ValueError("audio command requires an absolute WAV path")
        return {"type": "audio", "audio_path": path, "audio_type": "audio", "on_end": "stand"}
    if values:
        raise ValueError(f"{command} command does not accept arguments")
    return {"type": command}


class _CommandConnection:
    def __init__(self, connect: str, timeout_ms: int) -> None:
        zmq = _zmq()
        timeout = int(timeout_ms)
        if timeout <= 0:
            raise ValueError("timeout_ms must be positive")
        self._zmq = zmq
        self._socket = zmq.Context.instance().socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(str(connect))
        self._timeout_ms = timeout

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._socket.send_json(payload)
        if (int(self._socket.poll(self._timeout_ms)) & self._zmq.POLLIN) == 0:
            raise TimeoutError(f"Timed out waiting {self._timeout_ms}ms for command response")
        response = self._socket.recv_json()
        if not isinstance(response, dict):
            raise RuntimeError(f"Command server returned non-object JSON: {response!r}")
        return response

    def close(self) -> None:
        self._socket.close(0)


def _print_response(response: dict[str, Any]) -> None:
    print(json.dumps(response, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


def _interactive(connection: _CommandConnection) -> None:
    print("Commands: text <prompt>, audio <absolute.wav>, stand, status, help, quit", flush=True)
    while True:
        try:
            line = input("OMG realtime command > ").strip()
        except EOFError:
            print()
            return
        if not line:
            continue
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            print(f"error: {exc}", flush=True)
            continue
        command = parts[0].lower()
        if command == "quit":
            return
        if command == "help":
            print("text <prompt> | audio <absolute.wav> | stand | status | help | quit", flush=True)
            continue
        if command not in {"text", "audio", "stand", "status"}:
            print(f"error: unsupported command {command!r}", flush=True)
            continue
        try:
            _print_response(connection.request(_request_from_parts(command, parts[1:])))
        except Exception as exc:
            print(f"error: {type(exc).__name__}: {exc}", flush=True)


def main() -> None:
    args = _parse_args()
    if not args.interactive and args.command is None:
        raise ValueError("Provide a command or use --interactive")
    if args.interactive and args.command is not None:
        raise ValueError("--interactive cannot be combined with a one-shot command")
    connection = _CommandConnection(args.connect, args.timeout_ms)
    try:
        if args.interactive:
            _interactive(connection)
        else:
            _print_response(connection.request(_request_from_parts(args.command, list(args.value))))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
