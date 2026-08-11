from __future__ import annotations

from pathlib import Path

from omg.realtime.audio_playback import SynchronizedAudioPlayer


class _Process:
    next_pid = 10000

    def __init__(self, command) -> None:
        self.command = command
        self.pid = _Process.next_pid
        _Process.next_pid += 1
        self.running = True

    def poll(self):
        return None if self.running else 0

    def terminate(self):
        self.running = False

    def kill(self):
        self.running = False

    def wait(self, timeout=None):
        del timeout
        self.running = False
        return 0


def test_new_audio_replaces_previous_and_close_leaves_no_process(
    tmp_path: Path, monkeypatch
) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first.write_bytes(b"wav")
    second.write_bytes(b"wav")
    processes = []

    def popen(command, **_kwargs):
        process = _Process(command)
        processes.append(process)
        return process

    monkeypatch.setattr("os.killpg", lambda _pid, _signal: processes[-1].terminate())
    player = SynchronizedAudioPlayer(
        enabled=True,
        executable="/bin/true",
        popen_factory=popen,
        clock=lambda: 1.0,
    )
    assert player.start(first, command_id="a", tracker_frame=10)
    assert player.start(second, command_id="b", tracker_frame=20)
    assert not processes[0].running
    assert processes[1].running
    player.close()
    assert not processes[1].running
    assert not player.status()["playing"]
