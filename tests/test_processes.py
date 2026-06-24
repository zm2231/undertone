import subprocess
import sys
import time
from pathlib import Path

import pytest

from undertone_audio.audio import AudioInfo, AudioPreprocessor
from undertone_audio.config import Config
from undertone_audio.processes import (
    ExternalProcessError,
    atomic_write_path,
    load_json_file,
    load_json_text,
    run_process_async,
    run_process_sync,
)


def test_config_rejects_negative_process_timeout(tmp_path):
    with pytest.raises(ValueError, match="process timeout"):
        Config(db_path=tmp_path / "undertone.db", process_timeout_seconds=-1)


def test_run_process_sync_times_out_and_reports_label():
    with pytest.raises(ExternalProcessError, match="sleepy timed out"):
        run_process_sync(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            label="sleepy",
            timeout_seconds=0.1,
        )


def test_run_process_sync_cleans_up_on_keyboard_interrupt(monkeypatch):
    calls = {"terminated": 0}

    class FakeProc:
        pid = 123
        returncode = None

        def communicate(self, timeout=None):
            raise KeyboardInterrupt()

        def poll(self):
            return None

        def wait(self, timeout=None):
            self.returncode = -15
            return self.returncode

    fake_proc = FakeProc()
    monkeypatch.setattr("undertone_audio.processes.subprocess.Popen", lambda *args, **kwargs: fake_proc)

    def fake_terminate(proc):
        assert proc is fake_proc
        calls["terminated"] += 1

    monkeypatch.setattr("undertone_audio.processes._terminate_sync_process", fake_terminate)

    with pytest.raises(KeyboardInterrupt):
        run_process_sync([sys.executable, "-c", "pass"], label="interruptible")

    assert calls["terminated"] == 1


def test_process_group_signal_permission_errors_are_ignored(monkeypatch):
    from undertone_audio.processes import _kill_process_group, _terminate_process_group

    calls = []

    def deny(pid, sig):
        calls.append((pid, sig))
        raise PermissionError("already gone or not owned")

    monkeypatch.setattr("undertone_audio.processes.os.name", "posix")
    monkeypatch.setattr("undertone_audio.processes.os.killpg", deny)

    _terminate_process_group(123)
    _kill_process_group(456)

    assert [pid for pid, _sig in calls] == [123, 456]


def test_terminate_sync_does_not_hang_when_kill_cannot_reap(monkeypatch):
    import subprocess as sp
    import threading
    import time

    from undertone_audio import processes

    monkeypatch.setattr(processes, "TERMINATE_GRACE_SECONDS", 0.05)
    monkeypatch.setattr(processes, "_terminate_process_group", lambda pid: None)
    monkeypatch.setattr(processes, "_kill_process_group", lambda pid: None)

    class UnkillableProc:
        pid = 999999

        def poll(self):
            return None

        def wait(self, timeout=None):
            if timeout is None:
                time.sleep(30)
                return 0
            raise sp.TimeoutExpired(cmd="stub", timeout=timeout)

    done = threading.Event()

    def run():
        processes._terminate_sync_process(UnkillableProc())
        done.set()

    threading.Thread(target=run, daemon=True).start()
    assert done.wait(timeout=2.0), "cleanup hung after kill could not reap the process"


def test_sync_timeout_escalates_to_kill_after_grace(monkeypatch):
    from undertone_audio.processes import _terminate_sync_process

    calls = []

    class FakeProc:
        pid = 123

        def __init__(self):
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired(["cmd"], timeout)
            self.returncode = -9
            return self.returncode

    fake_proc = FakeProc()
    monkeypatch.setattr(
        "undertone_audio.processes._terminate_process_group",
        lambda pid: calls.append(("term", pid)),
    )
    monkeypatch.setattr(
        "undertone_audio.processes._kill_process_group",
        lambda pid: calls.append(("kill", pid)),
    )

    _terminate_sync_process(fake_proc)

    assert calls == [("term", 123), ("kill", 123)]


def test_run_process_async_timeout_terminates_process_group():
    import asyncio

    with pytest.raises(ExternalProcessError, match="async sleepy timed out"):
        asyncio.run(
            run_process_async(
                [sys.executable, "-c", "import time; time.sleep(5)"],
                label="async sleepy",
                timeout_seconds=0.1,
            )
        )


def test_external_json_validation_names_producer_and_shape(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(RuntimeError, match="producer did not produce JSON output"):
        load_json_file(missing, producer="producer")

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{")
    with pytest.raises(RuntimeError, match="producer produced invalid JSON"):
        load_json_file(invalid, producer="producer")

    with pytest.raises(RuntimeError, match="missing required key"):
        load_json_text("{}", producer="producer", required_keys=("segments",))


def test_atomic_write_path_preserves_existing_file_on_failure(tmp_path):
    dest = tmp_path / "audio.wav"
    dest.write_bytes(b"old")

    with pytest.raises(RuntimeError, match="boom"):
        with atomic_write_path(dest) as tmp:
            tmp.write_bytes(b"new")
            raise RuntimeError("boom")

    assert dest.read_bytes() == b"old"
    assert not list(tmp_path.glob(".audio.wav.*.tmp"))


def test_atomic_write_path_cleans_temp_file_on_keyboard_interrupt(tmp_path):
    dest = tmp_path / "audio.wav"

    with pytest.raises(KeyboardInterrupt):
        with atomic_write_path(dest) as tmp:
            tmp.write_bytes(b"partial")
            raise KeyboardInterrupt()

    assert not dest.exists()
    assert not list(tmp_path.glob(".audio.wav.*.tmp"))


def test_run_process_sync_timeout_signals_child_process_group(tmp_path):
    marker = tmp_path / "child-terminated"

    with pytest.raises(ExternalProcessError, match="sync group timed out"):
        run_process_sync(
            _spawn_child_command(marker),
            label="sync group",
            timeout_seconds=0.5,
        )

    assert _wait_for_file(marker)


def test_run_process_async_timeout_signals_child_process_group(tmp_path):
    import asyncio

    marker = tmp_path / "child-terminated"

    with pytest.raises(ExternalProcessError, match="async group timed out"):
        asyncio.run(
            run_process_async(
                _spawn_child_command(marker),
                label="async group",
                timeout_seconds=0.5,
            )
        )

    assert _wait_for_file(marker)


def test_audio_preprocessor_normalize_does_not_publish_failed_ffmpeg(monkeypatch, tmp_path):
    source = tmp_path / "source.mp3"
    source.write_bytes(b"audio")
    cache_dir = tmp_path / "cache"
    preprocessor = AudioPreprocessor(cache_dir=cache_dir, process_timeout_seconds=12)

    monkeypatch.setattr(AudioPreprocessor, "_verify_ffmpeg", staticmethod(lambda: None))
    monkeypatch.setattr(
        preprocessor,
        "probe",
        lambda path: AudioInfo(
            path=path,
            duration_s=1.0,
            sample_rate=44_100,
            channels=2,
            codec="mp3",
        ),
    )

    def fake_run(cmd, **kwargs):
        assert kwargs["timeout_seconds"] == 12
        Path(cmd[-1]).write_bytes(b"partial")
        raise RuntimeError("ffmpeg crashed")

    monkeypatch.setattr("undertone_audio.audio.run_process_sync", fake_run)

    with pytest.raises(RuntimeError, match="ffmpeg crashed"):
        preprocessor.normalize(source)

    assert not list(cache_dir.glob("*.wav"))
    assert not list(cache_dir.glob(".*.tmp"))


def _spawn_child_command(marker: Path) -> list[str]:
    child_code = (
        "import pathlib, signal, sys, time\n"
        "marker = pathlib.Path(sys.argv[1])\n"
        "def handle_term(signum, frame):\n"
        "    marker.write_text('terminated')\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, handle_term)\n"
        "time.sleep(30)\n"
    )
    parent_code = (
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]])\n"
        "time.sleep(30)\n"
    )
    return [sys.executable, "-c", parent_code, child_code, str(marker)]


def _wait_for_file(path: Path, timeout_seconds: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.05)
    return path.exists()
