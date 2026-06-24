from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Iterator, Sequence

DEFAULT_PROCESS_TIMEOUT_SECONDS = 7200.0
TERMINATE_GRACE_SECONDS = 5.0

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessResult:
    args: list[str]
    returncode: int
    stdout: str | bytes
    stderr: str | bytes


class ExternalProcessError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        cmd: Sequence[str],
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
    ):
        super().__init__(message)
        self.cmd = list(cmd)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def process_timeout_from_env(default: float = DEFAULT_PROCESS_TIMEOUT_SECONDS) -> float:
    value = os.environ.get("UNDERTONE_PROCESS_TIMEOUT_SECONDS")
    if value in (None, ""):
        return default
    timeout = float(value)
    if timeout < 0:
        raise ValueError("UNDERTONE_PROCESS_TIMEOUT_SECONDS must be >= 0")
    return timeout


def validate_process_timeout(timeout_seconds: float) -> float:
    if timeout_seconds < 0:
        raise ValueError("process timeout must be >= 0")
    return timeout_seconds


async def run_process_async(
    cmd: Sequence[str],
    *,
    label: str,
    timeout_seconds: float | None = None,
) -> ProcessResult:
    timeout = _normalize_timeout(timeout_seconds)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **_process_group_kwargs(),
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError as exc:
        await _terminate_async_process(proc)
        raise _timeout_error(cmd, label, timeout_seconds) from exc
    except asyncio.CancelledError:
        await _terminate_async_process(proc)
        raise
    stdout = _decode(stdout_b)
    stderr = _decode(stderr_b)
    if proc.returncode != 0:
        raise _exit_error(cmd, label, proc.returncode, stdout, stderr)
    return ProcessResult(list(cmd), int(proc.returncode or 0), stdout, stderr)


def run_process_sync(
    cmd: Sequence[str],
    *,
    label: str,
    timeout_seconds: float | None = None,
    text: bool = True,
) -> ProcessResult:
    timeout = _normalize_timeout(timeout_seconds)
    proc = subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        **_process_group_kwargs(),
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_sync_process(proc)
        stdout_b, stderr_b = proc.communicate()
        raise _timeout_error(
            cmd,
            label,
            timeout_seconds,
            stdout=_decode(stdout_b),
            stderr=_decode(stderr_b),
        ) from exc
    except BaseException:
        _terminate_sync_process(proc)
        raise
    stdout = stdout or ("" if text else b"")
    stderr = stderr or ("" if text else b"")
    if proc.returncode != 0:
        raise _exit_error(cmd, label, proc.returncode, _decode(stdout), _decode(stderr))
    return ProcessResult(list(cmd), int(proc.returncode or 0), stdout, stderr)


def load_json_file(path: Path, *, producer: str, required_keys: Sequence[str] = ()) -> dict:
    if not path.exists():
        raise RuntimeError(f"{producer} did not produce JSON output at {path}")
    try:
        text = path.read_text()
    except OSError as exc:
        raise RuntimeError(f"{producer} JSON output could not be read at {path}: {exc}") from exc
    return load_json_text(text, producer=producer, source=str(path), required_keys=required_keys)


def load_json_text(
    text: str,
    *,
    producer: str,
    source: str = "stdout",
    required_keys: Sequence[str] = (),
) -> dict:
    if not text.strip():
        raise RuntimeError(f"{producer} produced empty JSON from {source}")
    try:
        value = json.loads(text)
    except JSONDecodeError as exc:
        raise RuntimeError(f"{producer} produced invalid JSON from {source}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{producer} produced non-object JSON from {source}")
    missing = [key for key in required_keys if key not in value]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"{producer} JSON from {source} is missing required key(s): {joined}")
    return value


@contextmanager
def atomic_write_path(dest: Path) -> Iterator[Path]:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.tmp")
    try:
        yield tmp
        if not tmp.exists() or tmp.stat().st_size == 0:
            raise RuntimeError(f"temporary output was not written: {tmp}")
        os.replace(tmp, dest)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def _normalize_timeout(timeout_seconds: float | None) -> float | None:
    if timeout_seconds is None:
        timeout_seconds = process_timeout_from_env()
    validate_process_timeout(timeout_seconds)
    return None if timeout_seconds <= 0 else timeout_seconds


def _process_group_kwargs() -> dict:
    return {"start_new_session": True} if os.name != "nt" else {}


async def _terminate_async_process(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    _terminate_process_group(proc.pid)
    try:
        await asyncio.wait_for(proc.wait(), timeout=TERMINATE_GRACE_SECONDS)
    except TimeoutError:
        _kill_process_group(proc.pid)
        try:
            await asyncio.wait_for(proc.wait(), timeout=TERMINATE_GRACE_SECONDS)
        except TimeoutError:
            log.warning("process group for pid %s survived SIGKILL; abandoning", proc.pid)


def _terminate_sync_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    _terminate_process_group(proc.pid)
    try:
        proc.wait(timeout=TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc.pid)
        try:
            proc.wait(timeout=TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            log.warning("process group for pid %s survived SIGKILL; abandoning", proc.pid)


def _terminate_process_group(pid: int) -> None:
    try:
        if os.name != "nt":
            os.killpg(pid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    except (PermissionError, ProcessLookupError):
        pass


def _kill_process_group(pid: int) -> None:
    try:
        if os.name != "nt":
            os.killpg(pid, signal.SIGKILL)
        else:
            os.kill(pid, signal.SIGTERM)
    except (PermissionError, ProcessLookupError):
        pass


def _timeout_error(
    cmd: Sequence[str],
    label: str,
    timeout_seconds: float | None,
    *,
    stdout: str = "",
    stderr: str = "",
) -> ExternalProcessError:
    timeout = _normalize_timeout(timeout_seconds)
    return ExternalProcessError(
        f"{label} timed out after {timeout:.0f}s; killed process group for {cmd[0]}",
        cmd=cmd,
        stdout=_tail(stdout),
        stderr=_tail(stderr),
    )


def _exit_error(
    cmd: Sequence[str],
    label: str,
    returncode: int | None,
    stdout: str,
    stderr: str,
) -> ExternalProcessError:
    detail = _tail(stderr) or _tail(stdout) or "no output"
    return ExternalProcessError(
        f"{label} failed with exit {returncode}: {detail}",
        cmd=cmd,
        returncode=returncode,
        stdout=_tail(stdout),
        stderr=_tail(stderr),
    )


def _tail(value: str, limit: int = 1000) -> str:
    return (value or "").strip()[-limit:]


def _decode(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return value.decode("utf-8", errors="replace")
