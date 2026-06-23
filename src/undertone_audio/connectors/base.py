from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


class ConnectorError(RuntimeError):
    pass


@dataclass(frozen=True)
class ConnectorAsset:
    audio_path: Path
    source_url: str
    source_kind: str
    title: str | None = None
    transcript_id_hint: str | None = None
    recorded_at: str | None = None
    metadata: dict = field(default_factory=dict)


def default_download_dir() -> Path:
    override = os.environ.get("UNDERTONE_DOWNLOAD_DIR")
    if override:
        return Path(override).expanduser()
    cache_home = os.environ.get("XDG_CACHE_HOME")
    if cache_home:
        return Path(cache_home).expanduser() / "undertone" / "downloads"
    return Path.home() / ".cache" / "undertone" / "downloads"


def ensure_binary(binary: str) -> str:
    resolved = shutil.which(binary)
    if resolved is None:
        raise ConnectorError(
            f"required binary not found on PATH: {binary}. "
            "Install connectors with `pip install -e '.[connectors]'`, "
            "install yt-dlp directly, or pass --yt-dlp-bin. "
            "Check readiness with `undertone doctor --check-yt-dlp`."
        )
    return resolved


def run_json(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise ConnectorError(_command_error(cmd, proc))
    try:
        import json

        value = json.loads(proc.stdout)
    except Exception as exc:
        raise ConnectorError(f"command did not return valid JSON: {cmd[0]}") from exc
    if not isinstance(value, dict):
        raise ConnectorError(f"command returned non-object JSON: {cmd[0]}")
    return value


def run_checked(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise ConnectorError(_command_error(cmd, proc))
    return proc


def safe_stem(*parts: str | None, fallback: str = "audio") -> str:
    joined = "-".join(part for part in parts if part)
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", joined).strip(".-")
    return normalized[:160] or fallback


def compact_metadata(data: Mapping) -> dict:
    return {str(key): value for key, value in data.items() if value not in (None, "", [], {})}


def _command_error(cmd: list[str], proc: subprocess.CompletedProcess[str]) -> str:
    stderr = proc.stderr.strip()
    stdout = proc.stdout.strip()
    detail = stderr or stdout or "no output"
    return f"{cmd[0]} failed with exit {proc.returncode}: {detail[:1000]}"
