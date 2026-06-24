from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from undertone_audio.processes import load_json_text, run_process_sync


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


def run_json(cmd: list[str], *, timeout_seconds: float | None = None) -> dict:
    try:
        proc = run_process_sync(cmd, label=cmd[0], timeout_seconds=timeout_seconds)
    except RuntimeError as exc:
        raise ConnectorError(str(exc)) from exc
    try:
        value = load_json_text(proc.stdout, producer=cmd[0])
    except Exception as exc:
        raise ConnectorError(str(exc)) from exc
    return value


def run_checked(cmd: list[str], *, timeout_seconds: float | None = None):
    try:
        return run_process_sync(cmd, label=cmd[0], timeout_seconds=timeout_seconds)
    except RuntimeError as exc:
        raise ConnectorError(str(exc)) from exc


def safe_stem(*parts: str | None, fallback: str = "audio") -> str:
    joined = "-".join(part for part in parts if part)
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", joined).strip(".-")
    return normalized[:160] or fallback


def compact_metadata(data: Mapping) -> dict:
    return {str(key): value for key, value in data.items() if value not in (None, "", [], {})}
