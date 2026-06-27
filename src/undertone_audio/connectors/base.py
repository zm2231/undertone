from __future__ import annotations

import os
import re
import shutil
import logging
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Mapping, Protocol, runtime_checkable

from undertone_audio.processes import load_json_text, run_process_sync
from undertone_audio.schema import ConnectorAssetSchema

log = logging.getLogger(__name__)


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
    schema_version: str = "1"

    def to_schema(self) -> ConnectorAssetSchema:
        if self.schema_version != "1":
            raise ConnectorError(f"unsupported ConnectorAsset schema_version: {self.schema_version}")
        return ConnectorAssetSchema(
            schema_version=self.schema_version,
            audio_path=str(self.audio_path),
            source_url=self.source_url,
            source_kind=self.source_kind,
            title=self.title,
            transcript_id_hint=self.transcript_id_hint,
            recorded_at=self.recorded_at,
            metadata=self.metadata,
        )


@runtime_checkable
class Connector(Protocol):
    name: str
    source_kind: str

    def matches(self, ref: str) -> bool: ...

    def fetch(self, ref: str) -> ConnectorAsset: ...


def discover_connectors() -> list[Connector]:
    connectors: list[Connector] = _builtin_connectors()
    seen: set[str] = {connector.name for connector in connectors}
    for connector in _entrypoint_connectors():
        if connector.name in seen:
            if connector.__class__.__module__.startswith("undertone_audio.connectors."):
                continue
            raise ConnectorError(f"duplicate connector name: {connector.name}")
        connectors.append(connector)
        seen.add(connector.name)
    return connectors


def connector_for_ref(ref: str, preferred: str | None = None) -> Connector:
    connectors = discover_connectors()
    if preferred:
        for connector in connectors:
            if connector.name == preferred:
                return connector
        raise ConnectorError(f"connector not found: {preferred}")
    for connector in connectors:
        if connector.matches(ref):
            return connector
    names = ", ".join(connector.name for connector in connectors) or "none"
    raise ConnectorError(f"no connector matched {ref!r}; installed connectors: {names}")


def _entrypoint_connectors() -> list[Connector]:
    found: list[Connector] = []
    try:
        group = entry_points(group="undertone.connectors")
    except TypeError:
        group = entry_points().get("undertone.connectors", [])
    for item in group:
        try:
            factory = item.load()
            connector = factory() if callable(factory) else factory
        except Exception as exc:
            log.warning("skipping undertone connector %s: %s", item.name, exc)
            continue
        if not isinstance(connector, Connector):
            log.warning("skipping undertone connector %s: not a Connector", item.name)
            continue
        found.append(connector)
    return found


def _builtin_connectors() -> list[Connector]:
    from undertone_audio.connectors.podcast import PodcastConnector
    from undertone_audio.connectors.youtube import YouTubeConnector

    return [YouTubeConnector(), PodcastConnector()]


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
