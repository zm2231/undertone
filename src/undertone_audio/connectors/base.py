from __future__ import annotations

import logging
import math
import os
import re
import shutil
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Mapping, Protocol, runtime_checkable
from urllib.parse import parse_qsl, quote, urlencode, unquote, urlsplit, urlunsplit

from undertone_audio.processes import load_json_text, run_process_sync
from undertone_audio.schema import ConnectorAssetSchema, ConnectorCandidateSchema

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
            source_url=_redact_url(self.source_url) or self.source_url,
            source_kind=self.source_kind,
            title=redact_url_values(self.title),
            transcript_id_hint=self.transcript_id_hint,
            recorded_at=self.recorded_at,
            metadata=redact_url_values(self.metadata),
        )


@dataclass(frozen=True)
class ConnectorCandidate:
    candidate_id: str
    original_url: str
    extractor: str | None = None
    extractor_key: str | None = None
    webpage_url: str | None = None
    url: str | None = None
    media_id: str | None = None
    format_id: str | None = None
    title: str | None = None
    duration: float | None = None
    kind: str = "generic-media"
    availability: str = "downloadable"
    reason: str | None = None
    metadata: dict = field(default_factory=dict)
    schema_version: str = "1"

    def to_schema(self) -> ConnectorCandidateSchema:
        if self.schema_version != "1":
            raise ConnectorError(f"unsupported ConnectorCandidate schema_version: {self.schema_version}")
        return ConnectorCandidateSchema(
            schema_version=self.schema_version,
            candidate_id=self.candidate_id,
            extractor=redact_url_values(self.extractor),
            extractor_key=redact_url_values(self.extractor_key),
            webpage_url=_redact_url(self.webpage_url),
            original_url=_redact_url(self.original_url) or self.original_url,
            url=_redact_url(self.url),
            media_id=redact_url_values(self.media_id),
            format_id=redact_url_values(self.format_id),
            title=redact_url_values(self.title),
            duration=self.duration,
            kind=self.kind,  # type: ignore[arg-type]
            availability=self.availability,  # type: ignore[arg-type]
            reason=redact_url_values(self.reason),
            metadata=redact_url_values(self.metadata),
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
    safe_ref = redact_url_values(ref)
    raise ConnectorError(
        f"no connector matched {safe_ref!r}; installed connectors: {names}. "
        "For article pages or arbitrary web URLs, use `undertone connector-resolve` "
        "or `undertone web-ingest`."
    )


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
    from undertone_audio.connectors.web import WebMediaConnector
    from undertone_audio.connectors.youtube import YouTubeConnector

    return [YouTubeConnector(), PodcastConnector(), WebMediaConnector()]


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
        raise ConnectorError(redact_url_values(str(exc))) from exc
    try:
        value = load_json_text(proc.stdout, producer=cmd[0])
    except Exception as exc:
        raise ConnectorError(redact_url_values(str(exc))) from exc
    return value


def run_checked(cmd: list[str], *, timeout_seconds: float | None = None):
    try:
        return run_process_sync(cmd, label=cmd[0], timeout_seconds=timeout_seconds)
    except RuntimeError as exc:
        raise ConnectorError(redact_url_values(str(exc))) from exc


def safe_stem(*parts: str | None, fallback: str = "audio") -> str:
    joined = "-".join(part for part in parts if part)
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", joined).strip(".-")
    return normalized[:160] or fallback


def compact_metadata(data: Mapping) -> dict:
    return {str(key): value for key, value in data.items() if value not in (None, "", [], {})}


def redact_url_values(value: Any):
    if isinstance(value, dict):
        return {_redact_mapping_key(key): redact_url_values(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact_url_values(item) for item in value]
    if isinstance(value, str):
        return _redact_url_text(value)
    return value


def _redact_mapping_key(value: Any):
    if isinstance(value, str):
        return re.sub(r"https?://[^\s'\"<>]+", lambda match: _redact_url(match.group(0)) or match.group(0), value)
    return redact_url_values(value)


def _redact_url_text(value: str) -> str:
    value = _redact_bare_tokens(value)
    if _looks_sensitive_token(value):
        return "[redacted]"
    parsed = urlsplit(value)
    if parsed.scheme and parsed.netloc:
        return _redact_url(value) or value
    return re.sub(r"https?://[^\s'\"<>]+", lambda match: _redact_url(match.group(0)) or match.group(0), value)


def _redact_url(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        return value
    host = parsed.hostname or ""
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port:
        host = f"{host}:{port}"
    return urlunsplit((parsed.scheme, host, _redact_url_path(parsed.path), _redact_query(parsed), ""))


def _redact_url_path(path: str) -> str:
    if not path:
        return ""
    segments = path.split("/")
    redacted = [_redact_path_segment(segment) for segment in segments]
    return "/".join(redacted)


def _redact_path_segment(segment: str) -> str:
    if not segment:
        return segment
    decoded = unquote(segment)
    if _looks_sensitive_token(decoded):
        return "[redacted]"
    return quote(decoded, safe=":@!$&'()*+,;=-._~")


def _looks_sensitive_token(value: str) -> bool:
    normalized = value.lower()
    if re.match(r"sk_(?:live|test)_[a-z0-9_=-]+$", normalized):
        return True
    if _looks_like_jwt(value):
        return True
    if _looks_like_opaque_token(value):
        return True
    return False


def _looks_like_jwt(value: str) -> bool:
    parts = value.split(".")
    if len(parts) != 3:
        return False
    return all(re.fullmatch(r"[A-Za-z0-9_-]{8,}", part) for part in parts)


def _looks_like_opaque_token(value: str) -> bool:
    if len(value) < 24:
        return False
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        return False
    has_mixed_case = bool(re.search(r"[a-z]", value)) and bool(re.search(r"[A-Z]", value))
    has_digit = bool(re.search(r"\d", value))
    if not (has_mixed_case and has_digit):
        return False
    return _shannon_entropy(value) >= 4.0


def _shannon_entropy(value: str) -> float:
    counts = {char: value.count(char) for char in set(value)}
    length = len(value)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def _redact_query(parsed) -> str:
    host = (parsed.hostname or "").lower().rstrip(".")
    if host in {"youtube.com", "www.youtube.com", "music.youtube.com", "m.youtube.com"}:
        preserved = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True) if key == "v"]
        return urlencode(preserved)
    return ""


def _redact_bare_tokens(value: str) -> str:
    value = re.sub(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [redacted]", value)
    value = re.sub(r"\bsk_(?:live|test)_[A-Za-z0-9_=-]+", "sk_[redacted]", value)
    return value
