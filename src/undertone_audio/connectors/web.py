from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import socket
import tempfile
from dataclasses import replace
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse

from undertone_audio.connectors.base import (
    ConnectorAsset,
    ConnectorCandidate,
    ConnectorError,
    compact_metadata,
    default_download_dir,
    ensure_binary,
    redact_url_values,
    run_checked,
    run_json,
    safe_stem,
)
from undertone_audio.processes import process_timeout_from_env

DEFAULT_MAX_DOWNLOAD_SIZE = "2G"
VIDEO_EXTRACTORS = {
    "youtube",
    "vimeo",
    "dailymotion",
    "twitch",
    "rumble",
}
TTS_HINTS = ("tts", "voiceover", "text-to-speech", "text_to_speech", "read aloud", "read-aloud")
AUDIO_EXTENSIONS = (".mp3", ".m4a", ".wav", ".aac", ".ogg", ".opus", ".flac")
VOLATILE_ID_QUERY_KEYS = {
    "access_token",
    "awsaccesskeyid",
    "credential",
    "expires",
    "expire",
    "key-pair-id",
    "key_pair_id",
    "policy",
    "signature",
    "sig",
    "token",
    "x-amz-algorithm",
    "x-amz-credential",
    "x-amz-date",
    "x-amz-expires",
    "x-amz-security-token",
    "x-amz-signature",
    "x-amz-signedheaders",
}


class WebMediaConnector:
    name = "web"
    source_kind = "web-media-audio"

    def __init__(
        self,
        *,
        download_dir: Path | None = None,
        yt_dlp_bin: str = "yt-dlp",
        audio_format: str = "wav",
        process_timeout_seconds: float | None = None,
        max_download_size: str | None = None,
        cookies: Path | None = None,
        cookies_from_browser: str | None = None,
    ):
        self.download_dir = download_dir or default_download_dir() / "web"
        self.yt_dlp_bin = yt_dlp_bin
        self.audio_format = audio_format
        self.process_timeout_seconds = (
            process_timeout_from_env() if process_timeout_seconds is None else process_timeout_seconds
        )
        self.max_download_size = max_download_size or os.environ.get("UNDERTONE_MAX_DOWNLOAD_SIZE") or DEFAULT_MAX_DOWNLOAD_SIZE
        self.cookies = cookies
        self.cookies_from_browser = cookies_from_browser

    def matches(self, ref: str) -> bool:
        return False

    def resolve(self, ref: str) -> list[ConnectorCandidate]:
        _validate_public_url(ref)
        binary = ensure_binary(self.yt_dlp_bin)
        try:
            info = run_json(self._info_cmd(binary, ref), timeout_seconds=self.process_timeout_seconds)
        except ConnectorError as exc:
            return [
                ConnectorCandidate(
                    candidate_id=_candidate_id("unsupported", None, ref),
                    original_url=ref,
                    kind="unsupported",
                    availability="unsupported",
                    reason=redact_url_values(str(exc)),
                )
            ]
        candidates = _candidates_from_info(info, original_url=ref)
        if not candidates:
            return [
                ConnectorCandidate(
                    candidate_id=_candidate_id("unsupported", None, ref),
                    original_url=ref,
                    kind="unsupported",
                    availability="unsupported",
                    reason="yt-dlp did not report downloadable media candidates",
                )
            ]
        return sorted(candidates, key=_candidate_rank)

    def fetch(self, ref: str) -> ConnectorAsset:
        candidates = self.resolve(ref)
        downloadable = [row for row in candidates if row.availability == "downloadable"]
        if not downloadable:
            reason = candidates[0].reason if candidates else "no downloadable media candidates"
            raise ConnectorError(reason or "no downloadable media candidates")
        if len(downloadable) > 1:
            ids = ", ".join(row.candidate_id for row in downloadable[:5])
            raise ConnectorError(
                "web media URL is ambiguous; run `undertone connector-resolve` and then "
                f"`undertone web-ingest --select <candidate-id>`. candidates: {ids}"
            )
        return self.fetch_candidate(downloadable[0])

    def fetch_candidate(self, candidate: ConnectorCandidate) -> ConnectorAsset:
        if candidate.availability != "downloadable":
            raise ConnectorError(candidate.reason or f"candidate {candidate.candidate_id} is not downloadable")
        _validate_candidate_public_urls(candidate)
        ref = _candidate_download_ref(candidate)
        binary = ensure_binary(self.yt_dlp_bin)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        stem = safe_stem(candidate.candidate_id, fallback="web-media")
        with tempfile.TemporaryDirectory(dir=self.download_dir, prefix=f".{stem}.") as tmpdir:
            tmp_download_dir = Path(tmpdir)
            run_checked(
                self._download_cmd(binary, ref, stem, tmp_download_dir),
                timeout_seconds=self.process_timeout_seconds,
            )
            audio_path = self._publish_download(stem, tmp_download_dir)
        return ConnectorAsset(
            audio_path=audio_path,
            source_url=redact_url_values(candidate.original_url),
            source_kind=self.source_kind,
            title=candidate.title,
            transcript_id_hint=f"web-{candidate.candidate_id}",
            metadata=compact_metadata(
                {
                    "source": "web",
                    "audio_priority": "resolved-web-media",
                    "candidate_id": candidate.candidate_id,
                    "candidate_kind": candidate.kind,
                    "candidate_availability": candidate.availability,
                    "extractor": candidate.extractor,
                    "extractor_key": candidate.extractor_key,
                    "webpage_url": redact_url_values(candidate.webpage_url),
                    "media_id": redact_url_values(candidate.media_id),
                    "format_id": redact_url_values(candidate.format_id),
                    "duration_seconds": candidate.duration,
                    "audio_format": self.audio_format,
                }
            ),
        )

    def _info_cmd(self, binary: str, url: str) -> list[str]:
        cmd = [
            binary,
            "--ignore-config",
            "--dump-single-json",
            "--no-warnings",
            "--no-playlist",
        ]
        cmd.extend(self._auth_args())
        cmd.append(url)
        return cmd

    def _download_cmd(self, binary: str, url: str, stem: str, download_dir: Path) -> list[str]:
        cmd = [
            binary,
            "--ignore-config",
            "-f",
            "bestaudio/best",
            "-x",
            "--audio-format",
            self.audio_format,
            "--max-filesize",
            self.max_download_size,
            "--paths",
            str(download_dir),
            "-o",
            f"{stem}.%(ext)s",
            "--no-playlist",
        ]
        cmd.extend(self._auth_args())
        cmd.append(url)
        return cmd

    def _auth_args(self) -> list[str]:
        args: list[str] = []
        if self.cookies:
            args.extend(["--cookies", str(self.cookies)])
        if self.cookies_from_browser:
            args.extend(["--cookies-from-browser", self.cookies_from_browser])
        return args

    def _publish_download(self, stem: str, tmp_download_dir: Path) -> Path:
        tmp_path = self._downloaded_path(stem, tmp_download_dir)
        if tmp_path.stat().st_size == 0:
            raise ConnectorError(f"yt-dlp produced an empty audio file for {stem}")
        dest = self.download_dir / tmp_path.name
        os.replace(tmp_path, dest)
        return dest

    def _downloaded_path(self, stem: str, download_dir: Path) -> Path:
        candidates = set(download_dir.glob(f"{stem}.*"))
        if not candidates:
            candidates = set(download_dir.glob(f"{stem}.{self.audio_format}"))
        if not candidates:
            raise ConnectorError(f"yt-dlp did not produce an audio file for {stem}")
        return max(candidates, key=lambda path: path.stat().st_mtime)


def select_candidate(candidates: list[ConnectorCandidate], candidate_id: str | None) -> ConnectorCandidate:
    downloadable = [row for row in candidates if row.availability == "downloadable"]
    if candidate_id:
        for row in candidates:
            if row.candidate_id == candidate_id:
                if row.availability != "downloadable":
                    raise ConnectorError(row.reason or f"candidate {candidate_id} is not downloadable")
                return row
        raise ConnectorError(f"candidate not found: {candidate_id}")
    if not downloadable:
        reason = candidates[0].reason if candidates else "no downloadable media candidates"
        raise ConnectorError(reason or "no downloadable media candidates")
    return downloadable[0]


def _validate_public_url(value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        raise ConnectorError("web media URLs must use http or https")
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        raise ConnectorError("web media URL is missing a hostname")
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        raise ConnectorError(f"refusing private/local host: {host}")
    try:
        addresses = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise ConnectorError(f"could not resolve host {host}: {exc}") from exc
    for item in addresses:
        ip = ipaddress.ip_address(item[4][0])
        if not ip.is_global:
            raise ConnectorError(f"refusing private/local host: {host} resolves to {ip}")


def _validate_candidate_public_urls(candidate: ConnectorCandidate) -> None:
    seen: set[str] = set()
    for value in (candidate.original_url, candidate.webpage_url, candidate.url):
        if not value or value in seen:
            continue
        seen.add(value)
        _validate_public_url(value)


def _candidates_from_info(info: dict, *, original_url: str) -> list[ConnectorCandidate]:
    entries = info.get("entries")
    rows = entries if isinstance(entries, list) and entries else [info]
    candidates: list[ConnectorCandidate] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        candidate = _candidate_from_row(row, original_url=original_url)
        candidates.append(candidate)
    return _disambiguate_candidate_ids(candidates)


def _disambiguate_candidate_ids(candidates: list[ConnectorCandidate]) -> list[ConnectorCandidate]:
    candidates = _dedupe_stable_candidates(candidates)
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.candidate_id] = counts.get(candidate.candidate_id, 0) + 1
    if all(count == 1 for count in counts.values()):
        return candidates
    rows = [
        replace(candidate, candidate_id=_disambiguated_candidate_id(candidate))
        if counts[candidate.candidate_id] > 1
        else candidate
        for candidate in candidates
    ]
    seen_ids: set[str] = set()
    unique: list[ConnectorCandidate] = []
    for candidate in rows:
        if candidate.candidate_id in seen_ids:
            continue
        seen_ids.add(candidate.candidate_id)
        unique.append(candidate)
    return unique


def _dedupe_stable_candidates(candidates: list[ConnectorCandidate]) -> list[ConnectorCandidate]:
    seen: set[str] = set()
    unique: list[ConnectorCandidate] = []
    for candidate in candidates:
        key = _candidate_stable_payload(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _disambiguated_candidate_id(candidate: ConnectorCandidate) -> str:
    seed = _candidate_stable_payload(candidate)
    return hashlib.blake2b(seed.encode("utf-8"), digest_size=10).hexdigest()


def _candidate_stable_payload(candidate: ConnectorCandidate) -> str:
    payload = {
        "base": candidate.candidate_id,
        "extractor": candidate.extractor,
        "extractor_key": candidate.extractor_key,
        "webpage_url": _url_identity(candidate.webpage_url) if candidate.webpage_url else None,
        "url": _url_identity(candidate.url) if candidate.url else None,
        "media_id": candidate.media_id,
        "format_id": candidate.format_id,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _candidate_from_row(row: dict, *, original_url: str) -> ConnectorCandidate:
    extractor_key = _as_str(row.get("extractor_key"))
    extractor = _as_str(row.get("extractor"))
    media_id = _as_str(row.get("id"))
    webpage_url = _as_str(row.get("webpage_url") or row.get("original_url"))
    media_url = _as_str(row.get("url"))
    identity_url = _candidate_identity_url(extractor_key, extractor, webpage_url, media_url)
    kind = _candidate_kind(row, original_url=original_url)
    availability = _candidate_availability(
        row,
        kind=kind,
        original_url=original_url,
        webpage_url=webpage_url,
        media_url=media_url,
    )
    reason = _candidate_reason(
        row,
        availability=availability,
        kind=kind,
        original_url=original_url,
        webpage_url=webpage_url,
        media_url=media_url,
    )
    candidate = ConnectorCandidate(
        candidate_id=_candidate_id(
            extractor_key or extractor,
            media_id,
            original_url,
            fallback_url=identity_url,
        ),
        extractor=extractor,
        extractor_key=extractor_key,
        webpage_url=webpage_url,
        original_url=original_url,
        url=media_url,
        media_id=media_id,
        format_id=_as_str(row.get("format_id")),
        title=_as_str(row.get("title")),
        duration=_as_float(row.get("duration")),
        kind=kind,
        availability=availability,
        reason=reason,
        metadata=compact_metadata(
            {
                "ext": row.get("ext"),
                "protocol": row.get("protocol"),
                "availability": row.get("availability"),
                "live_status": row.get("live_status"),
            }
        ),
    )
    if candidate.availability == "requires-auth" and not candidate.reason:
        return replace(candidate, reason="media requires authentication or subscription")
    return candidate


def _candidate_identity_url(
    extractor_key: str | None,
    extractor: str | None,
    webpage_url: str | None,
    media_url: str | None,
) -> str | None:
    key = (extractor_key or extractor or "").lower()
    page_host = (urlparse(webpage_url or "").hostname or "").lower().rstrip(".")
    if key in VIDEO_EXTRACTORS or page_host in {"youtu.be", "youtube.com"} or page_host.endswith(".youtube.com"):
        return webpage_url or media_url
    return media_url or webpage_url


def _candidate_id(
    extractor_key: str | None,
    media_id: str | None,
    original_url: str,
    *,
    fallback_url: str | None = None,
) -> str:
    candidate_identity = "" if media_id else (_url_identity(fallback_url) if fallback_url else "")
    seed = "\x1f".join(
        [
            extractor_key or "",
            media_id or "",
            _url_identity(original_url),
            candidate_identity,
        ]
    )
    return hashlib.blake2b(seed.encode("utf-8"), digest_size=10).hexdigest()


def _url_identity(value: str) -> str:
    parsed = urlparse(value)
    if not parsed.scheme or not parsed.netloc:
        return value
    host = parsed.hostname or ""
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port:
        host = f"{host}:{port}"
    identity_query = _identity_query(parsed.query)
    return parsed._replace(netloc=host, query=identity_query, fragment="").geturl()


def _identity_query(query: str) -> str:
    pairs = [
        (key, value)
        for key, value in parse_qsl(query, keep_blank_values=True)
        if not _is_volatile_identity_query_param(key)
    ]
    return urlencode(sorted(pairs))


def _is_volatile_identity_query_param(key: str) -> bool:
    normalized = key.lower()
    return normalized in VOLATILE_ID_QUERY_KEYS or normalized.startswith("x-amz-")


def _candidate_kind(row: dict, *, original_url: str) -> str:
    extractor_key = (_as_str(row.get("extractor_key")) or "").lower()
    webpage_url = _as_str(row.get("webpage_url") or row.get("original_url") or row.get("url")) or ""
    title = (_as_str(row.get("title")) or "").lower()
    ext = (_as_str(row.get("ext")) or "").lower()
    host = (urlparse(webpage_url).hostname or "").lower().rstrip(".")
    if any(hint in title or hint in extractor_key for hint in TTS_HINTS):
        return "page-voiceover"
    if extractor_key in VIDEO_EXTRACTORS or host in {"youtu.be", "youtube.com"} or host.endswith(".youtube.com"):
        return "external-video"
    if ext in {item.lstrip(".") for item in AUDIO_EXTENSIONS} or urlparse(webpage_url).path.lower().endswith(AUDIO_EXTENSIONS):
        return "podcast-enclosure"
    return "generic-media"


def _candidate_availability(
    row: dict,
    *,
    kind: str,
    original_url: str,
    webpage_url: str | None,
    media_url: str | None,
) -> str:
    availability = (_as_str(row.get("availability")) or "").lower()
    if any(token in availability for token in ("premium", "subscriber", "private", "login", "auth")):
        return "requires-auth"
    if row.get("is_live") and row.get("live_status") not in {None, "was_live", "not_live"}:
        return "found-but-unavailable"
    if not _candidate_download_ref_or_none(
        kind=kind,
        original_url=original_url,
        webpage_url=webpage_url,
        media_url=media_url,
    ):
        return "found-but-unavailable"
    return "downloadable"


def _candidate_reason(
    row: dict,
    *,
    availability: str,
    kind: str,
    original_url: str,
    webpage_url: str | None,
    media_url: str | None,
) -> str | None:
    source_availability = _as_str(row.get("availability"))
    live_status = _as_str(row.get("live_status"))
    if source_availability and source_availability not in {"public", "unlisted"}:
        return f"availability: {source_availability}"
    if live_status and live_status not in {"was_live", "not_live"}:
        return f"live_status: {live_status}"
    if availability == "found-but-unavailable" and not _candidate_download_ref_or_none(
        kind=kind,
        original_url=original_url,
        webpage_url=webpage_url,
        media_url=media_url,
    ):
        return "candidate is not directly downloadable; select a candidate with a concrete media or extractor URL"
    return None


def _candidate_rank(candidate: ConnectorCandidate) -> tuple[int, int, float, str]:
    if candidate.availability != "downloadable":
        availability_score = 1
    else:
        availability_score = 0
    tier_score = {
        "external-video": 0,
        "podcast-enclosure": 0,
        "generic-media": 0,
        "page-voiceover": 3,
        "unsupported": 4,
    }.get(candidate.kind, 2)
    duration_score = -(candidate.duration or 0.0)
    return (availability_score, tier_score, duration_score, candidate.candidate_id)


def _candidate_download_ref(candidate: ConnectorCandidate) -> str:
    ref = _candidate_download_ref_or_none(
        kind=candidate.kind,
        original_url=candidate.original_url,
        webpage_url=candidate.webpage_url,
        media_url=candidate.url,
    )
    if not ref:
        raise ConnectorError(
            candidate.reason
            or "candidate is not directly downloadable; select a candidate with a concrete media or extractor URL"
        )
    return ref


def _candidate_download_ref_or_none(
    *,
    kind: str,
    original_url: str,
    webpage_url: str | None,
    media_url: str | None,
) -> str | None:
    if kind == "external-video" and webpage_url and _url_identity(webpage_url) != _url_identity(original_url):
        return webpage_url
    if media_url and _url_identity(media_url) != _url_identity(original_url):
        return media_url
    if _looks_like_direct_audio_url(webpage_url):
        return webpage_url
    if _looks_like_direct_audio_url(original_url):
        return original_url
    return None


def _looks_like_direct_audio_url(value: str | None) -> bool:
    if not value:
        return False
    return urlparse(value).path.lower().endswith(AUDIO_EXTENSIONS)


def _as_str(value) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _as_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
