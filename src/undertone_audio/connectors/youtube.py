from __future__ import annotations

import os
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from undertone_audio.connectors.base import (
    ConnectorAsset,
    ConnectorError,
    compact_metadata,
    default_download_dir,
    ensure_binary,
    run_checked,
    run_json,
    safe_stem,
)
from undertone_audio.processes import process_timeout_from_env


class YouTubeConnector:
    name = "youtube"
    source_kind = "youtube-audio"

    def __init__(
        self,
        *,
        download_dir: Path | None = None,
        yt_dlp_bin: str = "yt-dlp",
        audio_format: str = "wav",
        include_playlist: bool = False,
        process_timeout_seconds: float | None = None,
    ):
        self.download_dir = download_dir or default_download_dir() / "youtube"
        self.yt_dlp_bin = yt_dlp_bin
        self.audio_format = audio_format
        self.include_playlist = include_playlist
        self.process_timeout_seconds = (
            process_timeout_from_env() if process_timeout_seconds is None else process_timeout_seconds
        )

    def matches(self, ref: str) -> bool:
        parsed = urlparse(ref)
        host = (parsed.hostname or "").lower().rstrip(".")
        return host == "youtube.com" or host.endswith(".youtube.com") or host == "youtu.be"

    def fetch(self, url: str) -> ConnectorAsset:
        binary = ensure_binary(self.yt_dlp_bin)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        info = run_json(self._info_cmd(binary, url), timeout_seconds=self.process_timeout_seconds)
        video_id = str(info.get("id") or _video_id_hint(url) or safe_stem(url))
        stem = safe_stem(video_id, fallback="youtube-audio")
        with tempfile.TemporaryDirectory(
            dir=self.download_dir,
            prefix=f".{stem}.",
        ) as tmpdir:
            tmp_download_dir = Path(tmpdir)
            run_checked(
                self._download_cmd(binary, url, stem, tmp_download_dir),
                timeout_seconds=self.process_timeout_seconds,
            )
            audio_path = self._publish_download(stem, tmp_download_dir)
        return ConnectorAsset(
            audio_path=audio_path,
            source_url=url,
            source_kind="youtube-audio",
            title=info.get("title"),
            transcript_id_hint=f"youtube-{video_id}",
            recorded_at=info.get("upload_date"),
            metadata=compact_metadata(
                {
                    "source": "youtube",
                    "youtube_video_id": video_id,
                    "title": info.get("title"),
                    "webpage_url": info.get("webpage_url") or url,
                    "channel": info.get("channel") or info.get("uploader"),
                    "duration_seconds": info.get("duration"),
                    "audio_priority": "downloaded-youtube-audio",
                    "audio_format": self.audio_format,
                }
            ),
        )

    def _info_cmd(self, binary: str, url: str) -> list[str]:
        cmd = [binary, "--dump-single-json", "--no-warnings"]
        if not self.include_playlist:
            cmd.append("--no-playlist")
        cmd.append(url)
        return cmd

    def _download_cmd(self, binary: str, url: str, stem: str, download_dir: Path | None = None) -> list[str]:
        download_dir = download_dir or self.download_dir
        cmd = [
            binary,
            "-f",
            "bestaudio/best",
            "-x",
            "--audio-format",
            self.audio_format,
            "--paths",
            str(download_dir),
            "-o",
            f"{stem}.%(ext)s",
        ]
        if not self.include_playlist:
            cmd.append("--no-playlist")
        cmd.append(url)
        return cmd

    def _publish_download(self, stem: str, tmp_download_dir: Path) -> Path:
        tmp_path = self._downloaded_path(stem, tmp_download_dir)
        if tmp_path.stat().st_size == 0:
            raise ConnectorError(f"yt-dlp produced an empty audio file for {stem}")
        dest = self.download_dir / tmp_path.name
        os.replace(tmp_path, dest)
        return dest

    def _downloaded_path(self, stem: str, download_dir: Path | None = None) -> Path:
        download_dir = download_dir or self.download_dir
        candidates = set(download_dir.glob(f"{stem}.*"))
        if not candidates:
            candidates = set(download_dir.glob(f"{stem}.{self.audio_format}"))
        if not candidates:
            raise ConnectorError(f"yt-dlp did not produce an audio file for {stem}")
        return max(candidates, key=lambda path: path.stat().st_mtime)


def _video_id_hint(url: str) -> str | None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().rstrip(".")
    if host == "youtu.be":
        return parsed.path.strip("/") or None
    if host != "youtube.com" and not host.endswith(".youtube.com"):
        return None
    values = parse_qs(parsed.query).get("v")
    return values[0] if values else None
