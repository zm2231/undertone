from __future__ import annotations

import hashlib
import mimetypes
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path

from undertone_audio.connectors.base import ConnectorAsset, ConnectorError, compact_metadata, default_download_dir, safe_stem
from undertone_audio.processes import atomic_write_path


@dataclass(frozen=True)
class PodcastEpisode:
    title: str
    enclosure_url: str
    guid: str | None = None
    published_at: str | None = None
    episode_index: int = 0
    duration: str | None = None


class PodcastConnector:
    def __init__(self, *, download_dir: Path | None = None):
        self.download_dir = download_dir or default_download_dir() / "podcasts"

    def fetch(
        self,
        source: str,
        *,
        episode: int | None = None,
        title_contains: str | None = None,
    ) -> ConnectorAsset:
        self.download_dir.mkdir(parents=True, exist_ok=True)
        if _looks_like_audio_url(source):
            return self._download_direct(source)
        episode_row = self._select_episode(
            self._episodes(source),
            episode=episode,
            title_contains=title_contains,
        )
        audio_path = self._download(episode_row.enclosure_url, episode_row)
        source_id = episode_row.guid or _stable_id(episode_row.enclosure_url)
        return ConnectorAsset(
            audio_path=audio_path,
            source_url=source,
            source_kind="podcast-audio",
            title=episode_row.title,
            transcript_id_hint=f"podcast-{safe_stem(source_id)}",
            recorded_at=episode_row.published_at,
            metadata=compact_metadata(
                {
                    "source": "podcast",
                    "feed_url": source,
                    "episode_title": episode_row.title,
                    "episode_guid": episode_row.guid,
                    "episode_index": episode_row.episode_index,
                    "published_at": episode_row.published_at,
                    "duration": episode_row.duration,
                    "enclosure_url": episode_row.enclosure_url,
                    "audio_priority": "downloaded-podcast-audio",
                }
            ),
        )

    def list_episodes(self, source: str, *, limit: int = 20) -> list[PodcastEpisode]:
        return self._episodes(source)[:limit]

    def _episodes(self, feed_url: str) -> list[PodcastEpisode]:
        with urllib.request.urlopen(feed_url, timeout=30) as response:
            data = response.read()
        root = ET.fromstring(data)
        episodes = []
        for index, item in enumerate(root.findall(".//channel/item")):
            enclosure = item.find("enclosure")
            enclosure_url = enclosure.get("url") if enclosure is not None else None
            enclosure_type = enclosure.get("type") if enclosure is not None else None
            if not enclosure_url or (enclosure_type and not enclosure_type.startswith("audio/")):
                continue
            title = _text(item, "title") or f"Episode {index + 1}"
            published_at = _published_at(_text(item, "pubDate"))
            episodes.append(
                PodcastEpisode(
                    title=title,
                    enclosure_url=enclosure_url,
                    guid=_text(item, "guid"),
                    published_at=published_at,
                    episode_index=index,
                    duration=_itunes_duration(item),
                )
            )
        if not episodes:
            raise ConnectorError(f"no audio enclosures found in podcast feed: {feed_url}")
        return episodes

    def _select_episode(
        self,
        episodes: list[PodcastEpisode],
        *,
        episode: int | None,
        title_contains: str | None,
    ) -> PodcastEpisode:
        if title_contains:
            needle = title_contains.lower()
            matches = [row for row in episodes if needle in row.title.lower()]
            if not matches:
                raise ConnectorError(f"no podcast episode title contains {title_contains!r}")
            return matches[0]
        index = episode or 0
        if index < 0 or index >= len(episodes):
            raise ConnectorError(f"episode index {index} out of range; feed has {len(episodes)} episodes")
        return episodes[index]

    def _download_direct(self, source: str) -> ConnectorAsset:
        pseudo = PodcastEpisode(title=Path(urllib.parse.urlparse(source).path).stem, enclosure_url=source)
        audio_path = self._download(source, pseudo)
        source_id = _stable_id(source)
        return ConnectorAsset(
            audio_path=audio_path,
            source_url=source,
            source_kind="podcast-direct-audio",
            title=pseudo.title,
            transcript_id_hint=f"podcast-{source_id}",
            metadata=compact_metadata(
                {
                    "source": "podcast",
                    "audio_url": source,
                    "audio_priority": "direct-audio-url",
                }
            ),
        )

    def _download(self, url: str, episode: PodcastEpisode) -> Path:
        suffix = _suffix_for_url(url) or ".mp3"
        stem = safe_stem(episode.guid, episode.title, fallback=_stable_id(url))
        dest = self.download_dir / f"{stem}{suffix}"
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        with urllib.request.urlopen(url, timeout=120) as response:
            with atomic_write_path(dest) as tmp_path:
                with tmp_path.open("wb") as fp:
                    while chunk := response.read(1 << 20):
                        fp.write(chunk)
        return dest


def _looks_like_audio_url(value: str) -> bool:
    path = urllib.parse.urlparse(value).path.lower()
    return path.endswith((".mp3", ".m4a", ".wav", ".aac", ".ogg", ".opus", ".flac"))


def _suffix_for_url(url: str) -> str | None:
    suffix = Path(urllib.parse.urlparse(url).path).suffix
    if suffix:
        return suffix
    content_type, _encoding = mimetypes.guess_type(url)
    if content_type:
        return mimetypes.guess_extension(content_type)
    return None


def _text(item: ET.Element, tag: str) -> str | None:
    value = item.findtext(tag)
    return value.strip() if value and value.strip() else None


def _itunes_duration(item: ET.Element) -> str | None:
    for child in item:
        if child.tag.endswith("duration") and child.text and child.text.strip():
            return child.text.strip()
    return None


def _published_at(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).isoformat()
    except Exception:
        return value


def _stable_id(value: str) -> str:
    return hashlib.sha1(value.encode()).hexdigest()[:16]
