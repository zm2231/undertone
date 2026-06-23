from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from undertone_audio.commands.common import (
    add_audio_pipeline_flags,
    add_duplicate_flags,
    config_for_args,
    emit_transcript,
    guard_existing_transcript,
)
from undertone_audio.connectors import ConnectorAsset, PodcastConnector, YouTubeConnector
from undertone_audio.engines import create_engine
from undertone_audio.pipeline import AudioPipeline, _audio_format
from undertone_audio.storage import TranscriptStore


def register(subcommands: argparse._SubParsersAction) -> None:
    youtube = subcommands.add_parser(
        "youtube-ingest",
        help="Download YouTube audio with yt-dlp and rerun undertone local diarization.",
    )
    youtube.add_argument("url")
    youtube.add_argument("--download-dir", type=Path)
    youtube.add_argument("--yt-dlp-bin", default="yt-dlp")
    youtube.add_argument("--audio-format", default="wav")
    youtube.add_argument("--include-playlist", action="store_true")
    youtube.add_argument("--transcript-id")
    youtube.add_argument("--dry-run", action="store_true")
    youtube.add_argument("--json", action="store_true", help="Print machine-readable JSON for dry runs.")
    add_audio_pipeline_flags(youtube)
    add_duplicate_flags(youtube)
    youtube.set_defaults(func=youtube_ingest_cmd)

    podcast_list = subcommands.add_parser(
        "podcast-list",
        help="List audio episodes from an RSS feed.",
    )
    podcast_list.add_argument("feed_url")
    podcast_list.add_argument("--download-dir", type=Path)
    podcast_list.add_argument("--limit", type=int, default=20)
    podcast_list.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    podcast_list.set_defaults(func=podcast_list_cmd)

    podcast = subcommands.add_parser(
        "podcast-ingest",
        help="Download podcast episode audio and rerun undertone local diarization.",
    )
    podcast.add_argument("source", help="Podcast RSS feed URL or direct audio URL.")
    podcast.add_argument("--download-dir", type=Path)
    podcast.add_argument("--episode", type=int, help="Zero-based episode index from the feed.")
    podcast.add_argument("--title-contains")
    podcast.add_argument("--transcript-id")
    podcast.add_argument("--dry-run", action="store_true")
    podcast.add_argument("--json", action="store_true", help="Print machine-readable JSON for dry runs.")
    add_audio_pipeline_flags(podcast)
    add_duplicate_flags(podcast)
    podcast.set_defaults(func=podcast_ingest_cmd)


def youtube_ingest_cmd(args: argparse.Namespace) -> int:
    connector = YouTubeConnector(
        download_dir=args.download_dir,
        yt_dlp_bin=args.yt_dlp_bin,
        audio_format=args.audio_format,
        include_playlist=args.include_playlist,
    )
    asset = connector.fetch(args.url)
    if args.dry_run:
        payload = _asset_payload(asset)
        print(json.dumps(payload, separators=(",", ":")) if args.json else _render_asset(payload))
        return 0
    return _ingest_asset(asset, args)


def podcast_list_cmd(args: argparse.Namespace) -> int:
    connector = PodcastConnector(download_dir=args.download_dir)
    rows = [
        {
            "index": episode.episode_index,
            "title": episode.title,
            "guid": episode.guid,
            "published_at": episode.published_at,
            "duration": episode.duration,
            "enclosure_url": episode.enclosure_url,
        }
        for episode in connector.list_episodes(args.feed_url, limit=args.limit)
    ]
    print(json.dumps(rows, separators=(",", ":")) if args.json else _render_podcast_list(rows))
    return 0


def podcast_ingest_cmd(args: argparse.Namespace) -> int:
    connector = PodcastConnector(download_dir=args.download_dir)
    asset = connector.fetch(args.source, episode=args.episode, title_contains=args.title_contains)
    if args.dry_run:
        payload = _asset_payload(asset)
        print(json.dumps(payload, separators=(",", ":")) if args.json else _render_asset(payload))
        return 0
    return _ingest_asset(asset, args)


def _ingest_asset(asset: ConnectorAsset, args: argparse.Namespace) -> int:
    config = config_for_args(args)
    store = TranscriptStore(config.db_path)
    try:
        transcript_id = args.transcript_id or asset.transcript_id_hint
        if guard_existing_transcript(store, transcript_id, args):
            return 0
        engine = create_engine(args.engine, config)
        pipeline = AudioPipeline(store=store, engine=engine, config=config)
        raw = asyncio.run(engine.transcribe(asset.audio_path))
        transcript = pipeline.finalize_raw(
            raw,
            transcript_id=transcript_id,
            recorded_at=_parse_datetime(asset.recorded_at),
            source_path=str(asset.audio_path),
            source_url=asset.source_url,
            source_metadata=asset.metadata,
            expected_speaker_count=args.expected_speaker_count,
            expected_speaker_source=args.expected_speaker_source,
            audio_format=_audio_format(asset.audio_path),
            audio_path=asset.audio_path,
        )
        emit_transcript(transcript, args, raw=raw)
        return 0
    finally:
        store.close()


def _asset_payload(asset: ConnectorAsset) -> dict:
    return {
        "audio_path": str(asset.audio_path),
        "source_url": asset.source_url,
        "source_kind": asset.source_kind,
        "title": asset.title,
        "transcript_id_hint": asset.transcript_id_hint,
        "recorded_at": asset.recorded_at,
        "metadata": asset.metadata,
    }


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    if len(value) == 8 and value.isdigit():
        value = f"{value[:4]}-{value[4:6]}-{value[6:]}T00:00:00"
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _render_asset(payload: dict) -> str:
    lines = [
        "Connector asset",
        f"  title:         {payload['title'] or '-'}",
        f"  source kind:   {payload['source_kind']}",
        f"  source url:    {payload['source_url']}",
        f"  audio path:    {payload['audio_path']}",
        f"  transcript id: {payload['transcript_id_hint']}",
        f"  recorded at:   {payload['recorded_at'] or '-'}",
    ]
    return "\n".join(lines)


def _render_podcast_list(rows: list[dict]) -> str:
    if not rows:
        return "No podcast episodes found."
    lines = ["Podcast episodes"]
    for row in rows:
        duration = f" ({row['duration']})" if row.get("duration") else ""
        published = row.get("published_at") or "-"
        lines.append(f"  [{row['index']}] {row['title']}{duration}")
        lines.append(f"      published: {published}")
        lines.append(f"      audio:     {row['enclosure_url']}")
    return "\n".join(lines)
