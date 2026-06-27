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
    emit_progress,
    emit_transcript,
    guard_existing_transcript,
    progress_warning_sink,
)
from undertone_audio.connectors import (
    ConnectorAsset,
    ConnectorError,
    PodcastConnector,
    YouTubeConnector,
    connector_for_ref,
    discover_connectors,
)
from undertone_audio.connectors.podcast import _looks_like_audio_url, _stable_id as _podcast_stable_id
from undertone_audio.connectors.youtube import _video_id_hint
from undertone_audio.engines import create_engine
from undertone_audio.pipeline import AudioPipeline, _audio_format
from undertone_audio.storage import TranscriptStore


def register(subcommands: argparse._SubParsersAction) -> None:
    connector_list = subcommands.add_parser(
        "connector-list",
        help="List installed Undertone connector plugins.",
    )
    connector_list.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    connector_list.set_defaults(func=connector_list_cmd)

    connector_ingest = subcommands.add_parser(
        "connector-ingest",
        help="Fetch audio with a discovered connector and run local Undertone diarization.",
    )
    connector_ingest.add_argument("ref", help="Source URL/path/ref accepted by an installed connector.")
    connector_ingest.add_argument("--connector", help="Connector name. Defaults to first match.")
    connector_ingest.add_argument("--transcript-id")
    connector_ingest.add_argument("--dry-run", action="store_true")
    connector_ingest.add_argument("--json", action="store_true", help="Print machine-readable dry-run JSON.")
    add_audio_pipeline_flags(connector_ingest)
    add_duplicate_flags(connector_ingest)
    connector_ingest.set_defaults(func=connector_ingest_cmd)

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
    config = config_for_args(args)
    video_id_hint = None if args.include_playlist else _video_id_hint(args.url)
    early_transcript_id = args.transcript_id or (
        f"youtube-{video_id_hint}" if video_id_hint else None
    )
    if not args.dry_run and _guard_existing_before_fetch(early_transcript_id, args, config):
        return 0
    connector = YouTubeConnector(
        download_dir=args.download_dir,
        yt_dlp_bin=args.yt_dlp_bin,
        audio_format=args.audio_format,
        include_playlist=args.include_playlist,
        process_timeout_seconds=config.process_timeout_seconds,
    )
    asset = connector.fetch(args.url)
    if args.dry_run:
        payload = _asset_payload(asset)
        print(json.dumps(payload, separators=(",", ":")) if args.json else _render_asset(payload))
        return 0
    return _ingest_asset(asset, args, config=config)


def connector_list_cmd(args: argparse.Namespace) -> int:
    rows = [
        {"name": connector.name, "source_kind": connector.source_kind}
        for connector in discover_connectors()
    ]
    if args.json:
        print(json.dumps(rows, separators=(",", ":")))
    else:
        lines = ["Connectors"]
        lines.extend(f"  {row['name']}  {row['source_kind']}" for row in rows)
        print("\n".join(lines) if rows else "No connectors installed.")
    return 0


def connector_ingest_cmd(args: argparse.Namespace) -> int:
    config = config_for_args(args)
    early_transcript_id = args.transcript_id or _known_connector_transcript_id(
        args.ref,
        preferred=args.connector,
    )
    if not args.dry_run and _guard_existing_before_fetch(early_transcript_id, args, config):
        return 0
    connector = connector_for_ref(args.ref, preferred=args.connector)
    emit_progress(args, "fetching", connector=connector.name, ref=args.ref)
    asset = connector.fetch(args.ref)
    if not isinstance(asset, ConnectorAsset):
        raise ConnectorError(
            f"connector {connector.name!r} returned {type(asset).__name__}; expected ConnectorAsset"
        )
    if args.dry_run:
        payload = _asset_payload(asset)
        print(json.dumps(payload, separators=(",", ":")) if args.json else _render_asset(payload))
        return 0
    return _ingest_asset(asset, args, config=config, connector_name=connector.name)


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
    config = config_for_args(args)
    early_transcript_id = args.transcript_id
    if not early_transcript_id and _looks_like_audio_url(args.source):
        early_transcript_id = f"podcast-{_podcast_stable_id(args.source)}"
    if not args.dry_run and _guard_existing_before_fetch(early_transcript_id, args, config):
        return 0
    connector = PodcastConnector(download_dir=args.download_dir)
    asset = connector.fetch(args.source, episode=args.episode, title_contains=args.title_contains)
    if args.dry_run:
        payload = _asset_payload(asset)
        print(json.dumps(payload, separators=(",", ":")) if args.json else _render_asset(payload))
        return 0
    return _ingest_asset(asset, args, config=config)


def _ingest_asset(
    asset: ConnectorAsset,
    args: argparse.Namespace,
    *,
    config=None,
    connector_name: str | None = None,
) -> int:
    config = config or config_for_args(args)
    store = TranscriptStore(config.db_path)
    try:
        transcript_id = args.transcript_id or asset.transcript_id_hint
        if guard_existing_transcript(store, transcript_id, args):
            emit_progress(args, "skipped", transcript_id=transcript_id, reason="exists")
            return 0
        engine = create_engine(args.engine, config)
        emit_progress(
            args,
            "start",
            command=getattr(args, "command", "connector-ingest"),
            connector=connector_name,
            transcript_id=transcript_id,
            engine=getattr(engine, "name", config.default_engine),
            audio_path=str(asset.audio_path),
        )
        pipeline = AudioPipeline(
            store=store,
            engine=engine,
            config=config,
            warning_sink=progress_warning_sink(args),
        )
        raw = asyncio.run(engine.transcribe(asset.audio_path))
        emit_progress(
            args,
            "transcribed",
            transcript_id=transcript_id,
            duration_ms=raw.duration_ms,
            segments=len(raw.segments),
            speakers=len(raw.speakers),
        )
        emit_progress(args, "finalizing", transcript_id=transcript_id)
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
        emit_progress(args, "saved", transcript_id=transcript.transcript_id)
        emit_transcript(transcript, args, raw=raw)
        return 0
    finally:
        store.close()


def _guard_existing_before_fetch(
    transcript_id: str | None,
    args: argparse.Namespace,
    config,
) -> bool:
    if not transcript_id:
        return False
    store = TranscriptStore(config.db_path)
    try:
        if guard_existing_transcript(store, transcript_id, args):
            emit_progress(args, "skipped", transcript_id=transcript_id, reason="exists")
            return True
        return False
    finally:
        store.close()


def _known_connector_transcript_id(ref: str, *, preferred: str | None = None) -> str | None:
    preferred_name = preferred.lower() if preferred else None
    video_id = _video_id_hint(ref)
    if video_id and preferred_name in {None, "youtube"}:
        return f"youtube-{video_id}"
    if _looks_like_audio_url(ref) and preferred_name in {None, "podcast"}:
        return f"podcast-{_podcast_stable_id(ref)}"
    return None


def _asset_payload(asset: ConnectorAsset) -> dict:
    return asset.to_schema().model_dump(mode="json")


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
