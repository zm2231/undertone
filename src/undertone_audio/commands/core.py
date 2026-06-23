from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from undertone_audio.commands.common import (
    add_audio_pipeline_flags,
    add_duplicate_flags,
    config_for_args,
    db_path,
    emit_transcript,
    guard_existing_transcript,
    output_format,
)
from undertone_audio.engines import create_engine
from undertone_audio.engines.base import RawTranscript
from undertone_audio.export import OUTPUT_DETAIL_LEVELS, OUTPUT_FORMATS
from undertone_audio.pipeline import AudioPipeline
from undertone_audio.storage import TranscriptStore


def register(subcommands: argparse._SubParsersAction) -> None:
    finalize = subcommands.add_parser("finalize-json", help="Save a RawTranscript JSON file.")
    finalize.add_argument("json_path", type=Path, help="Path to RawTranscript JSON, or '-' for stdin.")
    finalize.add_argument("--transcript-id", required=True)
    finalize.add_argument("--recorded-at")
    finalize.add_argument("--source-path")
    finalize.add_argument("--source-url")
    finalize.add_argument("--video-path")
    finalize.add_argument("--expected-speaker-count", type=int)
    finalize.add_argument("--expected-speaker-source")
    finalize.add_argument("--source-metadata", type=Path)
    finalize.add_argument("--diarization-state")
    finalize.add_argument("--diarization-error-code")
    finalize.add_argument("--diarization-error-detail")
    _add_output_flags(finalize)
    add_duplicate_flags(finalize)
    finalize.set_defaults(func=finalize_json_cmd)

    run_wav = subcommands.add_parser("run-wav", help="Run local audio pipeline on a WAV file.")
    run_wav.add_argument("audio_path", type=Path)
    run_wav.add_argument("--transcript-id")
    run_wav.add_argument("--recorded-at")
    run_wav.add_argument("--source-metadata", type=Path)
    add_audio_pipeline_flags(run_wav)
    _add_feature_toggles(run_wav)
    add_duplicate_flags(run_wav)
    run_wav.set_defaults(func=run_wav_cmd)

    list_cmd = subcommands.add_parser("list", help="List saved transcripts.")
    list_cmd.add_argument("--limit", type=int, default=50)
    list_cmd.add_argument("--offset", type=int, default=0)
    list_cmd.add_argument("--source")
    list_cmd.add_argument("--meeting-type")
    list_cmd.add_argument("--diarization-state")
    list_cmd.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    list_cmd.set_defaults(func=list_transcripts_cmd)

    load = subcommands.add_parser("load", help="Load an EnrichedTranscript as JSON.")
    load.add_argument("transcript_id")
    _add_output_flags(load)
    load.set_defaults(func=load_cmd)

    reenrich = subcommands.add_parser(
        "reenrich",
        help="Rebuild enrichment from a saved RawTranscript without retranscribing audio.",
    )
    reenrich.add_argument("transcript_id")
    add_audio_pipeline_flags(reenrich)
    _add_feature_toggles(reenrich)
    reenrich.set_defaults(func=reenrich_cmd)

    search = subcommands.add_parser("search", help="Search raw segment text.")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=20)
    search.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    search.set_defaults(func=search_cmd)


def finalize_json_cmd(args: argparse.Namespace) -> int:
    raw = RawTranscript.model_validate(_read_json(args.json_path))
    store = _store(args)
    try:
        if guard_existing_transcript(store, args.transcript_id, args):
            return 0
        pipeline = AudioPipeline(store=store, config=config_for_args(args))
        transcript = pipeline.finalize_raw(
            raw,
            transcript_id=args.transcript_id,
            recorded_at=_parse_datetime(args.recorded_at),
            source_path=args.source_path,
            source_url=args.source_url,
            video_path=args.video_path,
            expected_speaker_count=args.expected_speaker_count,
            expected_speaker_source=args.expected_speaker_source,
            source_metadata=_load_source_metadata(args.source_metadata),
            diarization_state=args.diarization_state,
            diarization_error_code=args.diarization_error_code,
            diarization_error_detail=args.diarization_error_detail,
        )
        emit_transcript(transcript, args, raw=raw)
        return 0
    finally:
        store.close()


def run_wav_cmd(args: argparse.Namespace) -> int:
    if not args.audio_path.exists():
        raise ValueError(f"audio file not found: {args.audio_path}")
    config = config_for_args(args)
    store = TranscriptStore(config.db_path)
    try:
        if guard_existing_transcript(store, args.transcript_id, args):
            return 0
        engine = create_engine(args.engine, config)
        pipeline = AudioPipeline(store=store, engine=engine, config=config)
        raw = asyncio.run(engine.transcribe(args.audio_path))
        transcript = pipeline.finalize_raw(
            raw,
            transcript_id=args.transcript_id,
            recorded_at=_parse_datetime(args.recorded_at),
            source_path=str(args.audio_path),
            source_metadata=_load_source_metadata(args.source_metadata),
            expected_speaker_count=args.expected_speaker_count,
            expected_speaker_source=args.expected_speaker_source,
            audio_format=_audio_format_for_cli(args.audio_path),
            audio_path=args.audio_path,
        )
        emit_transcript(transcript, args, raw=raw)
        return 0
    finally:
        store.close()


def list_transcripts_cmd(args: argparse.Namespace) -> int:
    store = _store(args)
    try:
        rows = store.list_transcripts(
            limit=args.limit,
            offset=args.offset,
            source=args.source,
            meeting_type=args.meeting_type,
            diarization_state=args.diarization_state,
        )
        print(json.dumps(rows, separators=(",", ":")) if args.json else _render_transcript_list(rows))
        return 0
    finally:
        store.close()


def load_cmd(args: argparse.Namespace) -> int:
    store = _store(args)
    try:
        transcript = store.load(args.transcript_id)
        if transcript is None:
            print(f"undertone: transcript not found: {args.transcript_id}", file=sys.stderr)
            return 1
        raw = store.load_raw(args.transcript_id) if output_format(args) == "raw-json" else None
        emit_transcript(transcript, args, raw=raw)
        return 0
    finally:
        store.close()


def reenrich_cmd(args: argparse.Namespace) -> int:
    store = _store(args)
    try:
        transcript = store.load(args.transcript_id)
        raw = store.load_raw(args.transcript_id)
        if transcript is None:
            print(f"undertone: transcript not found: {args.transcript_id}", file=sys.stderr)
            return 1
        if raw is None:
            print(f"undertone: raw transcript not found for: {args.transcript_id}", file=sys.stderr)
            return 1
        pipeline = AudioPipeline(store=store, config=config_for_args(args))
        refreshed = pipeline.finalize_raw(
            raw,
            transcript_id=args.transcript_id,
            recorded_at=transcript.metadata.recorded_at,
            source_path=transcript.metadata.source_path,
            source_url=transcript.metadata.source_url,
            video_path=transcript.metadata.video_path,
            expected_speaker_count=transcript.metadata.expected_speaker_count,
            expected_speaker_source=transcript.metadata.expected_speaker_source,
            source_metadata=transcript.metadata.source_metadata,
            diarization_state=transcript.metadata.diarization_state,
            diarization_error_code=transcript.metadata.diarization_error_code,
            diarization_error_detail=transcript.metadata.diarization_error_detail,
            audio_format=transcript.metadata.audio_format,
        )
        emit_transcript(refreshed, args, raw=raw)
        return 0
    finally:
        store.close()


def search_cmd(args: argparse.Namespace) -> int:
    store = _store(args)
    try:
        rows = [
            {
                "transcript_id": transcript_id,
                "segment_id": segment_id,
                "speaker_id": speaker_id,
                "snippet": snippet,
            }
            for transcript_id, segment_id, speaker_id, snippet in store.search(
                args.query,
                limit=args.limit,
            )
        ]
        print(json.dumps(rows, separators=(",", ":")) if args.json else _render_search(rows))
        return 0
    finally:
        store.close()


def _store(args: argparse.Namespace) -> TranscriptStore:
    return TranscriptStore(db_path(args))


def _read_json(path: Path) -> Any:
    if str(path) == "-":
        return json.loads(sys.stdin.read())
    return json.loads(path.read_text())


def _load_source_metadata(path: Path | None) -> dict:
    if path is None:
        return {}
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ValueError("--source-metadata must be a JSON object")
    return value


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    return datetime.fromisoformat(normalized)


def _add_output_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-format", choices=sorted(OUTPUT_FORMATS))
    parser.add_argument("--output-detail", choices=sorted(OUTPUT_DETAIL_LEVELS))
    parser.add_argument("--output", type=Path)


def _add_feature_toggles(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-turn-taking", action="store_true")
    parser.add_argument("--no-fillers", action="store_true")
    parser.add_argument("--no-linguistic", action="store_true")
    parser.add_argument("--no-meeting-type", action="store_true")


def _audio_format_for_cli(audio_path: Path) -> dict:
    from undertone_audio.pipeline import _audio_format

    return _audio_format(audio_path)


def _render_transcript_list(rows: list[dict]) -> str:
    if not rows:
        return "No transcripts found."
    lines = ["Transcripts"]
    for row in rows:
        duration = _duration(row["duration_ms"])
        source = row.get("source_url") or row.get("source_path") or "-"
        recorded = row.get("recorded_at") or "-"
        lines.append(
            f"  {row['transcript_id']}  {duration}  speakers={row['speaker_count']} "
            f"segments={row['segment_count']}  {row['engine']}  {row['diarization_state'] or '-'}"
        )
        lines.append(f"    recorded: {recorded}")
        lines.append(f"    source:   {source}")
    return "\n".join(lines)


def _render_search(rows: list[dict]) -> str:
    if not rows:
        return "No matching transcript segments found."
    lines = ["Search results"]
    for row in rows:
        lines.append(
            f"  {row['transcript_id']} {row['segment_id']} {row['speaker_id']}: "
            f"{row['snippet']}"
        )
    return "\n".join(lines)


def _duration(ms: int | None) -> str:
    if ms is None:
        return "-"
    total_seconds = ms // 1000
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"


__all__ = [
    "ValidationError",
    "finalize_json_cmd",
    "list_transcripts_cmd",
    "load_cmd",
    "reenrich_cmd",
    "register",
    "run_wav_cmd",
    "search_cmd",
]
