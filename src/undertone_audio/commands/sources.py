from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

from undertone_audio.audio import AudioPreprocessor
from undertone_audio.commands.common import (
    add_audio_pipeline_flags,
    add_duplicate_flags,
    config_for_args,
    emit_progress,
    emit_transcript,
    guard_existing_transcript,
    progress_warning_sink,
)
from undertone_audio.engines import create_engine
from undertone_audio.pipeline import AudioPipeline, _audio_format
from undertone_audio.sources.meet import MeetSource, list_recent_conferences, list_recordings, list_transcripts
from undertone_audio.sources.quill import DEFAULT_MEETINGS_DIR, DEFAULT_QUILL_DB, QuillSource
from undertone_audio.storage import TranscriptStore


def register(subcommands: argparse._SubParsersAction) -> None:
    quill_list = subcommands.add_parser("quill-list", help="List local Quill meetings and audio availability.")
    quill_list.add_argument("--quill-db", type=Path, default=DEFAULT_QUILL_DB)
    quill_list.add_argument("--meetings-dir", type=Path, default=DEFAULT_MEETINGS_DIR)
    quill_list.add_argument("--limit", type=int, default=20)
    quill_list.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    quill_list.set_defaults(func=quill_list_cmd)

    quill_ingest = subcommands.add_parser(
        "quill-ingest",
        help="Ingest Quill recordings by rerunning undertone FluidAudio diarization.",
    )
    quill_ingest.add_argument("meeting_id", nargs="?")
    quill_ingest.add_argument("--quill-db", type=Path, default=DEFAULT_QUILL_DB)
    quill_ingest.add_argument("--meetings-dir", type=Path, default=DEFAULT_MEETINGS_DIR)
    quill_ingest.add_argument("--limit", type=int)
    quill_ingest.add_argument("--dry-run", action="store_true")
    quill_ingest.add_argument("--json", action="store_true", help="Print machine-readable JSON reports.")
    add_audio_pipeline_flags(quill_ingest)
    add_duplicate_flags(quill_ingest)
    quill_ingest.set_defaults(func=quill_ingest_cmd)

    meet_ingest = subcommands.add_parser(
        "meet-ingest",
        help="Ingest Google Meet with local/Drive recording preferred and API text fallback.",
    )
    meet_ingest.add_argument("conference_record")
    meet_ingest.add_argument("--audio", type=Path, help="Explicit local recording to supersede Meet exports.")
    meet_ingest.add_argument("--download-dir", type=Path)
    meet_ingest.add_argument("--google-account")
    meet_ingest.add_argument("--adc-file", type=Path)
    meet_ingest.add_argument("--no-text-fallback", action="store_true")
    meet_ingest.add_argument("--transcript-id")
    add_audio_pipeline_flags(meet_ingest)
    add_duplicate_flags(meet_ingest)
    meet_ingest.set_defaults(func=meet_ingest_cmd)

    meet_discover = subcommands.add_parser(
        "meet-discover",
        help="List recent Google Meet conference records and audio/text availability.",
    )
    meet_discover.add_argument("--limit", type=int, default=25)
    meet_discover.add_argument("--google-account")
    meet_discover.add_argument("--adc-file", type=Path)
    meet_discover.add_argument("--no-probe", action="store_true")
    meet_discover.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    meet_discover.set_defaults(func=meet_discover_cmd)


def quill_list_cmd(args: argparse.Namespace) -> int:
    source = QuillSource(db_path=args.quill_db, meetings_dir=args.meetings_dir)
    rows = [
        {
            "meeting_id": meeting.meeting_id,
            "title": meeting.title,
            "word_count": meeting.word_count,
            "combined": str(meeting.combined) if meeting.combined else None,
            "mic": str(meeting.mic) if meeting.mic else None,
            "system": str(meeting.system) if meeting.system else None,
            "ingestable": meeting.has_audio,
        }
        for meeting in source.list_meetings(limit=args.limit)
    ]
    if args.json:
        print(json.dumps(rows, separators=(",", ":")))
    elif not args.quill_db.exists():
        print(
            f"Quill DB not found at {args.quill_db}.\n"
            "Install/sign in to Quill, or pass --quill-db and --meetings-dir.\n"
            "Check readiness with `undertone doctor`."
        )
    else:
        print(_render_quill_list(rows))
    return 0


def quill_ingest_cmd(args: argparse.Namespace) -> int:
    config = config_for_args(args)
    if args.meeting_id and not args.dry_run:
        store = TranscriptStore(config.db_path)
        try:
            if guard_existing_transcript(store, args.meeting_id, args):
                emit_progress(args, "skipped", transcript_id=args.meeting_id, reason="exists")
                return 0
        finally:
            store.close()
    source = QuillSource(
        db_path=args.quill_db,
        meetings_dir=args.meetings_dir,
        preprocessor=AudioPreprocessor(process_timeout_seconds=config.process_timeout_seconds),
    )
    meetings = [source.meeting(args.meeting_id)] if args.meeting_id else source.list_meetings(limit=args.limit or 50)
    report = {"seen": len(meetings), "ingested": 0, "skipped": []}
    if args.dry_run:
        report["candidates"] = [
            {
                "meeting_id": meeting.meeting_id,
                "title": meeting.title,
                "has_audio": meeting.has_audio,
                "combined": str(meeting.combined) if meeting.combined else None,
                "mic": str(meeting.mic) if meeting.mic else None,
                "system": str(meeting.system) if meeting.system else None,
            }
            for meeting in meetings
        ]
        print(json.dumps(report, separators=(",", ":")) if args.json else _render_quill_report(report))
        return 0
    store = TranscriptStore(config.db_path)
    try:
        pending = []
        for meeting in meetings:
            if guard_existing_transcript(store, meeting.meeting_id, args, quiet=True):
                emit_progress(args, "skipped", transcript_id=meeting.meeting_id, reason="exists")
                report["skipped"].append({"meeting_id": meeting.meeting_id, "reason": "exists"})
                continue
            try:
                audio_path, hydrated = source.local_audio_for_meeting(meeting.meeting_id)
            except Exception as exc:
                report["skipped"].append({"meeting_id": meeting.meeting_id, "error": str(exc)})
                continue
            pending.append((meeting, audio_path, hydrated))
        if not pending:
            if args.meeting_id and report["skipped"]:
                if getattr(args, "progress", "off") == "json":
                    first = report["skipped"][0]
                    raise RuntimeError(first.get("error") or first.get("reason") or "quill ingest failed")
                print(
                    json.dumps(report, separators=(",", ":")) if args.json else _render_quill_report(report),
                    file=sys.stderr,
                )
                return 1
            print(json.dumps(report, separators=(",", ":")) if args.json else _render_quill_report(report))
            return 0
        engine = create_engine(args.engine, config)
        pipeline = AudioPipeline(
            store=store,
            engine=engine,
            config=config,
            warning_sink=progress_warning_sink(args),
        )
        real_failures = 0
        for meeting, audio_path, hydrated in pending:
            try:
                emit_progress(
                    args,
                    "start",
                    command="quill-ingest",
                    transcript_id=meeting.meeting_id,
                    audio_path=str(audio_path),
                )
                raw = asyncio.run(engine.transcribe(audio_path))
                emit_progress(
                    args,
                    "transcribed",
                    transcript_id=meeting.meeting_id,
                    duration_ms=raw.duration_ms,
                    segments=len(raw.segments),
                    speakers=len(raw.speakers),
                )
                emit_progress(args, "finalizing", transcript_id=meeting.meeting_id)
                transcript = pipeline.finalize_raw(
                    raw,
                    transcript_id=meeting.meeting_id,
                    recorded_at=source.recorded_at(hydrated),
                    source_path=str(hydrated.combined or hydrated.mic or audio_path),
                    source_metadata=source.source_metadata(hydrated),
                    expected_speaker_count=args.expected_speaker_count,
                    expected_speaker_source=args.expected_speaker_source,
                    audio_format=_audio_format(audio_path),
                    audio_path=audio_path,
                )
                emit_progress(args, "saved", transcript_id=transcript.transcript_id)
            except Exception as exc:
                report["skipped"].append({"meeting_id": meeting.meeting_id, "error": str(exc)})
                real_failures += 1
                continue
            report["ingested"] += 1
            if args.meeting_id:
                emit_transcript(transcript, args, raw=raw)
        if not args.meeting_id or real_failures:
            print(
                json.dumps(report, separators=(",", ":")) if args.json else _render_quill_report(report),
                file=sys.stderr if args.meeting_id else sys.stdout,
            )
        return 1 if real_failures else 0
    finally:
        store.close()


def meet_ingest_cmd(args: argparse.Namespace) -> int:
    config = config_for_args(args)
    transcript_id = args.transcript_id or "meet-" + _stable_id(args.conference_record)
    store = TranscriptStore(config.db_path)
    try:
        if guard_existing_transcript(store, transcript_id, args):
            emit_progress(args, "skipped", transcript_id=transcript_id, reason="exists")
            return 0
    finally:
        store.close()
    source = MeetSource(
        download_dir=args.download_dir,
        preprocessor=AudioPreprocessor(process_timeout_seconds=config.process_timeout_seconds),
        google_account=args.google_account,
        adc_file=args.adc_file,
    )
    selection = source.select(
        args.conference_record,
        local_audio=args.audio,
        allow_text_fallback=not args.no_text_fallback,
    )
    store = TranscriptStore(config.db_path)
    try:
        pipeline = AudioPipeline(
            store=store,
            engine=create_engine(args.engine, config),
            config=config,
            warning_sink=progress_warning_sink(args),
        )
        if selection.audio_path is not None:
            emit_progress(
                args,
                "start",
                command="meet-ingest",
                transcript_id=transcript_id,
                audio_path=str(selection.audio_path),
            )
            raw = asyncio.run(pipeline.engine.transcribe(selection.audio_path))
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
                source_path=str(selection.audio_path),
                source_metadata=selection.source_metadata,
                expected_speaker_count=args.expected_speaker_count,
                expected_speaker_source=args.expected_speaker_source,
                audio_format=_audio_format(selection.audio_path),
                audio_path=selection.audio_path,
            )
            emit_progress(args, "saved", transcript_id=transcript.transcript_id)
            emit_transcript(transcript, args, raw=raw)
            return 0
        if selection.raw_fallback is None:
            raise ValueError("Meet source returned neither audio nor text fallback")
        fallback_config = replace(config, min_talk_seconds=0)
        pipeline = AudioPipeline(
            store=store,
            engine=pipeline.engine,
            config=fallback_config,
            warning_sink=progress_warning_sink(args),
        )
        emit_progress(args, "finalizing", transcript_id=transcript_id, fallback="text")
        transcript = pipeline.finalize_raw(
            selection.raw_fallback,
            transcript_id=transcript_id,
            source_path=args.conference_record,
            source_metadata=selection.source_metadata,
            expected_speaker_count=args.expected_speaker_count,
            expected_speaker_source=args.expected_speaker_source,
            diarization_state="text-fallback",
            diarization_error_code="no-audio",
            diarization_error_detail="No local or downloadable Meet recording was available.",
            apply_speaker_processing=True,
        )
        emit_progress(args, "saved", transcript_id=transcript.transcript_id)
        emit_transcript(transcript, args, raw=selection.raw_fallback)
        return 0
    finally:
        store.close()


def meet_discover_cmd(args: argparse.Namespace) -> int:
    rows = []
    for conference in list_recent_conferences(
        limit=args.limit,
        google_account=args.google_account,
        adc_file=args.adc_file,
    ):
        name = conference.get("name")
        entry = {
            "conference_record": name,
            "start_time": conference.get("startTime"),
            "end_time": conference.get("endTime"),
            "space": conference.get("space"),
            "has_recording": None,
            "has_transcript": None,
            "google_account": args.google_account,
        }
        if name and not args.no_probe:
            try:
                entry["has_recording"] = any(
                    item.get("state") in {"ENDED", "FILE_GENERATED"}
                    and item.get("driveDestination", {}).get("file")
                    for item in list_recordings(name, google_account=args.google_account, adc_file=args.adc_file)
                )
            except Exception:
                entry["has_recording"] = False
            try:
                entry["has_transcript"] = bool(
                    list_transcripts(name, google_account=args.google_account, adc_file=args.adc_file)
                )
            except Exception:
                entry["has_transcript"] = False
        rows.append(entry)
    print(json.dumps(rows, separators=(",", ":")) if args.json else _render_meet_discover(rows))
    return 0


def _stable_id(value: str) -> str:
    return hashlib.sha1(value.encode()).hexdigest()[:16]


def _render_quill_list(rows: list[dict]) -> str:
    if not rows:
        return "No Quill meetings found."
    lines = ["Quill meetings"]
    for row in rows:
        status = "audio" if row["ingestable"] else "text only"
        title = row["title"] or "(untitled)"
        lines.append(f"  {row['meeting_id']}  {status}  words={row['word_count']}  {title}")
        for key in ("combined", "mic", "system"):
            if row.get(key):
                lines.append(f"    {key}: {row[key]}")
    return "\n".join(lines)


def _render_quill_report(report: dict) -> str:
    lines = [
        "Quill ingest report",
        f"  seen:     {report['seen']}",
        f"  ingested: {report['ingested']}",
    ]
    for candidate in report.get("candidates", []):
        status = "audio" if candidate["has_audio"] else "text only"
        lines.append(f"  candidate {candidate['meeting_id']}: {status}  {candidate['title'] or ''}")
    for skipped in report.get("skipped", []):
        meeting_id = skipped.get("meeting_id", "-")
        reason = skipped.get("reason") or skipped.get("error") or "unknown"
        lines.append(f"  skipped {meeting_id}: {reason}")
    return "\n".join(lines)


def _render_meet_discover(rows: list[dict]) -> str:
    if not rows:
        return "No Google Meet conference records found."
    lines = ["Google Meet conference records"]
    for row in rows:
        recording = _availability(row["has_recording"])
        transcript = _availability(row["has_transcript"])
        lines.append(f"  {row['conference_record']}")
        lines.append(f"    start:      {row['start_time'] or '-'}")
        lines.append(f"    recording:  {recording}")
        lines.append(f"    transcript: {transcript}")
        if row.get("google_account"):
            lines.append(f"    account:    {row['google_account']}")
    return "\n".join(lines)


def _availability(value: bool | None) -> str:
    if value is None:
        return "not checked"
    return "yes" if value else "no"
