from __future__ import annotations

import argparse
import asyncio
import json
import shutil

from undertone_audio.commands.common import config_for_args, db_path
from undertone_audio.engines import create_engine
from undertone_audio.source_readiness import source_statuses
from undertone_audio.sources.meet import meet_auth_check
from undertone_audio.storage import TranscriptStore


def register(subcommands: argparse._SubParsersAction) -> None:
    models = subcommands.add_parser("models", help="Print effective undertone backend selections.")
    models.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    models.set_defaults(func=models_cmd)

    delete = subcommands.add_parser("delete", help="Delete a saved transcript.")
    delete.add_argument("transcript_id")
    delete.add_argument("--yes", action="store_true", help="Confirm deletion.")
    delete.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    delete.set_defaults(func=delete_cmd)

    stats = subcommands.add_parser("stats", help="Print database transcript/fingerprint totals.")
    stats.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    stats.set_defaults(func=stats_cmd)

    sources = subcommands.add_parser("sources", help="Show optional source readiness.")
    sources.add_argument("--check-meet", action="store_true", help="Refresh/test Google Meet ADC.")
    sources.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    sources.set_defaults(func=sources_cmd)

    doctor = subcommands.add_parser("doctor", help="Run local undertone preflight checks.")
    doctor.add_argument("--check-yt-dlp", action="store_true")
    doctor.add_argument("--check-meet", action="store_true")
    doctor.add_argument("--all", action="store_true", help="Check all optional source integrations.")
    doctor.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    doctor.set_defaults(func=doctor_cmd)


def models_cmd(args: argparse.Namespace) -> int:
    config = config_for_args(args)
    payload = {
        "engine": config.default_engine,
        "asr_model": config.asr_model,
        "diarization_model": config.diarization_model,
        "vad_model": config.vad_model,
        "embedding_model": config.embedding_model,
        "fingerprint_backend": config.fingerprint_backend,
        "voice_metrics": config.voice_metrics,
        "output_format": config.default_output_format,
        "output_detail": config.default_output_detail,
        "features": {
            "turn_taking": config.enable_turn_taking,
            "fillers": config.enable_fillers,
            "linguistic": config.enable_linguistic,
            "meeting_type": config.enable_meeting_type,
        },
        "thresholds": {
            "clustering": config.clustering_threshold,
            "speaker_merge": config.speaker_merge_threshold,
            "min_talk_seconds": config.min_talk_seconds,
            "fingerprint_similarity": config.fingerprint_similarity_threshold,
            "turn_gap_ms": config.turn_gap_ms,
        },
    }
    if args.json:
        print(json.dumps(payload, separators=(",", ":")))
    else:
        print(_render_models(payload))
    return 0


def delete_cmd(args: argparse.Namespace) -> int:
    if not args.yes:
        raise ValueError("delete requires --yes")
    store = TranscriptStore(db_path(args))
    try:
        deleted = store.delete(args.transcript_id)
        payload = {"transcript_id": args.transcript_id, "deleted": deleted}
        if args.json:
            print(json.dumps(payload, separators=(",", ":")))
        else:
            status = "deleted" if deleted else "not found"
            print(f"{args.transcript_id}: {status}")
        return 0 if deleted else 1
    finally:
        store.close()


def stats_cmd(args: argparse.Namespace) -> int:
    store = TranscriptStore(db_path(args))
    try:
        payload = store.stats()
        print(json.dumps(payload, separators=(",", ":")) if args.json else _render_stats(payload))
        return 0
    finally:
        store.close()


def sources_cmd(args: argparse.Namespace) -> int:
    rows = source_statuses(check_meet=args.check_meet)
    payload = {"sources": rows}
    print(json.dumps(payload, separators=(",", ":")) if args.json else _render_sources(rows))
    return 0


def doctor_cmd(args: argparse.Namespace) -> int:
    config = config_for_args(args)
    checks = []
    sources = source_statuses(check_meet=args.check_meet or args.all)
    ok = True
    store = None
    try:
        store = TranscriptStore(config.db_path)
        writable = True
    except Exception:
        writable = False
        ok = False
    finally:
        try:
            store.close()
        except Exception:
            pass
    checks.append({"name": "db_writable", "ok": writable, "path": str(config.db_path)})
    try:
        engine = create_engine(config.default_engine, config)
        engine_ok = asyncio.run(engine.healthcheck())
        checks.append(
            {
                "name": "engine",
                "ok": bool(engine_ok),
                "engine": config.default_engine,
                "fluidaudio_cli": getattr(engine, "cli_path", config.fluidaudio_cli),
            }
        )
        ok = ok and bool(engine_ok)
    except Exception as exc:
        checks.append(
            {
                "name": "engine",
                "ok": False,
                "engine": config.default_engine,
                "error": str(exc),
            }
        )
        ok = False
    if args.check_yt_dlp or args.all:
        path = shutil.which("yt-dlp")
        checks.append(
            {
                "name": "yt_dlp",
                "ok": path is not None,
                "path": path,
                "fix": None
                if path
                else "Install connectors with `pip install -e '.[connectors]'` or pass --yt-dlp-bin.",
            }
        )
        ok = ok and path is not None
    if args.check_meet or args.all:
        meet = meet_auth_check()
        checks.append(
            {
                "name": "meet_adc",
                "ok": meet["ok"],
                "project": meet.get("project"),
                "error": meet.get("error"),
                "fix": meet.get("fix"),
            }
        )
        ok = ok and bool(meet["ok"])
    payload = {"ok": ok, "checks": checks, "sources": sources}
    print(json.dumps(payload, separators=(",", ":")) if args.json else _render_doctor(payload))
    return 0 if ok else 1


def _render_models(payload: dict) -> str:
    features = ", ".join(
        name.replace("_", "-")
        for name, enabled in payload["features"].items()
        if enabled
    ) or "none"
    thresholds = payload["thresholds"]
    lines = [
        "undertone models",
        f"  engine:              {payload['engine']}",
        f"  ASR:                 {payload['asr_model']}",
        f"  diarization:         {payload['diarization_model']}",
        f"  VAD:                 {payload['vad_model']}",
        f"  embeddings:          {payload['embedding_model']}",
        f"  fingerprints:        {payload['fingerprint_backend']}",
        f"  voice metrics:       {payload['voice_metrics']}",
        f"  default output:      {payload['output_format']} ({payload['output_detail']})",
        f"  features:            {features}",
        "  thresholds:",
        f"    clustering:        {thresholds['clustering']}",
        f"    speaker merge:     {thresholds['speaker_merge']}",
        f"    min talk seconds:  {thresholds['min_talk_seconds']}",
        f"    fingerprint match: {thresholds['fingerprint_similarity']}",
        f"    turn gap ms:       {thresholds['turn_gap_ms']}",
    ]
    return "\n".join(lines)


def _render_stats(payload: dict) -> str:
    return "\n".join(
        [
            "undertone database",
            f"  path:          {payload['db_path']}",
            f"  transcripts:   {payload['transcript_count']}",
            f"  duration:      {payload['total_duration_minutes']} min",
            f"  speakers:      {payload['speaker_count']}",
            f"  segments:      {payload['segment_count']}",
            f"  fingerprints:  {payload['fingerprint_count']}",
        ]
    )


def _render_doctor(payload: dict) -> str:
    lines = ["undertone doctor", f"  core: {'ok' if payload['ok'] else 'failed'}", "", "Core checks"]
    for check in payload["checks"]:
        marker = "ok" if check["ok"] else "failed"
        detail = _check_detail(check)
        lines.append(f"  [{marker}] {check['name']}{detail}")
    lines.extend(["", "Sources"])
    for source in payload.get("sources", []):
        lines.append(f"  {source['source']:<8} {source['state']:<18} {source['detail']}")
        if source.get("fix"):
            lines.append(f"           fix: {source['fix']}")
    return "\n".join(lines)


def _render_sources(rows: list[dict]) -> str:
    lines = ["Sources"]
    for row in rows:
        lines.append(f"  {row['source']:<8} {row['state']:<18} {row['detail']}")
        if row.get("fix"):
            lines.append(f"           fix: {row['fix']}")
    return "\n".join(lines)


def _check_detail(check: dict) -> str:
    parts = []
    for key in ("path", "engine", "fluidaudio_cli", "project", "state", "detail", "error", "fix"):
        value = check.get(key)
        if value:
            parts.append(f"{key}={value}")
    return f" ({', '.join(parts)})" if parts else ""
