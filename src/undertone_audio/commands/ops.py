from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from undertone_audio.commands.common import config_for_args, db_path
from undertone_audio.engines import create_engine
from undertone_audio.engines.fluidaudio_pyannote import pyannote_status
from undertone_audio.pipeline import effective_fingerprint_embedding_model
from undertone_audio.processes import run_process_sync
from undertone_audio.schema import ConnectorAssetSchema, ConnectorCandidateSchema, EnrichedTranscript
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

    schema = subcommands.add_parser("schema", help="Print published JSON Schemas.")
    schema.add_argument(
        "name",
        choices=["transcript", "connector-asset", "connector-candidate"],
        nargs="?",
        default="transcript",
    )
    schema.add_argument("--output", type=Path, help="Write schema JSON to this path.")
    schema.set_defaults(func=schema_cmd)

    doctor = subcommands.add_parser("doctor", help="Run local undertone preflight checks.")
    doctor.add_argument("--check-yt-dlp", action="store_true")
    doctor.add_argument("--yt-dlp-bin", default="yt-dlp", help="yt-dlp binary name/path for --check-yt-dlp.")
    doctor.add_argument("--check-meet", action="store_true")
    doctor.add_argument("--check-pyannote", action="store_true")
    doctor.add_argument("--all", action="store_true", help="Check all optional integrations.")
    doctor.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    doctor.set_defaults(func=doctor_cmd)

    install_skills = subcommands.add_parser(
        "install-skills",
        help="Copy the bundled undertone skill into Claude or Codex skills directories.",
    )
    install_skills.add_argument(
        "--target",
        action="append",
        choices=["claude-user", "claude-project", "codex"],
        help="Install target (repeatable). Default: claude-user.",
    )
    install_skills.add_argument(
        "--force", action="store_true", help="Overwrite an existing installed skill."
    )
    install_skills.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    install_skills.set_defaults(func=install_skills_cmd)


def models_cmd(args: argparse.Namespace) -> int:
    config = config_for_args(args)
    store = TranscriptStore(config.db_path)
    try:
        fingerprint_models = store.fingerprint_model_counts(
            effective_fingerprint_embedding_model(config)
        )
        fingerprint_status = store.fingerprint_status_counts()
    finally:
        store.close()
    payload = {
        "engine": config.default_engine,
        "asr_model": config.asr_model,
        "diarization_model": config.diarization_model,
        "vad_model": config.vad_model,
        "embedding_model": config.embedding_model,
        "pyannote_model": config.pyannote_model,
        "pyannote_device": config.pyannote_device,
        "fingerprint_backend": config.fingerprint_backend,
        "fingerprint_models": fingerprint_models,
        "fingerprint_status": fingerprint_status,
        "voice_metrics": config.voice_metrics,
        "output_format": config.default_output_format,
        "output_detail": config.default_output_detail,
        "process_timeout_seconds": config.process_timeout_seconds,
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
    config = config_for_args(args)
    store = TranscriptStore(config.db_path)
    try:
        payload = store.stats(
            active_embedding_model=effective_fingerprint_embedding_model(config)
        )
        print(json.dumps(payload, separators=(",", ":")) if args.json else _render_stats(payload))
        return 0
    finally:
        store.close()


def sources_cmd(args: argparse.Namespace) -> int:
    rows = source_statuses(check_meet=args.check_meet)
    payload = {"sources": rows}
    print(json.dumps(payload, separators=(",", ":")) if args.json else _render_sources(rows))
    return 0


def schema_cmd(args: argparse.Namespace) -> int:
    models = {
        "transcript": EnrichedTranscript,
        "connector-asset": ConnectorAssetSchema,
        "connector-candidate": ConnectorCandidateSchema,
    }
    model = models[args.name]
    payload = model.model_json_schema()
    body = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body + "\n")
    else:
        print(body)
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
    if writable:
        store = TranscriptStore(config.db_path)
        try:
            fingerprint_models = store.fingerprint_model_counts(
                effective_fingerprint_embedding_model(config)
            )
            fingerprint_status = store.fingerprint_status_counts()
        finally:
            store.close()
        checks.append(
            {
                "name": "fingerprint_models",
                "ok": fingerprint_models["legacy"] == 0
                and fingerprint_models["incompatible"] == 0,
                **fingerprint_models,
                "fix": "Run `undertone fingerprint-adopt-model --dry-run` then `--yes` for legacy rows."
                if fingerprint_models["legacy"] or fingerprint_models["incompatible"]
                else None,
            }
        )
        checks.append(
            {
                "name": "fingerprint_status",
                "ok": True,
                **fingerprint_status,
            }
        )
        ok = ok and fingerprint_models["legacy"] == 0 and fingerprint_models["incompatible"] == 0
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
        path = shutil.which(args.yt_dlp_bin)
        yt_dlp_detail = _yt_dlp_version_detail(path) if path else {}
        checks.append(
            {
                "name": "yt_dlp",
                "ok": path is not None,
                "binary": args.yt_dlp_bin,
                "path": path,
                **yt_dlp_detail,
                "fix": None
                if path
                else "Install connectors with `pip install -e '.[connectors]'` or pass --yt-dlp-bin.",
            }
        )
        ok = ok and path is not None
    if args.check_pyannote or args.all:
        pyannote = pyannote_status(config.pyannote_model, config.pyannote_device)
        checks.append(
            {
                "name": "pyannote",
                "ok": pyannote["ok"],
                "model": pyannote.get("model"),
                "device": pyannote.get("device"),
                "detail": pyannote.get("detail"),
                "error": pyannote.get("error"),
                "fix": pyannote.get("fix"),
            }
        )
        ok = ok and bool(pyannote["ok"])
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
        f"  pyannote model:      {payload['pyannote_model']}",
        f"  pyannote device:     {payload['pyannote_device']}",
        f"  fingerprints:        {payload['fingerprint_backend']}",
        f"  fingerprint models:  compatible={payload['fingerprint_models']['compatible']} "
        f"legacy={payload['fingerprint_models']['legacy']} "
        f"incompatible={payload['fingerprint_models']['incompatible']}",
        f"  fingerprint status:  active={payload['fingerprint_status']['active']} "
        f"discarded={payload['fingerprint_status']['discarded']}",
        f"  voice metrics:       {payload['voice_metrics']}",
        f"  process timeout:     {payload['process_timeout_seconds']}s",
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
            f"    active:      {payload['fingerprint_status']['active']}",
            f"    discarded:   {payload['fingerprint_status']['discarded']}",
            f"    compatible:  {payload['fingerprint_models']['compatible']}",
            f"    legacy:      {payload['fingerprint_models']['legacy']}",
            f"    incompatible:{payload['fingerprint_models']['incompatible']}",
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


def _yt_dlp_version_detail(path: str | None) -> dict:
    if not path:
        return {}
    try:
        result = run_process_sync([path, "--version"], label="yt-dlp", timeout_seconds=15)
    except Exception as exc:
        return {"version": None, "stale": None, "detail": f"version check failed: {exc}"}
    version = str(result.stdout).strip()
    stale = _is_stale_yt_dlp_version(version)
    detail = "version may be stale; update yt-dlp if site extraction fails" if stale else None
    return {"version": version, "stale": stale, "detail": detail}


def _is_stale_yt_dlp_version(version: str) -> bool | None:
    if len(version) < 8 or not version[:8].isdigit():
        return None
    try:
        release = datetime.strptime(version[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return (datetime.now(timezone.utc) - release).days > 120


def _check_detail(check: dict) -> str:
    parts = []
    for key in (
        "path",
        "binary",
        "engine",
        "fluidaudio_cli",
        "model",
        "device",
        "project",
        "state",
        "active",
        "discarded",
        "total",
        "detail",
        "error",
        "fix",
    ):
        value = check.get(key)
        if value:
            parts.append(f"{key}={value}")
    return f" ({', '.join(parts)})" if parts else ""


def install_skills_cmd(args: argparse.Namespace) -> int:
    source = _bundled_skill_dir()
    targets = args.target or ["claude-user"]
    installs = []
    for target in dict.fromkeys(targets):
        base = _skill_target_dir(target)
        dest = base / "undertone"
        if dest.exists() and not args.force:
            installs.append(
                {"target": target, "path": str(dest), "status": "exists",
                 "hint": "pass --force to overwrite"}
            )
            continue
        base.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest)
        installs.append({"target": target, "path": str(dest), "status": "installed"})
    payload = {"version": _undertone_version(), "source": str(source), "installs": installs}
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2))
    else:
        print(f"undertone skill v{payload['version']} (from {source})")
        for item in installs:
            line = f"  {item['target']:16} {item['status']}: {item['path']}"
            if item.get("hint"):
                line += f" ({item['hint']})"
            print(line)
    return 0


def _bundled_skill_dir() -> Path:
    package = Path(__file__).resolve().parent.parent
    bundled = package / "_skills" / "undertone"
    if bundled.is_dir():
        return bundled
    source_tree = package.parent.parent / "skills" / "undertone"
    if source_tree.is_dir():
        return source_tree
    raise FileNotFoundError(
        "bundled undertone skill not found; reinstall undertone-audio or run from the source tree"
    )


def _skill_target_dir(target: str) -> Path:
    if target == "claude-user":
        return Path.home() / ".claude" / "skills"
    if target == "claude-project":
        return Path.cwd() / ".claude" / "skills"
    if target == "codex":
        return Path.home() / ".codex" / "skills"
    raise ValueError(f"unknown skill target: {target}")


def _undertone_version() -> str:
    try:
        from importlib.metadata import version

        return version("undertone-audio")
    except Exception:
        return "unknown"
