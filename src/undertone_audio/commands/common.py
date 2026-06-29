from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from undertone_audio.config import Config, load as load_config
from undertone_audio.dedupe import DuplicateTranscriptError, audio_signature_for_path
from undertone_audio.engines.base import RawTranscript
from undertone_audio.export import (
    OUTPUT_DETAIL_LEVELS,
    OUTPUT_FORMATS,
    render_transcript,
    write_or_print,
)


def add_audio_pipeline_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--engine",
        choices=["fluidaudio-hybrid", "fluidaudio-pyannote", "fluidaudio-cli"],
    )
    parser.add_argument("--fluidaudio-cli")
    parser.add_argument("--expected-speaker-count", type=int)
    parser.add_argument("--expected-speaker-source")
    parser.add_argument("--clustering-threshold", type=float)
    parser.add_argument("--speaker-merge-threshold", type=float)
    parser.add_argument("--min-talk-seconds", type=float)
    parser.add_argument("--fingerprint-similarity-threshold", type=float)
    parser.add_argument("--turn-gap-ms", type=int)
    parser.add_argument("--asr-model")
    parser.add_argument("--diarization-model")
    parser.add_argument("--vad-model")
    parser.add_argument("--embedding-model")
    parser.add_argument("--pyannote-model")
    parser.add_argument("--pyannote-device")
    parser.add_argument("--fingerprint-backend")
    parser.add_argument(
        "--process-timeout-seconds",
        type=float,
        help="Bound external FluidAudio/ffmpeg/yt-dlp subprocesses; 0 disables the timeout.",
    )
    parser.add_argument("--voice-metrics", choices=["off", "optional", "required"])
    parser.add_argument("--output-format", choices=sorted(OUTPUT_FORMATS))
    parser.add_argument("--output-detail", choices=sorted(OUTPUT_DETAIL_LEVELS))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--progress",
        choices=["off", "json"],
        default="off",
        help="Emit long-running job progress events. JSON events are written to stderr.",
    )


def add_duplicate_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--force", action="store_true", help="Overwrite an existing transcript id.")
    group.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip ingest when the target transcript id already exists.",
    )
    parser.add_argument(
        "--allow-duplicate",
        action="store_true",
        help="Allow content-duplicate ingest. By default duplicate content is skipped before fingerprints update.",
    )


def db_path(args: argparse.Namespace) -> Path:
    return args.db or load_config().db_path


def config_for_args(args: argparse.Namespace) -> Config:
    cfg = load_config()
    return Config(
        db_path=db_path(args),
        fluidaudio_cli=getattr(args, "fluidaudio_cli", None) or cfg.fluidaudio_cli,
        default_engine=getattr(args, "engine", None) or cfg.default_engine,
        asr_model=getattr(args, "asr_model", None) or cfg.asr_model,
        diarization_model=getattr(args, "diarization_model", None) or cfg.diarization_model,
        vad_model=getattr(args, "vad_model", None) or cfg.vad_model,
        embedding_model=getattr(args, "embedding_model", None) or cfg.embedding_model,
        pyannote_model=getattr(args, "pyannote_model", None) or cfg.pyannote_model,
        pyannote_device=getattr(args, "pyannote_device", None) or cfg.pyannote_device,
        fingerprint_backend=getattr(args, "fingerprint_backend", None) or cfg.fingerprint_backend,
        clustering_threshold=_override(args, "clustering_threshold", cfg.clustering_threshold),
        speaker_merge_threshold=_override(
            args,
            "speaker_merge_threshold",
            cfg.speaker_merge_threshold,
        ),
        min_talk_seconds=_override(args, "min_talk_seconds", cfg.min_talk_seconds),
        fingerprint_similarity_threshold=_override(
            args,
            "fingerprint_similarity_threshold",
            cfg.fingerprint_similarity_threshold,
        ),
        turn_gap_ms=_override(args, "turn_gap_ms", cfg.turn_gap_ms),
        enable_turn_taking=_feature_enabled(args, "no_turn_taking", cfg.enable_turn_taking),
        enable_fillers=_feature_enabled(args, "no_fillers", cfg.enable_fillers),
        enable_linguistic=_feature_enabled(args, "no_linguistic", cfg.enable_linguistic),
        enable_meeting_type=_feature_enabled(args, "no_meeting_type", cfg.enable_meeting_type),
        voice_metrics=getattr(args, "voice_metrics", None) or cfg.voice_metrics,
        default_output_format=getattr(args, "output_format", None) or cfg.default_output_format,
        default_output_detail=getattr(args, "output_detail", None) or cfg.default_output_detail,
        webhook_url=cfg.webhook_url,
        webhook_secret=cfg.webhook_secret,
        webhook_enabled=cfg.webhook_enabled,
        webhook_accept_degraded=cfg.webhook_accept_degraded,
        process_timeout_seconds=_override(
            args,
            "process_timeout_seconds",
            cfg.process_timeout_seconds,
        ),
    )


def emit_transcript(
    transcript,
    args: argparse.Namespace,
    *,
    raw: RawTranscript | None = None,
) -> None:
    config = config_for_args(args)
    fmt = getattr(args, "output_format", None) or config.default_output_format
    detail = getattr(args, "output_detail", None) or config.default_output_detail
    output = getattr(args, "output", None)
    body = render_transcript(transcript, fmt, raw=raw, detail=detail)
    write_or_print(body, output)


def guard_existing_transcript(
    store,
    transcript_id: str | None,
    args: argparse.Namespace,
    *,
    quiet: bool = False,
) -> bool:
    if not transcript_id or not store.exists(transcript_id):
        return False
    if getattr(args, "skip_existing", False):
        if not quiet:
            payload = json.dumps(
                {"transcript_id": transcript_id, "skipped": True, "reason": "exists"},
                separators=(",", ":"),
            )
            if getattr(args, "progress", "off") != "json":
                print(payload, file=sys.stderr if getattr(args, "output", None) else sys.stdout)
        return True
    if getattr(args, "force", False):
        return False
    raise ValueError(f"transcript already exists: {transcript_id}; pass --force or --skip-existing")


def audio_content_signature(
    store,
    args: argparse.Namespace,
    config: Config,
    audio_path: Path,
    *,
    transcript_id: str | None,
):
    signature = audio_signature_for_path(
        Path(audio_path),
        timeout_seconds=config.process_timeout_seconds,
    )
    if signature is None:
        return None
    if not getattr(args, "allow_duplicate", False):
        duplicate = store.find_audio_duplicate(
            signature.value,
            signature.algorithm,
            exclude_transcript_id=transcript_id,
        )
        if duplicate is not None:
            raise DuplicateTranscriptError(transcript_id or "", duplicate)
    return signature


def emit_duplicate_skip(args: argparse.Namespace, exc: DuplicateTranscriptError) -> int:
    payload = exc.payload()
    emit_progress(
        args,
        "skipped",
        transcript_id=payload["transcript_id"],
        reason="duplicate",
        existing_transcript_id=payload["existing_transcript_id"],
        match_type=payload["match_type"],
        algorithm=payload["algorithm"],
        distance=payload["distance"],
    )
    body = json.dumps(payload, separators=(",", ":"))
    if getattr(args, "progress", "off") != "json":
        print(body, file=sys.stderr if getattr(args, "output", None) else sys.stdout)
    return 0


def emit_progress(args: argparse.Namespace, event: str, **fields) -> None:
    if getattr(args, "progress", "off") != "json":
        return
    payload = {"event": event, **fields}
    print(json.dumps(payload, separators=(",", ":"), default=str), file=sys.stderr, flush=True)


def progress_warning_sink(args: argparse.Namespace):
    def _sink(name: str, payload: dict) -> None:
        emit_progress(args, "warning", warning=name, **payload)

    return _sink


def output_format(args: argparse.Namespace) -> str:
    config = config_for_args(args)
    return getattr(args, "output_format", None) or config.default_output_format


def _override(args: argparse.Namespace, name: str, default):
    value = getattr(args, name, None)
    return default if value is None else value


def _feature_enabled(args: argparse.Namespace, disable_flag: str, default: bool) -> bool:
    return False if getattr(args, disable_flag, False) else default
