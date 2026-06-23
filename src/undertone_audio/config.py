from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_path(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name) or default).expanduser()


@dataclass(frozen=True)
class Config:
    db_path: Path
    fluidaudio_cli: str | None = None
    default_engine: str = "fluidaudio-hybrid"
    asr_model: str = "FluidAudio Parakeet TDT"
    diarization_model: str = "FluidAudio Sortformer + process"
    vad_model: str = "FluidAudio/Silero VAD"
    embedding_model: str = "FluidAudio pyannote-derived speaker embeddings"
    fingerprint_backend: str = "undertone-speaker-fingerprints"
    clustering_threshold: float = 0.7045655
    speaker_merge_threshold: float = 0.82
    min_talk_seconds: float = 1.5
    fingerprint_similarity_threshold: float = 0.78
    turn_gap_ms: int = 800
    enable_turn_taking: bool = True
    enable_fillers: bool = True
    enable_linguistic: bool = True
    enable_meeting_type: bool = True
    voice_metrics: str = "optional"
    default_output_format: str = "json"
    default_output_detail: str = "full"
    webhook_url: str | None = None
    webhook_secret: str | None = None
    webhook_enabled: bool = True
    webhook_accept_degraded: bool = False


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


def load() -> Config:
    return Config(
        db_path=_env_path("UNDERTONE_DB_PATH", "./undertone.db"),
        fluidaudio_cli=os.environ.get("UNDERTONE_FLUIDAUDIO_CLI")
        or os.environ.get("FLUIDAUDIO_CLI"),
        default_engine=os.environ.get("UNDERTONE_ENGINE", "fluidaudio-hybrid"),
        asr_model=os.environ.get("UNDERTONE_ASR_MODEL", "FluidAudio Parakeet TDT"),
        diarization_model=os.environ.get(
            "UNDERTONE_DIARIZATION_MODEL",
            "FluidAudio Sortformer + process",
        ),
        vad_model=os.environ.get("UNDERTONE_VAD_MODEL", "FluidAudio/Silero VAD"),
        embedding_model=os.environ.get(
            "UNDERTONE_EMBEDDING_MODEL",
            "FluidAudio pyannote-derived speaker embeddings",
        ),
        fingerprint_backend=os.environ.get(
            "UNDERTONE_FINGERPRINT_BACKEND",
            "undertone-speaker-fingerprints",
        ),
        clustering_threshold=_env_float("UNDERTONE_CLUSTERING_THRESHOLD", 0.7045655),
        speaker_merge_threshold=_env_float("UNDERTONE_SPEAKER_MERGE_THRESHOLD", 0.82),
        min_talk_seconds=_env_float("UNDERTONE_MIN_TALK_SECONDS", 1.5),
        fingerprint_similarity_threshold=_env_float(
            "UNDERTONE_FINGERPRINT_SIMILARITY_THRESHOLD",
            0.78,
        ),
        turn_gap_ms=_env_int("UNDERTONE_TURN_GAP_MS", 800),
        enable_turn_taking=_env_bool("UNDERTONE_ENABLE_TURN_TAKING", True),
        enable_fillers=_env_bool("UNDERTONE_ENABLE_FILLERS", True),
        enable_linguistic=_env_bool("UNDERTONE_ENABLE_LINGUISTIC", True),
        enable_meeting_type=_env_bool("UNDERTONE_ENABLE_MEETING_TYPE", True),
        voice_metrics=os.environ.get("UNDERTONE_VOICE_METRICS", "optional"),
        default_output_format=os.environ.get("UNDERTONE_OUTPUT_FORMAT", "json"),
        default_output_detail=os.environ.get("UNDERTONE_OUTPUT_DETAIL", "full"),
        webhook_url=os.environ.get("UNDERTONE_WEBHOOK_URL"),
        webhook_secret=os.environ.get("UNDERTONE_WEBHOOK_SECRET"),
        webhook_enabled=_env_bool("UNDERTONE_WEBHOOK_ENABLED", True),
        webhook_accept_degraded=_env_bool("UNDERTONE_WEBHOOK_ACCEPT_DEGRADED", False),
    )
