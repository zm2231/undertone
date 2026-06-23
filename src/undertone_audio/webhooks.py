from __future__ import annotations

import hashlib
import hmac
import json
import logging
from pathlib import Path
from urllib.parse import urlsplit

import requests

from undertone_audio.config import Config, load as load_config
from undertone_audio.schema import EnrichedTranscript

log = logging.getLogger(__name__)

EVENT_NAME = "meeting.transcript.ready"
SOURCE = "undertone"
SIGNATURE_HEADER = "x-zen-signature-256"


def readiness_payload(transcript: EnrichedTranscript, db_path: str | Path) -> dict[str, object]:
    recorded_at = transcript.metadata.recorded_at
    return {
        "event": EVENT_NAME,
        "transcript_id": transcript.transcript_id,
        "source": SOURCE,
        "recorded_at": recorded_at.isoformat() if recorded_at else None,
        "store_ref": f"sqlite:{Path(db_path).expanduser().resolve()}#{transcript.transcript_id}",
    }


def encode_payload(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=False).encode("utf-8")


def signature_header(body: bytes, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def should_emit(transcript: EnrichedTranscript, cfg: Config) -> bool:
    return should_emit_reason(transcript, cfg)[0]


def should_emit_reason(transcript: EnrichedTranscript, cfg: Config) -> tuple[bool, str]:
    if not cfg.webhook_enabled:
        return False, "disabled"
    if not cfg.webhook_url:
        return False, "missing-url"
    if not cfg.webhook_secret:
        return False, "missing-secret"
    if not transcript.segments:
        return False, "no-segments"
    if cfg.webhook_accept_degraded:
        return True, "ok"
    state = (transcript.metadata.diarization_state or "").lower()
    if state in {"failed", "incomplete"} or state.startswith("failed"):
        return False, f"diarization-{state or 'unknown'}"
    return True, "ok"


def emit_transcript_ready(
    transcript: EnrichedTranscript,
    db_path: str | Path,
    cfg: Config | None = None,
    *,
    timeout: float = 5.0,
) -> bool:
    cfg = cfg or load_config()
    ok, reason = should_emit_reason(transcript, cfg)
    if not ok:
        if reason == "disabled":
            log.debug("webhook disabled; not emitting for %s", transcript.transcript_id)
        else:
            log.warning("webhook not emitted for %s: %s", transcript.transcript_id, reason)
        return False

    body = encode_payload(readiness_payload(transcript, db_path))
    headers = {
        "content-type": "application/json",
        SIGNATURE_HEADER: signature_header(body, cfg.webhook_secret or ""),
    }
    try:
        response = requests.post(cfg.webhook_url, data=body, headers=headers, timeout=timeout)
        if 200 <= response.status_code < 300:
            return True
        target = urlsplit(cfg.webhook_url)
        log.warning(
            "readiness webhook returned status=%s transcript_id=%s target=%s%s",
            response.status_code,
            transcript.transcript_id,
            target.netloc,
            target.path,
        )
    except Exception as exc:
        target = urlsplit(cfg.webhook_url or "")
        log.warning(
            "readiness webhook failed transcript_id=%s target=%s%s error=%s",
            transcript.transcript_id,
            target.netloc,
            target.path,
            type(exc).__name__,
        )
    return False
