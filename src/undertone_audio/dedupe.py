from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from collections.abc import Iterable
from pathlib import Path

from undertone_audio.processes import run_process_sync

TEXT_SIMHASH_ALGORITHM = "simhash64-token-word4-v1"
TEXT_SIMHASH_MIN_TOKENS = 6
AUDIO_FINGERPRINT_ALGORITHM = "chromaprint-fpcalc-v1"


@dataclass(frozen=True)
class ContentDuplicate:
    transcript_id: str
    match_type: str
    algorithm: str
    distance: int | None = None


class DuplicateTranscriptError(RuntimeError):
    def __init__(self, attempted_transcript_id: str, duplicate: ContentDuplicate):
        self.attempted_transcript_id = attempted_transcript_id
        self.duplicate = duplicate
        super().__init__(
            f"content duplicate of {duplicate.transcript_id}; "
            f"pass --allow-duplicate to ingest anyway"
        )

    def payload(self) -> dict:
        return {
            "transcript_id": self.attempted_transcript_id,
            "skipped": True,
            "reason": "duplicate",
            "existing_transcript_id": self.duplicate.transcript_id,
            "match_type": self.duplicate.match_type,
            "algorithm": self.duplicate.algorithm,
            "distance": self.duplicate.distance,
        }


@dataclass(frozen=True)
class TextSignature:
    value: str
    algorithm: str = TEXT_SIMHASH_ALGORITHM


@dataclass(frozen=True)
class AudioSignature:
    value: str
    algorithm: str = AUDIO_FINGERPRINT_ALGORITHM


def text_signature_for_segments(segments) -> TextSignature | None:
    return text_signature_for_texts(segment.text for segment in segments)


def text_signature_for_texts(texts: Iterable[str]) -> TextSignature | None:
    tokens = _tokens(" ".join(texts))
    if len(tokens) < TEXT_SIMHASH_MIN_TOKENS:
        return None
    shingles = tokens + _shingles(tokens, size=4)
    return TextSignature(_simhash64(shingles))


def hamming_distance_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def audio_signature_for_path(
    audio_path: Path,
    *,
    timeout_seconds: float | None,
    fpcalc_bin: str = "fpcalc",
) -> AudioSignature | None:
    if not shutil.which(fpcalc_bin):
        return None
    try:
        result = run_process_sync(
            [fpcalc_bin, "-raw", str(audio_path)],
            label="fpcalc",
            timeout_seconds=timeout_seconds,
        )
    except RuntimeError:
        return None
    fingerprint = None
    for line in result.stdout.splitlines():
        if line.startswith("FINGERPRINT="):
            fingerprint = line.split("=", 1)[1].strip()
            break
    if not fingerprint:
        return None
    return AudioSignature(hashlib.blake2b(fingerprint.encode(), digest_size=16).hexdigest())


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _shingles(tokens: list[str], *, size: int) -> list[str]:
    if len(tokens) <= size:
        return [" ".join(tokens)]
    return [" ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)]


def _simhash64(features: list[str]) -> str:
    weights = [0] * 64
    for feature in features:
        digest = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
        for bit in range(64):
            if digest & (1 << bit):
                weights[bit] += 1
            else:
                weights[bit] -= 1
    value = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            value |= 1 << bit
    return f"{value:016x}"
