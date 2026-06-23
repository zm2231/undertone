from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from undertone_audio.schema import Segment, Speaker


class RawTranscript(BaseModel):
    duration_ms: int
    language: str
    speakers: list[Speaker]
    segments: list[Segment]
    engine: str


class TranscriptionEngine(Protocol):
    name: str

    async def transcribe(self, audio_path: Path) -> RawTranscript: ...

    async def healthcheck(self) -> bool: ...
