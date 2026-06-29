from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from undertone_audio.privacy import sanitize_source_metadata


class MeetingType(str, Enum):
    DISCOVERY = "discovery"
    DEMO = "demo"
    STATUS = "status"
    ONE_ON_ONE = "1on1"
    BRAINSTORM = "brainstorm"
    RETRO = "retro"
    INTERVIEW = "interview"
    HUDDLE = "huddle"
    WORKSHOP = "workshop"
    COMMUNITY = "community"
    SALES = "sales"
    TRAINING = "training"
    UNKNOWN = "unknown"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class Word(BaseModel):
    text: str
    start_ms: int
    end_ms: int
    confidence: float | None = None


class Speaker(BaseModel):
    speaker_id: str
    fingerprint_id: str | None = None
    display_name: str | None = None
    embedding: list[float] | None = None


class LinguisticFeatures(BaseModel):
    word_count: int
    cognitive_process: int = 0
    tentative: int = 0
    certainty: int = 0
    inclusive: int = 0
    exclusive: int = 0
    insight: int = 0
    causation: int = 0


class SegmentEnrichment(BaseModel):
    sentiment: Sentiment | None = None
    sentiment_confidence: float | None = None
    tone_tags: list[str] = Field(default_factory=list)
    is_interruption: bool = False
    overlap_with_prev_ms: int = 0
    gap_before_ms: int = 0
    fillers: list[str] = Field(default_factory=list)
    linguistic: LinguisticFeatures | None = None


class Segment(BaseModel):
    segment_id: str
    speaker_id: str
    start_ms: int
    end_ms: int
    text: str
    asr_confidence: float | None = None
    diarization_quality: float | None = None
    words: list[Word] = Field(default_factory=list)
    enrichment: SegmentEnrichment = Field(default_factory=SegmentEnrichment)

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


class SpeakerVoiceMetrics(BaseModel):
    speaker_id: str
    talk_time_ms: int
    talk_ratio: float
    word_count: int
    wpm: float
    articulation_rate: float | None = None
    pause_count: int = 0
    avg_pause_ms: float = 0.0
    f0_mean_hz: float | None = None
    f0_stdev_hz: float | None = None
    jitter_local: float | None = None
    shimmer_local: float | None = None
    voiced_duration_s: float | None = None
    filler_count: int = 0
    filler_rate: float = 0.0
    interruptions_made: int = 0
    interruptions_received: int = 0


class TranscriptMetadata(BaseModel):
    source_path: str | None = None
    source_url: str | None = None
    video_path: str | None = None
    duration_ms: int
    language: str = "en"
    meeting_type: MeetingType = MeetingType.UNKNOWN
    meeting_type_confidence: float | None = None
    recorded_at: datetime | None = None
    engine: str
    asr_backend: str | None = None
    diarization_backend: str | None = None
    vad_backend: str | None = None
    embedding_backend: str | None = None
    fingerprint_backend: str | None = None
    model_versions: dict = Field(default_factory=dict)
    audio_format: dict = Field(default_factory=dict)
    expected_speaker_count: int | None = None
    expected_speaker_source: str | None = None
    source_metadata: dict = Field(default_factory=dict)
    diarization_state: str | None = None
    diarization_error_code: str | None = None
    diarization_error_detail: str | None = None
    pipeline_version: str = "0.1.0"

    @field_validator("source_metadata")
    @classmethod
    def _sanitize_source_metadata(cls, value: dict) -> dict:
        return sanitize_source_metadata(value)


class EnrichedTranscript(BaseModel):
    transcript_id: str
    store_ref: str | None = None
    metadata: TranscriptMetadata
    speakers: list[Speaker]
    segments: list[Segment]
    speaker_metrics: list[SpeakerVoiceMetrics] = Field(default_factory=list)
    schema_version: Literal["1"] = "1"


class ConnectorAssetSchema(BaseModel):
    schema_version: Literal["1"] = "1"
    audio_path: str
    source_url: str
    source_kind: str
    title: str | None = None
    transcript_id_hint: str | None = None
    recorded_at: str | None = None
    metadata: dict = Field(default_factory=dict)


class ConnectorCandidateSchema(BaseModel):
    schema_version: Literal["1"] = "1"
    candidate_id: str
    extractor: str | None = None
    extractor_key: str | None = None
    webpage_url: str | None = None
    original_url: str
    url: str | None = None
    media_id: str | None = None
    format_id: str | None = None
    title: str | None = None
    duration: float | None = None
    kind: Literal[
        "page-voiceover",
        "external-video",
        "podcast-enclosure",
        "generic-media",
        "unsupported",
    ] = "generic-media"
    availability: Literal[
        "downloadable",
        "requires-auth",
        "found-but-unavailable",
        "unsupported",
    ] = "downloadable"
    reason: str | None = None
    metadata: dict = Field(default_factory=dict)
