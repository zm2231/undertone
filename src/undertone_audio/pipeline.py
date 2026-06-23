from __future__ import annotations

import uuid
import wave
from datetime import datetime
from pathlib import Path

from undertone_audio.analytics import (
    annotate_fillers,
    annotate_turn_taking,
    compute_speaker_voice_metrics,
    interruption_counts,
    pause_profile_per_speaker,
    talk_ratio_per_speaker,
    talk_time_per_speaker,
    word_count_per_speaker,
    wpm_per_speaker,
)
from undertone_audio.config import Config, load as load_config
from undertone_audio.diarization.fingerprint import SpeakerFingerprintStore
from undertone_audio.diarization.merge import merge_adjacent_turns
from undertone_audio.diarization.merge_speakers import collapse_overdetected_speakers
from undertone_audio.engines.base import RawTranscript, TranscriptionEngine
from undertone_audio.enrichment import annotate_linguistic, classify_meeting_type
from undertone_audio.privacy import sanitize_source_metadata
from undertone_audio.schema import (
    EnrichedTranscript,
    MeetingType,
    SpeakerVoiceMetrics,
    TranscriptMetadata,
)
from undertone_audio.storage import TranscriptStore
from undertone_audio.webhooks import emit_transcript_ready

class AudioPipeline:
    def __init__(
        self,
        store: TranscriptStore,
        engine: TranscriptionEngine | None = None,
        config: Config | None = None,
        fingerprint_store: SpeakerFingerprintStore | None = None,
    ):
        self.store = store
        self.engine = engine
        self.config = config or load_config()
        self.fingerprint_store = fingerprint_store or SpeakerFingerprintStore(
            self.store.db_path,
            similarity_threshold=self.config.fingerprint_similarity_threshold,
        )

    async def run(
        self,
        audio_path: str | Path,
        transcript_id: str | None = None,
        recorded_at: datetime | None = None,
        source_metadata: dict | None = None,
        expected_speaker_count: int | None = None,
        expected_speaker_source: str | None = None,
    ) -> EnrichedTranscript:
        if self.engine is None:
            raise ValueError("AudioPipeline.run requires a TranscriptionEngine")
        raw = await self.engine.transcribe(Path(audio_path))
        return self.finalize_raw(
            raw,
            transcript_id=transcript_id,
            recorded_at=recorded_at,
            source_path=str(audio_path),
            source_metadata=source_metadata,
            expected_speaker_count=expected_speaker_count,
            expected_speaker_source=expected_speaker_source,
            audio_format=_audio_format(audio_path),
            audio_path=Path(audio_path),
        )

    def finalize_raw(
        self,
        raw: RawTranscript,
        *,
        transcript_id: str | None = None,
        recorded_at: datetime | None = None,
        source_path: str | None = None,
        source_url: str | None = None,
        video_path: str | None = None,
        expected_speaker_count: int | None = None,
        expected_speaker_source: str | None = None,
        source_metadata: dict | None = None,
        diarization_state: str | None = None,
        diarization_error_code: str | None = None,
        diarization_error_detail: str | None = None,
        speaker_metrics: list[SpeakerVoiceMetrics] | None = None,
        asr_backend: str | None = None,
        diarization_backend: str | None = None,
        vad_backend: str | None = None,
        embedding_backend: str | None = None,
        fingerprint_backend: str | None = None,
        model_versions: dict | None = None,
        audio_format: dict | None = None,
        audio_path: Path | None = None,
        meeting_type_override: MeetingType | None = None,
        voice_metrics_mode: str | None = None,
        apply_speaker_processing: bool = True,
    ) -> EnrichedTranscript:
        speakers = raw.speakers
        segments = raw.segments
        raw_transcript = raw.model_copy(deep=True)
        fingerprint_plan = None
        safe_source_metadata = sanitize_source_metadata(source_metadata)

        if apply_speaker_processing:
            speakers, segments, _report = collapse_overdetected_speakers(
                speakers,
                segments,
                merge_threshold=self.config.speaker_merge_threshold,
                min_talk_seconds=self.config.min_talk_seconds,
                target_speaker_count=expected_speaker_count,
            )
            if self.config.enable_turn_taking:
                annotate_turn_taking(segments)
            segments = merge_adjacent_turns(segments, gap_threshold_ms=self.config.turn_gap_ms)
            if self.config.enable_fillers:
                annotate_fillers(segments)
            if self.config.enable_linguistic:
                annotate_linguistic(segments)
            speakers, fingerprint_plan = self.fingerprint_store.assign_fingerprints(
                speakers,
                persist=False,
            )

        inferred_diarization_state = diarization_state
        if inferred_diarization_state is None:
            inferred_diarization_state = "ok" if speakers and segments else "incomplete"

        title_hint = safe_source_metadata.get("title")
        if title_hint is None and source_path:
            title_hint = Path(source_path).stem
        if meeting_type_override is not None:
            meeting_type, meeting_type_confidence = meeting_type_override, 1.0
        elif self.config.enable_meeting_type:
            meeting_type, meeting_type_confidence = classify_meeting_type(
                segments,
                title=title_hint,
            )
        else:
            meeting_type, meeting_type_confidence = MeetingType.UNKNOWN, None

        voice_mode = voice_metrics_mode or self.config.voice_metrics
        voice_metrics = {}
        if voice_mode not in {"off", "false", "0", "none", "skip"}:
            voice_metrics = compute_speaker_voice_metrics(
                audio_path,
                segments,
                required=voice_mode in {"required", "require", "on", "true", "1"},
            )

        transcript = EnrichedTranscript(
            transcript_id=transcript_id or str(uuid.uuid4()),
            metadata=TranscriptMetadata(
                source_path=source_path,
                source_url=source_url,
                video_path=video_path,
                duration_ms=raw.duration_ms,
                language=raw.language,
                meeting_type=meeting_type,
                meeting_type_confidence=meeting_type_confidence,
                recorded_at=recorded_at,
                engine=raw.engine,
                asr_backend=asr_backend or _asr_backend(raw.engine, self.config),
                diarization_backend=diarization_backend
                or _diarization_backend(raw.engine, self.config),
                vad_backend=vad_backend or self.config.vad_model,
                embedding_backend=embedding_backend or self.config.embedding_model,
                fingerprint_backend=fingerprint_backend
                or (self.config.fingerprint_backend if apply_speaker_processing else None),
                model_versions=model_versions or _model_versions(raw.engine, self.config),
                audio_format=audio_format or {},
                expected_speaker_count=expected_speaker_count,
                expected_speaker_source=expected_speaker_source,
                source_metadata=safe_source_metadata,
                diarization_state=inferred_diarization_state,
                diarization_error_code=diarization_error_code,
                diarization_error_detail=diarization_error_detail,
            ),
            speakers=speakers,
            segments=segments,
            speaker_metrics=speaker_metrics
            or self._aggregate_speaker_metrics(segments, raw.duration_ms, voice_metrics),
        )
        try:
            self.store.save_with_fingerprint_plan(
                transcript,
                fingerprint_plan,
                self.fingerprint_store,
                raw_transcript=raw_transcript,
            )
        except Exception:
            if fingerprint_plan is not None:
                fingerprint_plan.discard()
            raise
        emit_transcript_ready(transcript, self.store.db_path, self.config)
        return transcript

    def _aggregate_speaker_metrics(
        self,
        segments,
        total_duration_ms: int,
        voice_metrics: dict[str, dict[str, float]],
    ) -> list[SpeakerVoiceMetrics]:
        talk = talk_time_per_speaker(segments)
        ratio = talk_ratio_per_speaker(segments, total_duration_ms)
        words = word_count_per_speaker(segments)
        wpm = wpm_per_speaker(segments)
        pauses = pause_profile_per_speaker(segments)
        interruptions = interruption_counts(segments)
        fillers_per_speaker: dict[str, int] = {}
        for segment in segments:
            fillers_per_speaker[segment.speaker_id] = (
                fillers_per_speaker.get(segment.speaker_id, 0)
                + len(segment.enrichment.fillers)
            )

        metrics = []
        for speaker_id in talk:
            pause_count, avg_pause = pauses.get(speaker_id, (0, 0.0))
            made, received = interruptions.get(speaker_id, (0, 0))
            voice = voice_metrics.get(speaker_id, {})
            filler_count = fillers_per_speaker.get(speaker_id, 0)
            word_count = words.get(speaker_id, 0)
            metrics.append(
                SpeakerVoiceMetrics(
                    speaker_id=speaker_id,
                    talk_time_ms=talk[speaker_id],
                    talk_ratio=ratio.get(speaker_id, 0.0),
                    word_count=word_count,
                    wpm=wpm.get(speaker_id, 0.0),
                    articulation_rate=voice.get("articulation_rate"),
                    pause_count=pause_count,
                    avg_pause_ms=avg_pause,
                    f0_mean_hz=voice.get("f0_mean_hz"),
                    f0_stdev_hz=voice.get("f0_stdev_hz"),
                    jitter_local=voice.get("jitter_local"),
                    shimmer_local=voice.get("shimmer_local"),
                    voiced_duration_s=voice.get("voiced_duration_s"),
                    filler_count=filler_count,
                    filler_rate=(filler_count / word_count * 100) if word_count else 0.0,
                    interruptions_made=made,
                    interruptions_received=received,
                )
            )
        return metrics


def _asr_backend(engine: str, config: Config) -> str | None:
    if engine.startswith("fluidaudio"):
        return config.asr_model
    return None


def _diarization_backend(engine: str, config: Config) -> str | None:
    if engine == "fluidaudio-hybrid":
        return config.diarization_model
    if engine == "fluidaudio-cli":
        return "FluidAudio process diarization"
    return None


def _model_versions(engine: str, config: Config) -> dict:
    if not engine.startswith("fluidaudio"):
        return {}
    return {
        "asr": config.asr_model,
        "diarization": config.diarization_model
        if engine == "fluidaudio-hybrid"
        else "FluidAudio process diarization",
        "vad": config.vad_model,
        "embedding": config.embedding_model,
        "fingerprint": config.fingerprint_backend,
    }


def _audio_format(audio_path: str | Path) -> dict:
    path = Path(audio_path)
    result = {"path": str(path), "suffix": path.suffix.lower()}
    if path.suffix.lower() != ".wav":
        return result
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            frame_rate = wav.getframerate()
            result.update(
                {
                    "container": "wav",
                    "channels": wav.getnchannels(),
                    "sample_rate_hz": frame_rate,
                    "sample_width_bytes": wav.getsampwidth(),
                    "duration_ms": int((frames / frame_rate) * 1000) if frame_rate else 0,
                }
            )
    except (EOFError, wave.Error):
        result["container"] = "wav"
        result["parse_error"] = "invalid-wav"
    return result
