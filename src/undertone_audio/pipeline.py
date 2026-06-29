from __future__ import annotations

import uuid
import wave
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Any

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
from undertone_audio.dedupe import (
    DuplicateTranscriptError,
    audio_signature_for_path,
    text_signature_for_segments,
)
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

log = logging.getLogger(__name__)


class AudioPipeline:
    def __init__(
        self,
        store: TranscriptStore,
        engine: TranscriptionEngine | None = None,
        config: Config | None = None,
        fingerprint_store: SpeakerFingerprintStore | None = None,
        warning_sink: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        self.store = store
        self.engine = engine
        self.config = config or load_config()
        self.fingerprint_store = fingerprint_store or SpeakerFingerprintStore(
            self.store.db_path,
            similarity_threshold=self.config.fingerprint_similarity_threshold,
            embedding_model=effective_fingerprint_embedding_model(self.config),
        )
        self.warning_sink = warning_sink

    async def run(
        self,
        audio_path: str | Path,
        transcript_id: str | None = None,
        recorded_at: datetime | None = None,
        source_metadata: dict | None = None,
        expected_speaker_count: int | None = None,
        expected_speaker_source: str | None = None,
        allow_duplicate: bool = False,
    ) -> EnrichedTranscript:
        if self.engine is None:
            raise ValueError("AudioPipeline.run requires a TranscriptionEngine")
        audio_path = Path(audio_path)
        resolved_transcript_id = transcript_id or str(uuid.uuid4())
        audio_signature = audio_signature_for_path(
            audio_path,
            timeout_seconds=self.config.process_timeout_seconds,
        )
        if audio_signature is not None and not allow_duplicate:
            duplicate = self.store.find_audio_duplicate(
                audio_signature.value,
                audio_signature.algorithm,
                exclude_transcript_id=resolved_transcript_id,
            )
            if duplicate is not None:
                raise DuplicateTranscriptError(resolved_transcript_id, duplicate)
        raw = await self.engine.transcribe(audio_path)
        return self.finalize_raw(
            raw,
            transcript_id=resolved_transcript_id,
            recorded_at=recorded_at,
            source_path=str(audio_path),
            source_metadata=source_metadata,
            expected_speaker_count=expected_speaker_count,
            expected_speaker_source=expected_speaker_source,
            audio_format=_audio_format(audio_path),
            audio_path=audio_path,
            allow_duplicate=allow_duplicate,
            content_audio_fp=audio_signature.value if audio_signature else None,
            content_audio_fp_algorithm=audio_signature.algorithm if audio_signature else None,
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
        allow_duplicate: bool = False,
        content_audio_fp: str | None = None,
        content_audio_fp_algorithm: str | None = None,
    ) -> EnrichedTranscript:
        speakers = raw.speakers
        segments = raw.segments
        raw_transcript = raw.model_copy(deep=True)
        fingerprint_plan = None
        safe_source_metadata = sanitize_source_metadata(source_metadata)
        resolved_transcript_id = transcript_id or str(uuid.uuid4())
        active_embedding_model = embedding_backend or fingerprint_embedding_model_for_raw(
            raw,
            self.config,
        )
        if content_audio_fp and content_audio_fp_algorithm and not allow_duplicate:
            duplicate = self.store.find_audio_duplicate(
                content_audio_fp,
                content_audio_fp_algorithm,
                exclude_transcript_id=resolved_transcript_id,
            )
            if duplicate is not None:
                raise DuplicateTranscriptError(resolved_transcript_id, duplicate)

        if apply_speaker_processing:
            self.fingerprint_store.embedding_model = active_embedding_model
            fingerprint_counts = self.store.fingerprint_model_counts(
                active_embedding_model
            )
            if fingerprint_counts["legacy"] or fingerprint_counts["incompatible"]:
                warning_payload = {
                    "legacy": fingerprint_counts["legacy"],
                    "incompatible": fingerprint_counts["incompatible"],
                    "active_embedding_model": fingerprint_counts["active_embedding_model"],
                    "fix": "Run `undertone fingerprint-adopt-model --dry-run` to inspect.",
                }
                log.warning(
                    "%d legacy and %d incompatible voice fingerprints are dormant for model %r; "
                    "run `undertone fingerprint-adopt-model --dry-run` to inspect.",
                    fingerprint_counts["legacy"],
                    fingerprint_counts["incompatible"],
                    fingerprint_counts["active_embedding_model"],
                )
                if self.warning_sink is not None:
                    self.warning_sink("fingerprint_models", warning_payload)
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
            _annotate_asr_confidence(segments)
            text_signature = text_signature_for_segments(segments)
            speakers, fingerprint_plan = self.fingerprint_store.assign_fingerprints(
                speakers,
                persist=False,
                speaker_durations_ms=talk_time_per_speaker(segments),
            )
            for speaker in speakers:
                if speaker.fingerprint_id:
                    fingerprint_plan.sources.append(
                        (
                            speaker.fingerprint_id,
                            resolved_transcript_id,
                            speaker.speaker_id,
                        )
                    )
        else:
            text_signature = text_signature_for_segments(segments)

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
        resolved_model_versions = (
            model_versions
            if model_versions is not None
            else _model_versions_with_raw(
                raw,
                self.config,
                embedding_model=active_embedding_model,
            )
        )

        transcript = EnrichedTranscript(
            transcript_id=resolved_transcript_id,
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
                embedding_backend=active_embedding_model,
                fingerprint_backend=fingerprint_backend
                or (self.config.fingerprint_backend if apply_speaker_processing else None),
                model_versions=resolved_model_versions,
                audio_format=audio_format or {},
                content_text_simhash=text_signature.value if text_signature else None,
                content_text_simhash_algorithm=text_signature.algorithm if text_signature else None,
                content_audio_fp=content_audio_fp,
                content_audio_fp_algorithm=content_audio_fp_algorithm,
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
            fillers_per_speaker[segment.speaker_id] = fillers_per_speaker.get(
                segment.speaker_id, 0
            ) + len(segment.enrichment.fillers)

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


def _annotate_asr_confidence(segments) -> None:
    for segment in segments:
        confidences = [
            word.confidence for word in segment.words if word.confidence is not None
        ]
        segment.asr_confidence = (
            sum(confidences) / len(confidences) if confidences else None
        )


def _diarization_backend(engine: str, config: Config) -> str | None:
    if engine == "fluidaudio-pyannote":
        from undertone_audio.engines.fluidaudio_pyannote import resolve_pyannote_model

        return resolve_pyannote_model(config.pyannote_model)
    if engine == "fluidaudio-hybrid":
        return config.diarization_model
    if engine == "fluidaudio-cli":
        return "FluidAudio process diarization"
    return None


def _embedding_backend(engine: str, config: Config) -> str:
    if engine == "fluidaudio-pyannote":
        from undertone_audio.engines.fluidaudio_pyannote import resolve_pyannote_model

        return resolve_pyannote_model(config.pyannote_model)
    return config.embedding_model


def fingerprint_embedding_model_for_engine(engine: str, config: Config) -> str:
    return _embedding_backend(engine, config)


def fingerprint_embedding_model_for_raw(raw: RawTranscript, config: Config) -> str:
    embedding = raw.model_versions.get("embedding") if raw.model_versions else None
    if isinstance(embedding, str) and embedding.strip():
        return embedding
    return fingerprint_embedding_model_for_engine(raw.engine, config)


def effective_fingerprint_embedding_model(config: Config) -> str:
    return fingerprint_embedding_model_for_engine(config.default_engine, config)


def _model_versions(engine: str, config: Config) -> dict:
    if not engine.startswith("fluidaudio"):
        return {}
    return {
        "asr": config.asr_model,
        "diarization": _diarization_backend(engine, config),
        "vad": config.vad_model,
        "embedding": _embedding_backend(engine, config),
        "fingerprint": config.fingerprint_backend,
    }


def _model_versions_with_raw(
    raw: RawTranscript,
    config: Config,
    *,
    embedding_model: str,
) -> dict:
    resolved = _model_versions(raw.engine, config)
    if raw.model_versions:
        for key, value in raw.model_versions.items():
            if key == "embedding" and not (isinstance(value, str) and value.strip()):
                continue
            resolved[key] = value
    resolved["embedding"] = embedding_model
    return resolved


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
