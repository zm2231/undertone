import json
from pathlib import Path

import pytest

from undertone_audio import Segment, Speaker
from undertone_audio.config import Config
from undertone_audio.engines.base import RawTranscript
from undertone_audio.export import render_transcript
from undertone_audio.pipeline import AudioPipeline
from undertone_audio.storage import TranscriptStore


def _raw():
    return RawTranscript(
        duration_ms=2200,
        language="en",
        engine="fluidaudio-hybrid",
        speakers=[
            Speaker(speaker_id="S1", embedding=[1.0, 0.0]),
            Speaker(speaker_id="S2", embedding=[0.0, 1.0]),
        ],
        segments=[
            Segment(
                segment_id="a",
                speaker_id="S1",
                start_ms=0,
                end_ms=1000,
                text="um I think we should decide because it matters",
            ),
            Segment(
                segment_id="b",
                speaker_id="S2",
                start_ms=700,
                end_ms=2200,
                text="actually yes and also next steps",
            ),
        ],
    )


def test_full_audio_enrichment_without_voice_dependency(tmp_path):
    db = tmp_path / "undertone.db"
    store = TranscriptStore(db)
    try:
        pipeline = AudioPipeline(
            store=store,
            config=Config(
                db_path=db,
                webhook_enabled=False,
                voice_metrics="off",
                min_talk_seconds=0,
            ),
        )
        transcript = pipeline.finalize_raw(
            _raw(),
            transcript_id="full",
            source_metadata={"title": "Weekly huddle"},
        )
    finally:
        store.close()

    assert transcript.metadata.meeting_type.value == "huddle"
    assert transcript.segments[0].enrichment.fillers == ["um"]
    assert transcript.segments[0].enrichment.linguistic.cognitive_process == 2
    assert transcript.segments[0].enrichment.linguistic.causation == 1
    assert transcript.segments[1].enrichment.is_interruption is True
    metrics = {metric.speaker_id: metric for metric in transcript.speaker_metrics}
    assert metrics["S1"].filler_count == 1
    assert metrics["S2"].interruptions_made == 1
    assert metrics["S1"].interruptions_received == 1


def test_required_voice_metrics_surfaces_missing_dependency(tmp_path, monkeypatch):
    def missing_import(name, *args, **kwargs):
        if name == "parselmouth":
            raise ImportError("missing parselmouth")
        return real_import(name, *args, **kwargs)

    real_import = __import__
    monkeypatch.setattr("builtins.__import__", missing_import)

    db = tmp_path / "undertone.db"
    store = TranscriptStore(db)
    try:
        pipeline = AudioPipeline(
            store=store,
            config=Config(db_path=db, webhook_enabled=False, voice_metrics="required"),
        )
        with pytest.raises(RuntimeError, match="voice metrics required"):
            pipeline.finalize_raw(
                _raw(),
                transcript_id="voice-required",
                source_path="audio.wav",
                audio_path=Path("audio.wav"),
            )
    finally:
        store.close()


def test_required_voice_metrics_requires_audio_path(tmp_path):
    db = tmp_path / "undertone.db"
    store = TranscriptStore(db)
    try:
        pipeline = AudioPipeline(
            store=store,
            config=Config(db_path=db, webhook_enabled=False, voice_metrics="required"),
        )
        with pytest.raises(RuntimeError, match="no audio_path"):
            pipeline.finalize_raw(_raw(), transcript_id="voice-no-audio")
    finally:
        store.close()


def test_voice_metric_point_process_failures_do_not_drop_pitch(monkeypatch):
    import numpy as np
    from undertone_audio.analytics.voice import _analyze_speaker

    class FakePitch:
        selected_array = {"frequency": np.array([100.0, 110.0, 0.0])}

    class FakeIntensity:
        values = np.array([[30.0, 20.0, 30.0]])

    class FakeSound:
        def __init__(self, samples, sampling_frequency):
            self.samples = samples
            self.sampling_frequency = sampling_frequency

        def to_pitch(self, **kwargs):
            return FakePitch()

        def to_intensity(self, **kwargs):
            return FakeIntensity()

        def to_point_process_cc(self, **kwargs):
            raise AttributeError("not available")

    class FakePraat:
        @staticmethod
        def call(*args, **kwargs):
            raise AttributeError("not available")

    class FakeParselmouth:
        Sound = FakeSound
        praat = FakePraat()

    result = _analyze_speaker(np.ones(16000, dtype="float32"), 16000, np, FakeParselmouth)

    assert result["f0_mean_hz"] == 105.0
    assert result["jitter_local"] == 0.0
    assert result["shimmer_local"] == 0.0


def test_output_formats_render_expected_shapes(tmp_path):
    db = tmp_path / "undertone.db"
    store = TranscriptStore(db)
    try:
        pipeline = AudioPipeline(
            store=store,
            config=Config(
                db_path=db,
                webhook_enabled=False,
                voice_metrics="off",
                min_talk_seconds=0,
            ),
        )
        transcript = pipeline.finalize_raw(_raw(), transcript_id="formats")
    finally:
        store.close()

    json_payload = json.loads(render_transcript(transcript, "json"))
    assert json_payload["transcript_id"] == "formats"
    json_speakers = {speaker["speaker_id"]: speaker for speaker in json_payload["speakers"]}
    assert json_speakers["S1"]["embedding"] == [1.0, 0.0]
    assert json_speakers["S1"]["match"]["kind"] == "no_enroll"
    raw_payload = json.loads(render_transcript(transcript, "raw-json"))
    assert raw_payload["engine"] == "fluidaudio-hybrid"
    raw_speakers = {speaker["speaker_id"]: speaker for speaker in raw_payload["speakers"]}
    assert raw_speakers["S1"]["match"]["kind"] == "no_enroll"
    assert "SPEAKERS" in render_transcript(transcript, "text")
    assert "## Transcript" in render_transcript(transcript, "md")
    rows = [json.loads(line) for line in render_transcript(transcript, "jsonl").splitlines()]
    assert rows[0]["segment_id"] == "a"
    assert rows[0]["enrichment"]["fillers"] == ["um"]
    assert rows[0]["speaker_match_kind"] == "no_enroll"
    assert rows[0]["speaker_match_similarity"] is None
    assert rows[0]["speaker_match_second_similarity"] is None
    assert rows[0]["speaker_match_margin"] is None
    assert rows[0]["speaker_match_similarity_threshold"] == 0.78
    assert rows[0]["speaker_match_embedding_model"] == "FluidAudio pyannote-derived speaker embeddings"
    csv_payload = render_transcript(transcript, "csv")
    assert (
        "fingerprint_id,match_kind,match_similarity,match_second_similarity,"
        "match_margin,match_similarity_threshold,match_embedding_model"
    ) in csv_payload
    assert ",no_enroll,,,,0.78,FluidAudio pyannote-derived speaker embeddings," in csv_payload


def test_output_detail_profiles_control_exported_metrics(tmp_path):
    db = tmp_path / "undertone.db"
    store = TranscriptStore(db)
    try:
        pipeline = AudioPipeline(
            store=store,
            config=Config(
                db_path=db,
                webhook_enabled=False,
                voice_metrics="off",
                min_talk_seconds=0,
            ),
        )
        transcript = pipeline.finalize_raw(_raw(), transcript_id="detail")
        transcript.speaker_metrics[0].f0_mean_hz = 180.0
        transcript.speaker_metrics[0].jitter_local = 0.0123
    finally:
        store.close()

    full = json.loads(render_transcript(transcript, "json", detail="full"))
    standard = json.loads(render_transcript(transcript, "json", detail="standard"))
    minimal = json.loads(render_transcript(transcript, "json", detail="minimal"))

    assert "words" in full["segments"][0]
    assert full["speaker_metrics"][0]["jitter_local"] == 0.0123
    assert "words" not in standard["segments"][0]
    assert "enrichment" in standard["segments"][0]
    assert "jitter_local" not in standard["speaker_metrics"][0]
    assert minimal["speaker_metrics"] == []
    assert "enrichment" not in minimal["segments"][0]

    text_standard = render_transcript(transcript, "text", detail="standard")
    text_full = render_transcript(transcript, "text", detail="full")
    assert "jitter=" not in text_standard
    assert "jitter=0.0123" in text_full


def test_source_metadata_strips_private_boundary_fields(tmp_path):
    db = tmp_path / "undertone.db"
    store = TranscriptStore(db)
    try:
        pipeline = AudioPipeline(
            store=store,
            config=Config(
                db_path=db,
                webhook_enabled=False,
                voice_metrics="off",
                min_talk_seconds=0,
            ),
        )
        transcript = pipeline.finalize_raw(
            _raw(),
            transcript_id="privacy",
            source_metadata={
                "title": "Weekly huddle",
                "scope": "WORK",
                "attendees": ["person@example.com"],
                "owner_email": "owner@example.com",
                "crm_key": "abc",
                "nested": {"project": "secret", "room": "studio"},
            },
        )
    finally:
        store.close()

    assert transcript.metadata.source_metadata == {
        "title": "Weekly huddle",
        "nested": {"room": "studio"},
    }


def test_store_save_strips_private_source_metadata(tmp_path):
    from undertone_audio.schema import EnrichedTranscript

    db = tmp_path / "undertone.db"
    store = TranscriptStore(db)
    try:
        store.save(
            EnrichedTranscript.model_validate(
                {
                    "transcript_id": "direct",
                    "metadata": {
                        "duration_ms": 1000,
                        "engine": "fixture",
                        "source_metadata": {
                            "title": "Audio",
                            "ownerEmail": "owner@example.com",
                            "crm_key": "abc",
                            "nested": {"client_id": "hidden", "device": "mic"},
                        },
                    },
                    "speakers": [{"speaker_id": "S1"}],
                    "segments": [
                        {
                            "segment_id": "seg1",
                            "speaker_id": "S1",
                            "start_ms": 0,
                            "end_ms": 1000,
                            "text": "hello",
                        }
                    ],
                }
            )
        )
        loaded = store.load("direct")
    finally:
        store.close()

    assert loaded is not None
    assert loaded.metadata.source_metadata == {
        "title": "Audio",
        "nested": {"device": "mic"},
    }


def test_raw_json_uses_persisted_pre_enrichment_transcript(tmp_path):
    db = tmp_path / "undertone.db"
    store = TranscriptStore(db)
    try:
        pipeline = AudioPipeline(
            store=store,
            config=Config(
                db_path=db,
                webhook_enabled=False,
                voice_metrics="off",
                min_talk_seconds=0,
            ),
        )
        pipeline.finalize_raw(_raw(), transcript_id="raw-stable")
        loaded = store.load("raw-stable")
        raw = store.load_raw("raw-stable")
    finally:
        store.close()

    assert loaded is not None
    assert raw is not None
    assert loaded.segments[0].enrichment.fillers == ["um"]
    raw_payload = json.loads(render_transcript(loaded, "raw-json", raw=raw))
    assert "enrichment" not in raw_payload["segments"][0]
    assert raw_payload["segments"][0]["speaker_id"] == "S1"
