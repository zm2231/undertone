from undertone_audio.diarization.fingerprint import SpeakerFingerprintStore
from undertone_audio.diarization.merge_speakers import collapse_overdetected_speakers
from undertone_audio.engines.fluidaudio_cli import (
    FluidAudioCLIEngine,
    FluidAudioModelSelection,
    merge_transcribe_and_diarize,
)
from undertone_audio.engines.fluidaudio_hybrid import (
    FluidAudioHybridEngine,
    map_speakers_by_overlap,
    merge_hybrid,
)
from undertone_audio.engines.fluidaudio_pyannote import (
    FluidAudioPyannoteEngine,
    merge_pyannote,
    pyannote_output_to_sortformer_json,
    resolve_pyannote_model,
)
from undertone_audio.schema import Segment, Speaker
from undertone_audio.storage import TranscriptStore


def test_fluidaudio_cli_merge_preserves_embeddings_and_words():
    raw = merge_transcribe_and_diarize(
        {
            "wordTimings": [
                {"word": "hello", "startTime": 0.1, "endTime": 0.3, "confidence": 0.9},
                {"word": "world", "startTime": 0.4, "endTime": 0.7, "confidence": 0.8},
            ]
        },
        {
            "durationSeconds": 1.0,
            "segments": [
                {
                    "speakerId": "S1",
                    "startTimeSeconds": 0.0,
                    "endTimeSeconds": 1.0,
                    "embedding": [1.0, 0.0],
                }
            ],
        },
    )

    assert raw.engine == "fluidaudio-cli"
    assert raw.speakers == [Speaker(speaker_id="S1", embedding=[1.0, 0.0])]
    assert raw.segments[0].text == "hello world"
    assert raw.segments[0].words[0].confidence == 0.9


def test_fluidaudio_cli_custom_model_selection_reaches_commands(tmp_path):
    engine = FluidAudioCLIEngine(
        cli_path="/bin/echo",
        model_selection=FluidAudioModelSelection(
            asr_model="custom-asr",
            diarization_model="custom-diar",
            vad_model="custom-vad",
            embedding_model="custom-embed",
        ),
    )

    transcribe_cmd = engine._transcribe_cmd(tmp_path / "in.wav", tmp_path / "asr.json")
    process_cmd = engine._process_cmd(tmp_path / "in.wav", tmp_path / "process.json")

    assert transcribe_cmd[-2:] == ["--model", "custom-asr"]
    assert "--embedding-model" in process_cmd
    assert "custom-embed" in process_cmd
    assert "--vad-model" in process_cmd
    assert "custom-vad" in process_cmd
    assert process_cmd[-2:] == ["--model", "custom-diar"]


def test_fluidaudio_hybrid_custom_diarization_selection_reaches_sortformer(tmp_path):
    engine = FluidAudioHybridEngine(
        cli_path="/bin/echo",
        model_selection=FluidAudioModelSelection(diarization_model="custom-sortformer"),
    )

    cmd = engine._sortformer_cmd(tmp_path / "in.wav", tmp_path / "sort.json")

    assert cmd[-2:] == ["--model", "custom-sortformer"]


def test_fluidaudio_hybrid_maps_sortformer_speakers_to_process_embeddings():
    raw = merge_hybrid(
        {
            "wordTimings": [
                {"word": "first", "startTime": 0.1, "endTime": 0.2},
                {"word": "second", "startTime": 1.1, "endTime": 1.2},
            ]
        },
        {
            "segments": [
                {
                    "speakerId": "S1",
                    "startTimeSeconds": 0.0,
                    "endTimeSeconds": 0.9,
                    "embedding": [1.0, 0.0],
                },
                {
                    "speakerId": "S2",
                    "startTimeSeconds": 1.0,
                    "endTimeSeconds": 2.0,
                    "embedding": [0.0, 1.0],
                },
            ]
        },
        {
            "durationSeconds": 2.0,
            "segments": [
                {"speaker": "Speaker 0", "startTimeSeconds": 0.0, "endTimeSeconds": 1.0},
                {"speaker": "Speaker 1", "startTimeSeconds": 1.0, "endTimeSeconds": 2.0},
            ],
        },
    )

    assert raw.engine == "fluidaudio-hybrid"
    assert raw.speakers[0].speaker_id == "Speaker 0"
    assert raw.speakers[0].embedding == [1.0, 0.0]
    assert raw.speakers[1].embedding == [0.0, 1.0]
    assert [segment.text for segment in raw.segments] == ["first", "second"]


def test_fluidaudio_hybrid_falls_back_to_process_when_sortformer_empty():
    raw = merge_hybrid(
        {
            "wordTimings": [
                {"word": "fallback", "startTime": 0.1, "endTime": 0.5},
            ]
        },
        {
            "durationSeconds": 1.0,
            "segments": [
                {
                    "speakerId": "S1",
                    "startTimeSeconds": 0.0,
                    "endTimeSeconds": 1.0,
                    "embedding": [1.0, 0.0],
                }
            ],
        },
        {"durationSeconds": 1.0, "segments": []},
    )

    assert raw.engine == "fluidaudio-cli"
    assert raw.speakers == [Speaker(speaker_id="S1", embedding=[1.0, 0.0])]
    assert raw.segments[0].speaker_id == "S1"
    assert raw.segments[0].text == "fallback"


def test_fluidaudio_pyannote_merge_uses_pyannote_embeddings_and_spans():
    raw = merge_pyannote(
        {
            "wordTimings": [
                {"word": "first", "startTime": 0.1, "endTime": 0.3},
                {"word": "second", "startTime": 1.1, "endTime": 1.3},
            ]
        },
        {
            "durationSeconds": 2.0,
            "segments": [
                {"speaker": "Speaker 0", "startTimeSeconds": 0.0, "endTimeSeconds": 0.9},
                {"speaker": "Speaker 1", "startTimeSeconds": 1.0, "endTimeSeconds": 2.0},
            ],
            "speakerEmbeddings": {
                "Speaker 0": [1.0, 0.0],
                "Speaker 1": [0.0, 1.0],
            },
        },
    )

    assert raw.engine == "fluidaudio-pyannote"
    assert raw.speakers == [
        Speaker(speaker_id="Speaker 0", embedding=[1.0, 0.0]),
        Speaker(speaker_id="Speaker 1", embedding=[0.0, 1.0]),
    ]
    assert [segment.text for segment in raw.segments] == ["first", "second"]


def test_pyannote_output_conversion_preserves_labels_embeddings_and_alias():
    class FakeSegment:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    class FakeAnnotation:
        def labels(self):
            return ["SPEAKER_B", "SPEAKER_A"]

        def itertracks(self, yield_label=False):
            assert yield_label is True
            yield FakeSegment(1.0, 2.5), None, "SPEAKER_A"
            yield FakeSegment(0.0, 0.8), None, "SPEAKER_B"

    class FakeOutput:
        speaker_diarization = FakeAnnotation()
        speaker_embeddings = {
            "SPEAKER_A": [0.0, 1.0],
            "SPEAKER_B": [1.0, 0.0],
        }

    converted = pyannote_output_to_sortformer_json(
        FakeOutput(),
        model="community-1",
        duration_seconds=2.5,
    )

    assert converted["model"] == "pyannote/speaker-diarization-community-1"
    assert converted["speakerCount"] == 2
    assert [segment["speaker"] for segment in converted["segments"]] == [
        "Speaker 1",
        "Speaker 0",
    ]
    assert converted["speakerEmbeddings"] == {
        "Speaker 0": [0.0, 1.0],
        "Speaker 1": [1.0, 0.0],
    }


def test_pyannote_array_embeddings_align_to_sorted_labels():
    import numpy as np

    class FakeSegment:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    class FakeAnnotation:
        def labels(self):
            return ["SPEAKER_B", "SPEAKER_A"]

        def itertracks(self, yield_label=False):
            yield FakeSegment(0.0, 1.0), None, "SPEAKER_A"
            yield FakeSegment(1.0, 2.0), None, "SPEAKER_B"

    class FakeOutput:
        speaker_diarization = FakeAnnotation()
        speaker_embeddings = np.array([[0.0, 1.0], [1.0, 0.0]])

    converted = pyannote_output_to_sortformer_json(FakeOutput(), duration_seconds=2.0)

    assert converted["speakerEmbeddings"] == {
        "Speaker 0": [0.0, 1.0],
        "Speaker 1": [1.0, 0.0],
    }


def test_pyannote_clean_vector_rejects_empty_and_non_finite():
    import numpy as np

    from undertone_audio.engines.fluidaudio_pyannote import _clean_vector

    assert _clean_vector([1.0, 2.0]) == [1.0, 2.0]
    assert _clean_vector([]) is None
    assert _clean_vector([float("inf"), 1.0]) is None
    assert _clean_vector(np.array([])) is None
    assert _clean_vector(np.array([float("nan"), 0.0])) is None


def test_pyannote_provenance_resolves_alias_and_embedding_backend():
    from pathlib import Path

    from undertone_audio.config import Config
    from undertone_audio.pipeline import _diarization_backend, _embedding_backend

    config = Config(db_path=Path("/tmp/undertone-test.db"), pyannote_model="community-1")
    resolved = "pyannote/speaker-diarization-community-1"
    assert _diarization_backend("fluidaudio-pyannote", config) == resolved
    assert _embedding_backend("fluidaudio-pyannote", config) == resolved
    assert _embedding_backend("fluidaudio-hybrid", config) == config.embedding_model


def test_fluidaudio_pyannote_does_not_start_pyannote_when_asr_fails():
    import asyncio
    from pathlib import Path

    import pytest

    from undertone_audio.engines.fluidaudio_pyannote import FluidAudioPyannoteEngine

    engine = FluidAudioPyannoteEngine(cli_path="/tmp/fake-fluidaudiocli")
    started = {"pyannote": False}

    async def failing_asr(cmd, label):
        raise RuntimeError("asr boom")

    def spy_pyannote(audio_path):
        started["pyannote"] = True
        return {}

    engine._run = failing_asr
    engine._run_pyannote = spy_pyannote

    with pytest.raises(RuntimeError, match="asr boom"):
        asyncio.run(engine.transcribe(Path("missing.wav")))

    assert started["pyannote"] is False


def test_pyannote_model_aliases_are_portable():
    assert resolve_pyannote_model("community-1") == "pyannote/speaker-diarization-community-1"
    assert resolve_pyannote_model("3.1") == "pyannote/speaker-diarization-3.1"
    assert resolve_pyannote_model("org/custom") == "org/custom"


def test_fluidaudio_pyannote_engine_has_no_local_path_configuration():
    engine = FluidAudioPyannoteEngine(
        cli_path="/bin/echo",
        pyannote_model="community-1",
        pyannote_device="cpu",
    )

    assert engine.pyannote_model == "pyannote/speaker-diarization-community-1"
    assert engine.pyannote_device == "cpu"
    assert not hasattr(engine, "pyannote_python")
    assert not hasattr(engine, "pyannote_cli")


def test_fluidaudio_pyannote_derives_duration_from_loaded_waveform(monkeypatch, tmp_path):
    class FakeWaveform:
        shape = (1, 32_000)

    class FakeTorchaudio:
        @staticmethod
        def load(path):
            return FakeWaveform(), 16_000

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __call__(self, payload):
            assert payload["waveform"].shape[-1] == 32_000
            assert payload["sample_rate"] == 16_000
            return FakeAnnotation()

    class FakeAnnotation:
        speaker_diarization = None

        def labels(self):
            return []

        def itertracks(self, yield_label=False):
            return iter(())

    engine = FluidAudioPyannoteEngine(
        cli_path="/bin/echo",
        pyannote_model="community-1",
        pyannote_device="none",
    )
    monkeypatch.setattr(
        "undertone_audio.engines.fluidaudio_pyannote._load_pyannote_modules",
        lambda: (object(), FakeTorchaudio, FakePipeline),
    )

    data = engine._run_pyannote(tmp_path / "audio.wav")

    assert data["durationSeconds"] == 2.0


def test_overlap_mapping_uses_max_overlap():
    mapping = map_speakers_by_overlap(
        [{"speaker": "Speaker 0", "startTimeSeconds": 0.0, "endTimeSeconds": 3.0}],
        [
            {"speakerId": "S1", "startTimeSeconds": 0.0, "endTimeSeconds": 1.0},
            {"speakerId": "S2", "startTimeSeconds": 1.0, "endTimeSeconds": 3.0},
        ],
    )

    assert mapping == {"Speaker 0": "S2"}


def test_collapse_overdetected_speakers_rewrites_segments():
    speakers, segments, report = collapse_overdetected_speakers(
        [
            Speaker(speaker_id="S1", embedding=[1.0, 0.0]),
            Speaker(speaker_id="S2", embedding=[0.99, 0.01]),
        ],
        [
            Segment(segment_id="a", speaker_id="S1", start_ms=0, end_ms=2000, text="a"),
            Segment(segment_id="b", speaker_id="S2", start_ms=2100, end_ms=4000, text="b"),
        ],
        merge_threshold=0.9,
        min_talk_seconds=0,
    )

    assert [speaker.speaker_id for speaker in speakers] == ["S1"]
    assert {segment.speaker_id for segment in segments} == {"S1"}
    assert report.original_speaker_count == 2
    assert report.final_speaker_count == 1


def test_fingerprint_store_assigns_and_updates_mean(tmp_path):
    db = tmp_path / "undertone.db"
    store = TranscriptStore(db)
    try:
        fingerprints = SpeakerFingerprintStore(db, similarity_threshold=0.95)
        speakers, plan = fingerprints.assign_fingerprints(
            [Speaker(speaker_id="S1", embedding=[1.0, 0.0])],
            persist=False,
        )
        assert speakers[0].fingerprint_id is not None
        assert fingerprints.list_all() == []
        plan.commit()

        matched, plan = fingerprints.assign_fingerprints(
            [Speaker(speaker_id="S2", embedding=[0.99, 0.01])],
            persist=False,
        )
        assert matched[0].fingerprint_id == speakers[0].fingerprint_id
        plan.commit()

        row = store._conn.execute(
            "SELECT sample_count FROM speaker_fingerprints WHERE fingerprint_id = ?",
            (speakers[0].fingerprint_id,),
        ).fetchone()
        assert row["sample_count"] == 2
    finally:
        store.close()


def test_store_roundtrips_backend_provenance(tmp_path):
    db = tmp_path / "undertone.db"
    store = TranscriptStore(db)
    try:
        raw_json = {
            "transcript_id": "prov",
            "metadata": {
                "duration_ms": 1000,
                "engine": "fluidaudio-hybrid",
                "asr_backend": "FluidAudio Parakeet TDT",
                "diarization_backend": "FluidAudio Sortformer",
                "vad_backend": "FluidAudio/Silero VAD",
                "embedding_backend": "FluidAudio pyannote-derived speaker embeddings",
                "fingerprint_backend": "undertone-speaker-fingerprints",
                "model_versions": {"asr": "fixture"},
                "audio_format": {"container": "wav", "sample_rate_hz": 16000},
            },
            "speakers": [{"speaker_id": "S1"}],
            "segments": [
                {
                    "segment_id": "seg1",
                    "speaker_id": "S1",
                    "start_ms": 0,
                    "end_ms": 1000,
                    "text": "hi",
                }
            ],
        }
        from undertone_audio.schema import EnrichedTranscript

        store.save(EnrichedTranscript.model_validate(raw_json))
        loaded = store.load("prov")
    finally:
        store.close()

    assert loaded is not None
    assert loaded.metadata.asr_backend == "FluidAudio Parakeet TDT"
    assert loaded.metadata.model_versions == {"asr": "fixture"}
    assert loaded.metadata.audio_format["sample_rate_hz"] == 16000


def test_margin_match_recovers_cross_channel_speaker(tmp_path):
    db = tmp_path / "undertone.db"
    store = TranscriptStore(db)
    try:
        fp = SpeakerFingerprintStore(db)
        # Enroll a canonical print (strong, long).
        enrolled, plan = fp.assign_fingerprints(
            [Speaker(speaker_id="MAX", embedding=[1.0, 0.0, 0.0])],
            persist=False,
            speaker_durations_ms={"MAX": 30_000},
        )
        plan.commit()
        canonical_id = enrolled[0].fingerprint_id
        # Add a clearly-different second print so a runner-up exists.
        _, plan2 = fp.assign_fingerprints(
            [Speaker(speaker_id="OTHER", embedding=[0.0, 1.0, 0.0])],
            persist=False,
            speaker_durations_ms={"OTHER": 30_000},
        )
        plan2.commit()
        # Cross-channel Max: ~0.70 to canonical, ~0.0 to OTHER. Below 0.78 but a
        # dominant margin -> must MATCH the canonical print, not mint a duplicate.
        cross = [0.70, 0.0, 0.714]  # cosine to [1,0,0] ~= 0.70
        matched, plan3 = fp.assign_fingerprints(
            [Speaker(speaker_id="MAX_LOOM", embedding=cross)],
            persist=False,
            speaker_durations_ms={"MAX_LOOM": 30_000},
        )
        plan3.commit()
        assert matched[0].fingerprint_id == canonical_id
        # Still exactly 2 prints, no duplicate minted.
        assert len(fp.list_all()) == 2
    finally:
        store.close()


def test_quality_gate_blocks_short_segment_enrollment(tmp_path):
    db = tmp_path / "undertone.db"
    store = TranscriptStore(db)
    try:
        fp = SpeakerFingerprintStore(db)
        speakers, plan = fp.assign_fingerprints(
            [Speaker(speaker_id="BLIP", embedding=[1.0, 0.0])],
            persist=False,
            speaker_durations_ms={"BLIP": 2_000},  # < 15s enroll gate
        )
        plan.commit()
        assert speakers[0].fingerprint_id is None
        assert fp.list_all() == []
    finally:
        store.close()


def test_margin_match_does_not_fold_into_centroid(tmp_path):
    # Anti-drift: a margin (cross-channel) match gets the label but must not be
    # folded into the canonical centroid (that is how the magnet formed).
    db = tmp_path / "undertone.db"
    store = TranscriptStore(db)
    try:
        fp = SpeakerFingerprintStore(db)
        enrolled, plan = fp.assign_fingerprints(
            [Speaker(speaker_id="A", embedding=[1.0, 0.0, 0.0])],
            persist=False,
            speaker_durations_ms={"A": 30_000},
        )
        plan.commit()
        _, plan2 = fp.assign_fingerprints(
            [Speaker(speaker_id="B", embedding=[0.0, 1.0, 0.0])],
            persist=False,
            speaker_durations_ms={"B": 30_000},
        )
        plan2.commit()
        canonical_id = enrolled[0].fingerprint_id
        # Margin match (0.70, long) -> labeled but NOT folded: sample_count stays 1.
        _, plan3 = fp.assign_fingerprints(
            [Speaker(speaker_id="A2", embedding=[0.70, 0.0, 0.714])],
            persist=False,
            speaker_durations_ms={"A2": 30_000},
        )
        plan3.commit()
        row = store._conn.execute(
            "SELECT sample_count FROM speaker_fingerprints WHERE fingerprint_id = ?",
            (canonical_id,),
        ).fetchone()
        assert row["sample_count"] == 1
    finally:
        store.close()
