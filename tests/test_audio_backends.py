from undertone_audio.diarization.fingerprint import SpeakerFingerprintStore
from undertone_audio.diarization.merge import merge_adjacent_turns
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
from undertone_audio.schema import EnrichedTranscript, Segment, Speaker, TranscriptMetadata
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
    assert raw.segments[0].diarization_quality is None


def test_fluidaudio_cli_merge_preserves_process_quality_score():
    raw = merge_transcribe_and_diarize(
        {
            "wordTimings": [
                {"word": "quality", "startTime": 0.1, "endTime": 0.3, "confidence": 0.8},
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
                    "qualityScore": 0.72,
                }
            ],
        },
    )

    assert raw.segments[0].diarization_quality == 0.72


def test_merge_adjacent_turns_aggregates_diarization_quality_by_duration():
    merged = merge_adjacent_turns(
        [
            Segment(
                segment_id="s1",
                speaker_id="S1",
                start_ms=0,
                end_ms=1000,
                text="low",
                diarization_quality=0.1,
            ),
            Segment(
                segment_id="s2",
                speaker_id="S1",
                start_ms=1000,
                end_ms=3000,
                text="high",
                diarization_quality=0.9,
            ),
        ],
        gap_threshold_ms=0,
    )

    assert len(merged) == 1
    assert merged[0].text == "low high"
    assert merged[0].diarization_quality == (0.1 * 1000 + 0.9 * 2000) / 3000


def test_merge_adjacent_turns_clears_diarization_quality_when_partial():
    merged = merge_adjacent_turns(
        [
            Segment(
                segment_id="s1",
                speaker_id="S1",
                start_ms=0,
                end_ms=1000,
                text="known",
                diarization_quality=0.8,
            ),
            Segment(
                segment_id="s2",
                speaker_id="S1",
                start_ms=1000,
                end_ms=2000,
                text="unknown",
                diarization_quality=None,
            ),
        ],
        gap_threshold_ms=0,
    )

    assert len(merged) == 1
    assert merged[0].diarization_quality is None


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
                    "qualityScore": 0.25,
                },
                {
                    "speakerId": "S2",
                    "startTimeSeconds": 1.0,
                    "endTimeSeconds": 2.0,
                    "embedding": [0.0, 1.0],
                    "qualityScore": 0.75,
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
    assert [segment.diarization_quality for segment in raw.segments] == [0.25, 0.75]


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
        assert speakers[0].match is not None
        assert speakers[0].match.kind == "new"
        assert speakers[0].match.embedding_model is None
        assert fingerprints.list_all() == []
        plan.commit()

        matched, plan = fingerprints.assign_fingerprints(
            [Speaker(speaker_id="S2", embedding=[0.99, 0.01])],
            persist=False,
        )
        assert matched[0].fingerprint_id == speakers[0].fingerprint_id
        assert matched[0].match is not None
        assert matched[0].match.kind == "strong"
        assert matched[0].match.similarity > 0.99
        assert matched[0].match.second_similarity is None
        assert matched[0].match.margin is None
        assert matched[0].match.similarity_threshold == 0.95
        plan.commit()

        row = store._conn.execute(
            "SELECT sample_count FROM speaker_fingerprints WHERE fingerprint_id = ?",
            (speakers[0].fingerprint_id,),
        ).fetchone()
        assert row["sample_count"] == 2
    finally:
        store.close()


def test_fingerprint_match_kinds_cover_non_acoustic_branches(tmp_path):
    db = tmp_path / "undertone.db"
    store = TranscriptStore(db)
    try:
        fingerprints = SpeakerFingerprintStore(db, embedding_model="fixture-model")
        enrolled, plan = fingerprints.assign_fingerprints(
            [Speaker(speaker_id="S1", embedding=[1.0, 0.0], display_name="Alex")],
            persist=False,
            speaker_durations_ms={"S1": 16000},
        )
        plan.commit()
        fingerprint_id = enrolled[0].fingerprint_id

        preassigned, _plan = fingerprints.assign_fingerprints(
            [Speaker(speaker_id="S2", fingerprint_id=fingerprint_id)],
            persist=False,
        )
        assert preassigned[0].match is not None
        assert preassigned[0].match.kind == "preassigned"
        assert preassigned[0].match.similarity is None

        name_matched, _plan = fingerprints.assign_fingerprints(
            [Speaker(speaker_id="S3", display_name="alex")],
            persist=False,
        )
        assert name_matched[0].fingerprint_id == fingerprint_id
        assert name_matched[0].match is not None
        assert name_matched[0].match.kind == "name_match"
        assert name_matched[0].match.similarity is None

        no_embedding, _plan = fingerprints.assign_fingerprints(
            [Speaker(speaker_id="S4")],
            persist=False,
        )
        assert no_embedding[0].fingerprint_id is None
        assert no_embedding[0].match is not None
        assert no_embedding[0].match.kind == "no_embedding"
    finally:
        store.close()


def test_fingerprint_update_mean_is_model_guarded(tmp_path):
    db = tmp_path / "undertone.db"
    store = TranscriptStore(db)
    try:
        model_a = SpeakerFingerprintStore(db, similarity_threshold=0.95, embedding_model="model-a")
        speakers, plan = model_a.assign_fingerprints(
            [Speaker(speaker_id="S1", embedding=[1.0, 0.0])],
            persist=False,
            speaker_durations_ms={"S1": 16000},
        )
        fingerprint_id = speakers[0].fingerprint_id
        plan.commit()

        model_b = SpeakerFingerprintStore(db, similarity_threshold=0.0, embedding_model="model-b")
        model_b._update_mean(store._conn, fingerprint_id, [0.0, 1.0])

        row = store._conn.execute(
            "SELECT sample_count, embedding_model FROM speaker_fingerprints WHERE fingerprint_id = ?",
            (fingerprint_id,),
        ).fetchone()
        assert row["sample_count"] == 1
        assert row["embedding_model"] == "model-a"
    finally:
        store.close()


def test_discarded_fingerprint_is_not_matched_or_updated(tmp_path):
    db = tmp_path / "undertone.db"
    store = TranscriptStore(db)
    try:
        fingerprints = SpeakerFingerprintStore(db, similarity_threshold=0.95)
        speakers, plan = fingerprints.assign_fingerprints(
            [Speaker(speaker_id="S1", embedding=[1.0, 0.0])],
            persist=False,
            speaker_durations_ms={"S1": 16000},
        )
        original_id = speakers[0].fingerprint_id
        plan.commit()

        store.discard_fingerprint(original_id, "mixed speaker")
        matched, next_plan = fingerprints.assign_fingerprints(
            [Speaker(speaker_id="S2", embedding=[1.0, 0.0])],
            persist=False,
            speaker_durations_ms={"S2": 16000},
        )
        next_plan.commit()

        assert matched[0].fingerprint_id is not None
        assert matched[0].fingerprint_id != original_id
        rows = store._conn.execute(
            "SELECT fingerprint_id, sample_count, status FROM speaker_fingerprints ORDER BY fingerprint_id"
        ).fetchall()
        assert len(rows) == 2
        discarded = next(row for row in rows if row["fingerprint_id"] == original_id)
        assert discarded["status"] == "discarded"
        assert discarded["sample_count"] == 1
    finally:
        store.close()


def test_stale_match_plan_does_not_update_discarded_fingerprint(tmp_path):
    db = tmp_path / "undertone.db"
    store = TranscriptStore(db)
    try:
        fingerprints = SpeakerFingerprintStore(db, similarity_threshold=0.95)
        speakers, plan = fingerprints.assign_fingerprints(
            [Speaker(speaker_id="S1", embedding=[1.0, 0.0])],
            persist=False,
            speaker_durations_ms={"S1": 16000},
        )
        fingerprint_id = speakers[0].fingerprint_id
        plan.commit()

        matched, stale_plan = fingerprints.assign_fingerprints(
            [Speaker(speaker_id="S2", embedding=[0.99, 0.01])],
            persist=False,
            speaker_durations_ms={"S2": 16000},
        )
        assert matched[0].fingerprint_id == fingerprint_id

        store.discard_fingerprint(fingerprint_id, "discarded before commit")
        stale_plan.commit()

        row = store._conn.execute(
            "SELECT sample_count, status FROM speaker_fingerprints WHERE fingerprint_id = ?",
            (fingerprint_id,),
        ).fetchone()
        assert row["status"] == "discarded"
        assert row["sample_count"] == 1
    finally:
        store.close()


def test_stale_match_plan_does_not_attach_discarded_fingerprint_to_saved_transcript(tmp_path):
    db = tmp_path / "undertone.db"
    store = TranscriptStore(db)
    try:
        fingerprints = SpeakerFingerprintStore(db, similarity_threshold=0.95)
        speakers, plan = fingerprints.assign_fingerprints(
            [Speaker(speaker_id="S1", embedding=[1.0, 0.0], display_name="Old Match")],
            persist=False,
            speaker_durations_ms={"S1": 16000},
        )
        fingerprint_id = speakers[0].fingerprint_id
        plan.commit()

        matched, stale_plan = fingerprints.assign_fingerprints(
            [Speaker(speaker_id="S2", embedding=[0.99, 0.01])],
            persist=False,
            speaker_durations_ms={"S2": 16000},
        )
        assert matched[0].fingerprint_id == fingerprint_id
        stale_plan.sources.append((fingerprint_id, "stale-transcript", "S2"))

        store.discard_fingerprint(fingerprint_id, "discarded before save")
        transcript = EnrichedTranscript(
            transcript_id="stale-transcript",
            metadata=TranscriptMetadata(duration_ms=16000, engine="fixture"),
            speakers=matched,
            segments=[
                Segment(
                    segment_id="seg1",
                    speaker_id="S2",
                    start_ms=0,
                    end_ms=16000,
                    text="stale plan transcript",
                )
            ],
        )
        store.save_with_fingerprint_plan(transcript, stale_plan, fingerprints)

        fingerprint_row = store._conn.execute(
            "SELECT sample_count, status FROM speaker_fingerprints WHERE fingerprint_id = ?",
            (fingerprint_id,),
        ).fetchone()
        speaker_row = store._conn.execute(
            "SELECT fingerprint_id FROM speakers WHERE transcript_id = ? AND speaker_id = ?",
            ("stale-transcript", "S2"),
        ).fetchone()
        source_count = store._conn.execute(
            "SELECT COUNT(*) FROM fingerprint_sources WHERE fingerprint_id = ?",
            (fingerprint_id,),
        ).fetchone()[0]
        assert fingerprint_row["status"] == "discarded"
        assert fingerprint_row["sample_count"] == 1
        assert speaker_row["fingerprint_id"] is None
        assert source_count == 0
    finally:
        store.close()


def test_restored_fingerprint_matches_again(tmp_path):
    db = tmp_path / "undertone.db"
    store = TranscriptStore(db)
    try:
        fingerprints = SpeakerFingerprintStore(db, similarity_threshold=0.95)
        speakers, plan = fingerprints.assign_fingerprints(
            [Speaker(speaker_id="S1", embedding=[1.0, 0.0])],
            persist=False,
            speaker_durations_ms={"S1": 16000},
        )
        fingerprint_id = speakers[0].fingerprint_id
        plan.commit()

        store.discard_fingerprint(fingerprint_id, "test")
        store.restore_fingerprint(fingerprint_id)
        matched, plan = fingerprints.assign_fingerprints(
            [Speaker(speaker_id="S2", embedding=[0.99, 0.01])],
            persist=False,
            speaker_durations_ms={"S2": 16000},
        )
        plan.commit()

        assert matched[0].fingerprint_id == fingerprint_id
        row = store._conn.execute(
            "SELECT sample_count, status, discard_reason FROM speaker_fingerprints WHERE fingerprint_id = ?",
            (fingerprint_id,),
        ).fetchone()
        assert row["sample_count"] == 2
        assert row["status"] == "active"
        assert row["discard_reason"] is None
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
        assert matched[0].match is not None
        assert matched[0].match.kind == "margin"
        assert matched[0].match.second_similarity is not None
        assert abs(matched[0].match.second_similarity - 0.0) < 0.001
        assert matched[0].match.margin is not None
        assert matched[0].match.margin >= 0.15
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
        assert speakers[0].match is not None
        assert speakers[0].match.kind == "no_enroll"
        assert speakers[0].match.similarity is None
        assert speakers[0].match.second_similarity is None
        assert speakers[0].match.margin is None
        assert fp.list_all() == []
    finally:
        store.close()


def test_fingerprint_diagnostics_preserve_exact_cosine_floor(tmp_path):
    db = tmp_path / "undertone.db"
    store = TranscriptStore(db)
    try:
        fp = SpeakerFingerprintStore(db)
        _, plan = fp.assign_fingerprints(
            [Speaker(speaker_id="A", embedding=[1.0, 0.0])],
            persist=False,
            speaker_durations_ms={"A": 30_000},
        )
        plan.commit()

        speakers, plan = fp.assign_fingerprints(
            [Speaker(speaker_id="OPPOSITE", embedding=[-1.0, 0.0])],
            persist=False,
            speaker_durations_ms={"OPPOSITE": 2_000},
        )
        plan.commit()

        assert speakers[0].fingerprint_id is None
        assert speakers[0].match is not None
        assert speakers[0].match.kind == "no_enroll"
        assert speakers[0].match.similarity == -1.0
        assert speakers[0].match.second_similarity is None
        assert speakers[0].match.margin is None
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
