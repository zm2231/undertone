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
