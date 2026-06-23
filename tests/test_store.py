import sqlite3

import pytest

from undertone_audio import EnrichedTranscript, Segment, Speaker, TranscriptMetadata
from undertone_audio.storage import TranscriptStore


def test_store_roundtrips_raw_transcript_without_attribution_tables(tmp_path):
    db = tmp_path / "undertone.db"
    store = TranscriptStore(db)
    try:
        transcript = EnrichedTranscript(
            transcript_id="meeting-1",
            metadata=TranscriptMetadata(
                source_path="audio.wav",
                video_path="video.webm",
                duration_ms=1000,
                engine="test",
                diarization_state="ok",
                source_metadata={"title": "raw meeting"},
            ),
            speakers=[Speaker(speaker_id="S1", fingerprint_id="VP-1", embedding=[0.1, 0.2])],
            segments=[Segment(segment_id="seg1", speaker_id="S1", start_ms=0, end_ms=1000, text="hello")],
        )
        store.save(transcript)

        loaded = store.load("meeting-1")
        assert loaded is not None
        assert transcript.store_ref == f"sqlite:{db.resolve()}#meeting-1"
        assert loaded.store_ref == f"sqlite:{db.resolve()}#meeting-1"
        assert loaded.metadata.video_path == "video.webm"
        assert loaded.metadata.source_metadata == {"title": "raw meeting"}
        assert loaded.speakers[0].fingerprint_id == "VP-1"
        assert loaded.speakers[0].embedding == [0.1, 0.2]
        assert loaded.segments[0].text == "hello"
        assert store.search("hello")[0][:3] == ("meeting-1", "seg1", "S1")

        table_names = {
            row[0]
            for row in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }
        assert "transcript_attendees" not in table_names
        assert "transcript_projects" not in table_names
        assert "speaker_resolution" not in table_names
        columns = {
            row[1]
            for row in store._conn.execute("PRAGMA table_info(transcripts)")
        }
        assert "scope" not in columns
        assert "scope_source" not in columns
    finally:
        store.close()


def test_store_rejects_segments_for_unknown_speakers(tmp_path):
    store = TranscriptStore(tmp_path / "undertone.db")
    try:
        with pytest.raises(ValueError, match="unknown speakers"):
            store.save(
                EnrichedTranscript(
                    transcript_id="bad",
                    metadata=TranscriptMetadata(duration_ms=1000, engine="test"),
                    speakers=[Speaker(speaker_id="S1")],
                    segments=[
                        Segment(
                            segment_id="seg1",
                            speaker_id="S2",
                            start_ms=0,
                            end_ms=1000,
                            text="bad",
                        )
                    ],
                )
            )
        assert store.load("bad") is None
    finally:
        store.close()


def test_fresh_database_has_only_core_tables(tmp_path):
    db = tmp_path / "undertone.db"
    store = TranscriptStore(db)
    store.close()

    with sqlite3.connect(db) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
    assert tables == {
        "schema_migrations",
        "transcripts",
        "speakers",
        "speaker_fingerprints",
        "segments",
        "speaker_metrics",
        "segments_fts",
        "segments_fts_data",
        "segments_fts_idx",
        "segments_fts_content",
        "segments_fts_docsize",
        "segments_fts_config",
    }


def test_existing_skeleton_database_repairs_added_audio_tables(tmp_path):
    db = tmp_path / "old-undertone.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO schema_migrations(version) VALUES ('001_core_initial.sql');
            CREATE TABLE transcripts (
                transcript_id TEXT PRIMARY KEY,
                source_path TEXT,
                source_url TEXT,
                video_path TEXT,
                duration_ms INTEGER NOT NULL,
                language TEXT NOT NULL DEFAULT 'en',
                meeting_type TEXT NOT NULL DEFAULT 'unknown',
                meeting_type_confidence REAL,
                recorded_at TEXT,
                engine TEXT NOT NULL,
                pipeline_version TEXT NOT NULL,
                schema_version TEXT NOT NULL DEFAULT '1',
                expected_speaker_count INTEGER,
                expected_speaker_source TEXT,
                source_metadata TEXT,
                diarization_state TEXT,
                diarization_error_code TEXT,
                diarization_error_detail TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE speakers (
                transcript_id TEXT NOT NULL,
                speaker_id TEXT NOT NULL,
                fingerprint_id TEXT,
                display_name TEXT,
                embedding TEXT,
                PRIMARY KEY (transcript_id, speaker_id)
            );
            CREATE TABLE segments (
                segment_id TEXT NOT NULL,
                transcript_id TEXT NOT NULL,
                speaker_id TEXT NOT NULL,
                start_ms INTEGER NOT NULL,
                end_ms INTEGER NOT NULL,
                text TEXT NOT NULL,
                sentiment TEXT,
                sentiment_confidence REAL,
                tone_tags TEXT NOT NULL DEFAULT '[]',
                is_interruption INTEGER NOT NULL DEFAULT 0,
                overlap_with_prev_ms INTEGER NOT NULL DEFAULT 0,
                gap_before_ms INTEGER NOT NULL DEFAULT 0,
                fillers TEXT NOT NULL DEFAULT '[]',
                linguistic TEXT,
                words TEXT NOT NULL DEFAULT '[]',
                PRIMARY KEY (transcript_id, segment_id)
            );
            CREATE TABLE speaker_metrics (
                transcript_id TEXT NOT NULL,
                speaker_id TEXT NOT NULL,
                talk_time_ms INTEGER NOT NULL,
                talk_ratio REAL NOT NULL,
                word_count INTEGER NOT NULL,
                wpm REAL NOT NULL,
                articulation_rate REAL,
                pause_count INTEGER NOT NULL DEFAULT 0,
                avg_pause_ms REAL NOT NULL DEFAULT 0,
                f0_mean_hz REAL,
                f0_stdev_hz REAL,
                jitter_local REAL,
                shimmer_local REAL,
                voiced_duration_s REAL,
                filler_count INTEGER NOT NULL DEFAULT 0,
                filler_rate REAL NOT NULL DEFAULT 0,
                interruptions_made INTEGER NOT NULL DEFAULT 0,
                interruptions_received INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (transcript_id, speaker_id)
            );
            """
        )

    store = TranscriptStore(db)
    try:
        tables = {
            row[0]
            for row in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }
        columns = {row[1] for row in store._conn.execute("PRAGMA table_info(transcripts)")}
        assert "speaker_fingerprints" in tables
        assert "segments_fts" in tables
        assert "asr_backend" in columns
        assert "audio_format" in columns
    finally:
        store.close()


def test_store_creates_parent_directory(tmp_path):
    db = tmp_path / "nested" / "state" / "undertone.db"
    store = TranscriptStore(db)
    store.close()

    assert db.exists()
