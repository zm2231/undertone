CREATE TABLE IF NOT EXISTS transcripts (
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
    asr_backend TEXT,
    diarization_backend TEXT,
    vad_backend TEXT,
    embedding_backend TEXT,
    fingerprint_backend TEXT,
    model_versions TEXT,
    audio_format TEXT,
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

CREATE INDEX IF NOT EXISTS idx_transcripts_recorded_at
    ON transcripts(recorded_at);

CREATE TABLE IF NOT EXISTS speakers (
    transcript_id TEXT NOT NULL REFERENCES transcripts(transcript_id) ON DELETE CASCADE,
    speaker_id TEXT NOT NULL,
    fingerprint_id TEXT,
    display_name TEXT,
    embedding TEXT,
    PRIMARY KEY (transcript_id, speaker_id)
);

CREATE INDEX IF NOT EXISTS idx_speakers_fingerprint ON speakers(fingerprint_id);

CREATE TABLE IF NOT EXISTS speaker_fingerprints (
    fingerprint_id TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    display_name TEXT,
    sample_count INTEGER NOT NULL DEFAULT 1,
    embedding_model TEXT,
    embedding_dimension INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fingerprint_sources (
    fingerprint_id TEXT NOT NULL REFERENCES speaker_fingerprints(fingerprint_id) ON DELETE CASCADE,
    transcript_id TEXT NOT NULL REFERENCES transcripts(transcript_id) ON DELETE CASCADE,
    speaker_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (fingerprint_id, transcript_id, speaker_id)
);

CREATE INDEX IF NOT EXISTS idx_fingerprint_sources_transcript
    ON fingerprint_sources(transcript_id, speaker_id);

CREATE TABLE IF NOT EXISTS segments (
    segment_id TEXT NOT NULL,
    transcript_id TEXT NOT NULL REFERENCES transcripts(transcript_id) ON DELETE CASCADE,
    speaker_id TEXT NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    text TEXT NOT NULL,
    asr_confidence REAL,
    diarization_quality REAL,
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

CREATE INDEX IF NOT EXISTS idx_segments_transcript ON segments(transcript_id);
CREATE INDEX IF NOT EXISTS idx_segments_speaker ON segments(transcript_id, speaker_id);
CREATE INDEX IF NOT EXISTS idx_segments_transcript_start ON segments(transcript_id, start_ms);

CREATE TABLE IF NOT EXISTS speaker_metrics (
    transcript_id TEXT NOT NULL REFERENCES transcripts(transcript_id) ON DELETE CASCADE,
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

CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
    text,
    transcript_id UNINDEXED,
    segment_id UNINDEXED,
    speaker_id UNINDEXED
);
