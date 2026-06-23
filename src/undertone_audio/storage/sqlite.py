from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from undertone_audio.schema import (
    EnrichedTranscript,
    LinguisticFeatures,
    Segment,
    SegmentEnrichment,
    Sentiment,
    Speaker,
    SpeakerVoiceMetrics,
    TranscriptMetadata,
    Word,
)
from undertone_audio.privacy import sanitize_source_metadata
from undertone_audio.diarization.fingerprint import FingerprintAssignmentPlan, SpeakerFingerprintStore

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


class TranscriptStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    def _migrate(self) -> None:
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version TEXT PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        self._conn.commit()
        applied = {
            row[0] for row in self._conn.execute("SELECT version FROM schema_migrations")
        }
        for sql_file in sorted(MIGRATIONS_DIR.glob("*.sql")):
            version = sql_file.name
            if version in applied:
                continue
            try:
                self._conn.execute("BEGIN")
                self._conn.executescript(sql_file.read_text())
                self._conn.execute(
                    "INSERT INTO schema_migrations(version) VALUES (?)",
                    (version,),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        self._ensure_transcript_columns()
        self._ensure_core_tables()

    def _ensure_transcript_columns(self) -> None:
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(transcripts)")}
        columns = {
            "asr_backend": "TEXT",
            "diarization_backend": "TEXT",
            "vad_backend": "TEXT",
            "embedding_backend": "TEXT",
            "fingerprint_backend": "TEXT",
            "model_versions": "TEXT",
            "audio_format": "TEXT",
            "raw_transcript_json": "TEXT",
        }
        for name, definition in columns.items():
            if name not in existing:
                self._conn.execute(f"ALTER TABLE transcripts ADD COLUMN {name} {definition}")
        self._conn.commit()

    def _ensure_core_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS speaker_fingerprints (
                fingerprint_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                display_name TEXT,
                sample_count INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
                text,
                transcript_id UNINDEXED,
                segment_id UNINDEXED,
                speaker_id UNINDEXED
            );

            CREATE INDEX IF NOT EXISTS idx_speakers_fingerprint ON speakers(fingerprint_id);
            CREATE INDEX IF NOT EXISTS idx_segments_transcript ON segments(transcript_id);
            CREATE INDEX IF NOT EXISTS idx_segments_speaker ON segments(transcript_id, speaker_id);
            CREATE INDEX IF NOT EXISTS idx_segments_transcript_start
                ON segments(transcript_id, start_ms);
            """
        )
        self._conn.commit()

    def save(self, transcript: EnrichedTranscript) -> None:
        self.save_with_fingerprint_plan(transcript, None)

    def exists(self, transcript_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM transcripts WHERE transcript_id = ?",
            (transcript_id,),
        ).fetchone()
        return row is not None

    def save_with_fingerprint_plan(
        self,
        transcript: EnrichedTranscript,
        fingerprint_plan: FingerprintAssignmentPlan | None,
        fingerprint_store: SpeakerFingerprintStore | None = None,
        raw_transcript=None,
    ) -> None:
        transcript.store_ref = self._store_ref(transcript.transcript_id)
        m = transcript.metadata
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            self._save_within_transaction(transcript, m, raw_transcript)
            if fingerprint_plan is not None:
                store = fingerprint_store or SpeakerFingerprintStore(self.db_path)
                store.apply_plan_on_conn(self._conn, fingerprint_plan)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _save_within_transaction(
        self,
        transcript: EnrichedTranscript,
        m: TranscriptMetadata,
        raw_transcript=None,
    ) -> None:
        c = self._conn.cursor()
        c.execute(
            """INSERT INTO transcripts
               (transcript_id, source_path, source_url, video_path, duration_ms, language,
                meeting_type, meeting_type_confidence, recorded_at, engine,
                asr_backend, diarization_backend, vad_backend, embedding_backend,
                fingerprint_backend, model_versions, audio_format,
                raw_transcript_json, pipeline_version, schema_version, expected_speaker_count,
                expected_speaker_source, source_metadata, diarization_state,
                diarization_error_code, diarization_error_detail)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(transcript_id) DO UPDATE SET
                 source_path = excluded.source_path,
                 source_url = excluded.source_url,
                 video_path = excluded.video_path,
                 duration_ms = excluded.duration_ms,
                 language = excluded.language,
                 meeting_type = excluded.meeting_type,
                 meeting_type_confidence = excluded.meeting_type_confidence,
                 recorded_at = excluded.recorded_at,
                 engine = excluded.engine,
                 asr_backend = excluded.asr_backend,
                 diarization_backend = excluded.diarization_backend,
                 vad_backend = excluded.vad_backend,
                 embedding_backend = excluded.embedding_backend,
                 fingerprint_backend = excluded.fingerprint_backend,
                 model_versions = excluded.model_versions,
                 audio_format = excluded.audio_format,
                 raw_transcript_json = excluded.raw_transcript_json,
                 pipeline_version = excluded.pipeline_version,
                 schema_version = excluded.schema_version,
                 expected_speaker_count = excluded.expected_speaker_count,
                 expected_speaker_source = excluded.expected_speaker_source,
                 source_metadata = excluded.source_metadata,
                 diarization_state = excluded.diarization_state,
                 diarization_error_code = excluded.diarization_error_code,
                 diarization_error_detail = excluded.diarization_error_detail""",
            (
                transcript.transcript_id,
                m.source_path,
                m.source_url,
                m.video_path,
                m.duration_ms,
                m.language,
                m.meeting_type.value,
                m.meeting_type_confidence,
                m.recorded_at.isoformat() if m.recorded_at else None,
                m.engine,
                m.asr_backend,
                m.diarization_backend,
                m.vad_backend,
                m.embedding_backend,
                m.fingerprint_backend,
                json.dumps(m.model_versions) if m.model_versions else None,
                json.dumps(m.audio_format) if m.audio_format else None,
                raw_transcript.model_dump_json() if raw_transcript is not None else None,
                m.pipeline_version,
                transcript.schema_version,
                m.expected_speaker_count,
                m.expected_speaker_source,
                json.dumps(sanitize_source_metadata(m.source_metadata)) if m.source_metadata else None,
                m.diarization_state,
                m.diarization_error_code,
                m.diarization_error_detail,
            ),
        )

        speaker_ids = {speaker.speaker_id for speaker in transcript.speakers}
        orphan_segments = [
            segment for segment in transcript.segments if segment.speaker_id not in speaker_ids
        ]
        if orphan_segments:
            sample = sorted({segment.speaker_id for segment in orphan_segments})[:5]
            raise ValueError(
                f"transcript {transcript.transcript_id} has {len(orphan_segments)} "
                f"segments referencing unknown speakers {sample}"
            )

        c.execute("DELETE FROM speakers WHERE transcript_id = ?", (transcript.transcript_id,))
        c.executemany(
            """INSERT INTO speakers
               (transcript_id, speaker_id, fingerprint_id, display_name, embedding)
               VALUES (?, ?, ?, ?, ?)""",
            [
                (
                    transcript.transcript_id,
                    speaker.speaker_id,
                    speaker.fingerprint_id,
                    speaker.display_name,
                    json.dumps(speaker.embedding) if speaker.embedding else None,
                )
                for speaker in transcript.speakers
            ],
        )

        c.execute("DELETE FROM segments WHERE transcript_id = ?", (transcript.transcript_id,))
        c.executemany(
            """INSERT INTO segments
               (segment_id, transcript_id, speaker_id, start_ms, end_ms, text,
                sentiment, sentiment_confidence, tone_tags, is_interruption,
                overlap_with_prev_ms, gap_before_ms, fillers, linguistic, words)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [self._segment_row(transcript.transcript_id, segment) for segment in transcript.segments],
        )

        c.execute("DELETE FROM segments_fts WHERE transcript_id = ?", (transcript.transcript_id,))
        c.executemany(
            "INSERT INTO segments_fts (text, transcript_id, segment_id, speaker_id) VALUES (?, ?, ?, ?)",
            [
                (segment.text, transcript.transcript_id, segment.segment_id, segment.speaker_id)
                for segment in transcript.segments
            ],
        )

        c.execute("DELETE FROM speaker_metrics WHERE transcript_id = ?", (transcript.transcript_id,))
        c.executemany(
            """INSERT INTO speaker_metrics
               (transcript_id, speaker_id, talk_time_ms, talk_ratio, word_count, wpm,
                articulation_rate, pause_count, avg_pause_ms, f0_mean_hz, f0_stdev_hz,
                jitter_local, shimmer_local, voiced_duration_s, filler_count, filler_rate,
                interruptions_made, interruptions_received)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    transcript.transcript_id,
                    metric.speaker_id,
                    metric.talk_time_ms,
                    metric.talk_ratio,
                    metric.word_count,
                    metric.wpm,
                    metric.articulation_rate,
                    metric.pause_count,
                    metric.avg_pause_ms,
                    metric.f0_mean_hz,
                    metric.f0_stdev_hz,
                    metric.jitter_local,
                    metric.shimmer_local,
                    metric.voiced_duration_s,
                    metric.filler_count,
                    metric.filler_rate,
                    metric.interruptions_made,
                    metric.interruptions_received,
                )
                for metric in transcript.speaker_metrics
            ],
        )

    def _segment_row(self, transcript_id: str, segment: Segment) -> tuple:
        enrichment = segment.enrichment
        return (
            segment.segment_id,
            transcript_id,
            segment.speaker_id,
            segment.start_ms,
            segment.end_ms,
            segment.text,
            enrichment.sentiment.value if enrichment.sentiment else None,
            enrichment.sentiment_confidence,
            json.dumps(enrichment.tone_tags),
            int(enrichment.is_interruption),
            enrichment.overlap_with_prev_ms,
            enrichment.gap_before_ms,
            json.dumps(enrichment.fillers),
            enrichment.linguistic.model_dump_json() if enrichment.linguistic else None,
            json.dumps([word.model_dump() for word in segment.words]),
        )

    def load(self, transcript_id: str) -> EnrichedTranscript | None:
        c = self._conn.cursor()
        t_row = c.execute(
            "SELECT * FROM transcripts WHERE transcript_id = ?",
            (transcript_id,),
        ).fetchone()
        if t_row is None:
            return None

        speakers = [
            Speaker(
                speaker_id=row["speaker_id"],
                fingerprint_id=row["fingerprint_id"],
                display_name=row["display_name"],
                embedding=json.loads(row["embedding"]) if row["embedding"] else None,
            )
            for row in c.execute(
                "SELECT * FROM speakers WHERE transcript_id = ?",
                (transcript_id,),
            )
        ]
        segments = [
            self._row_to_segment(row)
            for row in c.execute(
                "SELECT * FROM segments WHERE transcript_id = ? ORDER BY start_ms",
                (transcript_id,),
            )
        ]
        metrics = [
            SpeakerVoiceMetrics(**{key: row[key] for key in row.keys() if key != "transcript_id"})
            for row in c.execute(
                "SELECT * FROM speaker_metrics WHERE transcript_id = ?",
                (transcript_id,),
            )
        ]
        metadata = TranscriptMetadata(
            source_path=t_row["source_path"],
            source_url=t_row["source_url"],
            video_path=t_row["video_path"],
            duration_ms=t_row["duration_ms"],
            language=t_row["language"],
            meeting_type=t_row["meeting_type"],
            meeting_type_confidence=t_row["meeting_type_confidence"],
            recorded_at=datetime.fromisoformat(t_row["recorded_at"])
            if t_row["recorded_at"]
            else None,
            engine=t_row["engine"],
            asr_backend=t_row["asr_backend"],
            diarization_backend=t_row["diarization_backend"],
            vad_backend=t_row["vad_backend"],
            embedding_backend=t_row["embedding_backend"],
            fingerprint_backend=t_row["fingerprint_backend"],
            model_versions=json.loads(t_row["model_versions"]) if t_row["model_versions"] else {},
            audio_format=json.loads(t_row["audio_format"]) if t_row["audio_format"] else {},
            pipeline_version=t_row["pipeline_version"],
            expected_speaker_count=t_row["expected_speaker_count"],
            expected_speaker_source=t_row["expected_speaker_source"],
            source_metadata=json.loads(t_row["source_metadata"]) if t_row["source_metadata"] else {},
            diarization_state=t_row["diarization_state"],
            diarization_error_code=t_row["diarization_error_code"],
            diarization_error_detail=t_row["diarization_error_detail"],
        )
        return EnrichedTranscript(
            transcript_id=transcript_id,
            store_ref=self._store_ref(transcript_id),
            metadata=metadata,
            speakers=speakers,
            segments=segments,
            speaker_metrics=metrics,
        )

    def _store_ref(self, transcript_id: str) -> str:
        return f"sqlite:{self.db_path.expanduser().resolve()}#{transcript_id}"

    def load_raw(self, transcript_id: str):
        from undertone_audio.engines.base import RawTranscript

        row = self._conn.execute(
            "SELECT raw_transcript_json FROM transcripts WHERE transcript_id = ?",
            (transcript_id,),
        ).fetchone()
        if row is None or not row["raw_transcript_json"]:
            return None
        return RawTranscript.model_validate_json(row["raw_transcript_json"])

    def list_transcripts(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        source: str | None = None,
        meeting_type: str | None = None,
        diarization_state: str | None = None,
    ) -> list[dict]:
        where = []
        params: list[object] = []
        if source:
            where.append("(source_path LIKE ? OR source_url LIKE ?)")
            params.extend([f"%{source}%", f"%{source}%"])
        if meeting_type:
            where.append("meeting_type = ?")
            params.append(meeting_type)
        if diarization_state:
            where.append("diarization_state = ?")
            params.append(diarization_state)
        sql = """SELECT t.transcript_id, t.source_path, t.source_url, t.duration_ms, t.language,
                        t.meeting_type, t.recorded_at, t.engine, t.diarization_state,
                        COUNT(DISTINCT s.speaker_id) AS speaker_count,
                        COUNT(DISTINCT g.segment_id) AS segment_count
                 FROM transcripts t
                 LEFT JOIN speakers s ON s.transcript_id = t.transcript_id
                 LEFT JOIN segments g ON g.transcript_id = t.transcript_id"""
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " GROUP BY t.transcript_id ORDER BY COALESCE(t.recorded_at, t.created_at) DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        return [dict(row) for row in self._conn.execute(sql, params)]

    def stats(self) -> dict:
        row = self._conn.execute(
            """SELECT COUNT(*) AS transcript_count,
                      COALESCE(SUM(duration_ms), 0) AS total_duration_ms
               FROM transcripts"""
        ).fetchone()
        speakers = self._conn.execute("SELECT COUNT(*) AS count FROM speakers").fetchone()["count"]
        segments = self._conn.execute("SELECT COUNT(*) AS count FROM segments").fetchone()["count"]
        fingerprints = self._conn.execute(
            "SELECT COUNT(*) AS count FROM speaker_fingerprints"
        ).fetchone()["count"]
        return {
            "db_path": str(self.db_path),
            "transcript_count": row["transcript_count"],
            "total_duration_ms": row["total_duration_ms"],
            "total_duration_minutes": round(row["total_duration_ms"] / 60000, 3),
            "speaker_count": speakers,
            "segment_count": segments,
            "fingerprint_count": fingerprints,
        }

    def fingerprint_excerpts(
        self,
        *,
        fingerprint_id: str | None = None,
        unnamed_only: bool = False,
        limit_per_fingerprint: int = 3,
    ) -> dict[str, list[dict]]:
        where = ["s.fingerprint_id IS NOT NULL"]
        params: list[object] = []
        if fingerprint_id:
            where.append("s.fingerprint_id = ?")
            params.append(fingerprint_id)
        if unnamed_only:
            where.append("(f.display_name IS NULL OR f.display_name = '')")
        sql = f"""SELECT s.fingerprint_id, g.transcript_id, g.segment_id, g.speaker_id,
                         g.start_ms, g.end_ms, g.text
                  FROM speakers s
                  JOIN segments g
                    ON g.transcript_id = s.transcript_id
                   AND g.speaker_id = s.speaker_id
                  LEFT JOIN speaker_fingerprints f
                    ON f.fingerprint_id = s.fingerprint_id
                  WHERE {' AND '.join(where)}
                  ORDER BY s.fingerprint_id, g.transcript_id, g.start_ms"""
        excerpts: dict[str, list[dict]] = {}
        for row in self._conn.execute(sql, params):
            bucket = excerpts.setdefault(row["fingerprint_id"], [])
            if len(bucket) >= limit_per_fingerprint:
                continue
            bucket.append(
                {
                    "transcript_id": row["transcript_id"],
                    "segment_id": row["segment_id"],
                    "speaker_id": row["speaker_id"],
                    "start_ms": row["start_ms"],
                    "end_ms": row["end_ms"],
                    "text": row["text"],
                }
            )
        return excerpts

    def delete(self, transcript_id: str) -> bool:
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            existed = self.exists(transcript_id)
            if not existed:
                self._conn.rollback()
                return False
            self._conn.execute("DELETE FROM speaker_metrics WHERE transcript_id = ?", (transcript_id,))
            self._conn.execute("DELETE FROM segments_fts WHERE transcript_id = ?", (transcript_id,))
            self._conn.execute("DELETE FROM segments WHERE transcript_id = ?", (transcript_id,))
            self._conn.execute("DELETE FROM speakers WHERE transcript_id = ?", (transcript_id,))
            self._conn.execute("DELETE FROM transcripts WHERE transcript_id = ?", (transcript_id,))
            self._conn.commit()
            return True
        except Exception:
            self._conn.rollback()
            raise

    def _row_to_segment(self, row: sqlite3.Row) -> Segment:
        linguistic = (
            LinguisticFeatures.model_validate_json(row["linguistic"])
            if row["linguistic"]
            else None
        )
        return Segment(
            segment_id=row["segment_id"],
            speaker_id=row["speaker_id"],
            start_ms=row["start_ms"],
            end_ms=row["end_ms"],
            text=row["text"],
            words=[Word(**word) for word in json.loads(row["words"] or "[]")],
            enrichment=SegmentEnrichment(
                sentiment=Sentiment(row["sentiment"]) if row["sentiment"] else None,
                sentiment_confidence=row["sentiment_confidence"],
                tone_tags=json.loads(row["tone_tags"] or "[]"),
                is_interruption=bool(row["is_interruption"]),
                overlap_with_prev_ms=row["overlap_with_prev_ms"],
                gap_before_ms=row["gap_before_ms"],
                fillers=json.loads(row["fillers"] or "[]"),
                linguistic=linguistic,
            ),
        )

    def search(self, query: str, limit: int = 20) -> list[tuple[str, str, str, str]]:
        rows = self._conn.execute(
            """SELECT transcript_id, segment_id, speaker_id,
                      snippet(segments_fts, 0, '<<', '>>', '...', 16) AS snippet
               FROM segments_fts WHERE segments_fts MATCH ? LIMIT ?""",
            (query, limit),
        )
        return [
            (row["transcript_id"], row["segment_id"], row["speaker_id"], row["snippet"])
            for row in rows
        ]

    def close(self) -> None:
        self._conn.close()
