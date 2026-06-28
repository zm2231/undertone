from __future__ import annotations

import json
import math
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
from undertone_audio.diarization.fingerprint import (
    FingerprintAssignmentPlan,
    SpeakerFingerprintStore,
    _blob_to_vec,
    _vec_to_blob,
)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


class TranscriptStore:
    def __init__(self, db_path: str | Path, *, migrate: bool = True, read_only: bool = False):
        self.db_path = Path(db_path)
        if read_only:
            self._conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if migrate:
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
        self._ensure_segment_columns()
        self._ensure_core_tables()
        self._ensure_fingerprint_columns()

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

    def _ensure_segment_columns(self) -> None:
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(segments)")}
        columns = {
            "asr_confidence": "REAL",
            "diarization_quality": "REAL",
        }
        for name, definition in columns.items():
            if name not in existing:
                self._conn.execute(f"ALTER TABLE segments ADD COLUMN {name} {definition}")
        self._conn.commit()

    def _ensure_core_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS speaker_fingerprints (
                fingerprint_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                display_name TEXT,
                sample_count INTEGER NOT NULL DEFAULT 1,
                embedding_model TEXT,
                embedding_dimension INTEGER,
                status TEXT NOT NULL DEFAULT 'active',
                discard_reason TEXT,
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

            CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
                text,
                transcript_id UNINDEXED,
                segment_id UNINDEXED,
                speaker_id UNINDEXED
            );

            CREATE INDEX IF NOT EXISTS idx_speakers_fingerprint ON speakers(fingerprint_id);
            CREATE INDEX IF NOT EXISTS idx_fingerprint_sources_transcript
                ON fingerprint_sources(transcript_id, speaker_id);
            CREATE INDEX IF NOT EXISTS idx_segments_transcript ON segments(transcript_id);
            CREATE INDEX IF NOT EXISTS idx_segments_speaker ON segments(transcript_id, speaker_id);
            CREATE INDEX IF NOT EXISTS idx_segments_transcript_start
                ON segments(transcript_id, start_ms);
            """
        )
        self._conn.commit()

    def _ensure_fingerprint_columns(self) -> None:
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(speaker_fingerprints)")}
        columns = {
            "embedding_model": "TEXT",
            "embedding_dimension": "INTEGER",
            "status": "TEXT NOT NULL DEFAULT 'active'",
            "discard_reason": "TEXT",
        }
        for name, definition in columns.items():
            if name not in existing:
                self._conn.execute(f"ALTER TABLE speaker_fingerprints ADD COLUMN {name} {definition}")
        self._conn.execute(
            """UPDATE speaker_fingerprints
               SET embedding_dimension = length(embedding) / 4
               WHERE embedding_dimension IS NULL"""
        )
        self._conn.execute(
            """UPDATE speaker_fingerprints
               SET status = 'active'
               WHERE status IS NULL"""
        )
        self._conn.commit()

    def _table_columns(self, table: str) -> set[str]:
        return {row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")}

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
                self._clear_inactive_plan_speaker_refs(transcript.transcript_id, fingerprint_plan)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _clear_inactive_plan_speaker_refs(
        self,
        transcript_id: str,
        fingerprint_plan: FingerprintAssignmentPlan,
    ) -> None:
        fingerprint_ids = {fingerprint_id for fingerprint_id, _embedding in fingerprint_plan.matches}
        fingerprint_ids.update(
            fingerprint_id for fingerprint_id, _transcript_id, _speaker_id in fingerprint_plan.sources
        )
        if not fingerprint_ids:
            return
        placeholders = ",".join("?" for _ in fingerprint_ids)
        self._conn.execute(
            f"""UPDATE speakers
                SET fingerprint_id = NULL
                WHERE transcript_id = ?
                  AND fingerprint_id IN ({placeholders})
                  AND EXISTS (
                      SELECT 1 FROM speaker_fingerprints AS f
                      WHERE f.fingerprint_id = speakers.fingerprint_id
                        AND f.status = 'discarded'
                  )""",
            (transcript_id, *sorted(fingerprint_ids)),
        )

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
                 raw_transcript_json = COALESCE(excluded.raw_transcript_json, transcripts.raw_transcript_json),
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

        c.execute("DELETE FROM fingerprint_sources WHERE transcript_id = ?", (transcript.transcript_id,))
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
                asr_confidence, diarization_quality,
                sentiment, sentiment_confidence, tone_tags, is_interruption,
                overlap_with_prev_ms, gap_before_ms, fillers, linguistic, words)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            segment.asr_confidence,
            segment.diarization_quality,
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

    def relabel_speakers(self, transcript_id: str | None = None) -> dict:
        where = ""
        params: list[object] = []
        if transcript_id:
            where = "WHERE s.transcript_id = ?"
            params.append(transcript_id)
        sql = f"""SELECT s.transcript_id, s.speaker_id, s.fingerprint_id,
                         s.display_name AS old_name, f.display_name AS new_name
                  FROM speakers s
                  LEFT JOIN speaker_fingerprints f
                    ON f.fingerprint_id = s.fingerprint_id
                  {where}"""
        rows = [dict(row) for row in self._conn.execute(sql, params)]
        if transcript_id and not rows and not self.exists(transcript_id):
            raise ValueError(f"transcript not found: {transcript_id}")

        updates = [
            row
            for row in rows
            if row["fingerprint_id"]
            and row["new_name"] is not None
            and (row["old_name"] or None) != row["new_name"]
        ]
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            for row in updates:
                self._conn.execute(
                    """UPDATE speakers
                       SET display_name = ?
                       WHERE transcript_id = ? AND speaker_id = ?""",
                    (row["new_name"], row["transcript_id"], row["speaker_id"]),
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

        return {
            "transcript_id": transcript_id,
            "speakers_seen": len(rows),
            "speakers_updated": len(updates),
            "updates": updates,
        }

    def export_fingerprints(self) -> list[dict]:
        rows = []
        for row in self._conn.execute(
            """SELECT fingerprint_id, embedding, display_name, sample_count,
                      embedding_model, embedding_dimension, status, discard_reason,
                      created_at, updated_at
               FROM speaker_fingerprints ORDER BY fingerprint_id"""
        ):
            rows.append(
                {
                    "fingerprint_id": row["fingerprint_id"],
                    "embedding": _blob_to_vec(row["embedding"]),
                    "display_name": row["display_name"],
                    "sample_count": row["sample_count"],
                    "embedding_model": row["embedding_model"],
                    "embedding_dimension": row["embedding_dimension"],
                    "status": row["status"],
                    "discard_reason": row["discard_reason"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            )
        return rows

    def fingerprint_import_plan(self, fingerprints: list[dict], *, replace: bool) -> dict:
        rows = _normalize_fingerprint_rows(fingerprints)
        return self._fingerprint_import_plan_on_conn(rows, replace=replace)

    def import_fingerprints(self, fingerprints: list[dict], *, replace: bool) -> dict:
        rows = _normalize_fingerprint_rows(fingerprints)
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            plan = self._fingerprint_import_plan_on_conn(rows, replace=replace)
            self._apply_fingerprint_import_plan(plan)
            self._conn.commit()
            plan["dry_run"] = False
            return plan
        except Exception:
            self._conn.rollback()
            raise

    def _fingerprint_import_plan_on_conn(self, fingerprints: list[dict], *, replace: bool) -> dict:
        plan = {
            "operation": "fingerprint-import",
            "dry_run": True,
            "replace": replace,
            "to_insert": [],
            "to_replace": [],
            "skipped_existing": [],
        }
        existing = {
            row["fingerprint_id"]
            for row in self._conn.execute("SELECT fingerprint_id FROM speaker_fingerprints")
        }
        for row in fingerprints:
            fingerprint_id = row["fingerprint_id"]
            if fingerprint_id in existing and replace:
                plan["to_replace"].append(row)
            elif fingerprint_id in existing:
                plan["skipped_existing"].append(fingerprint_id)
            else:
                plan["to_insert"].append(row)
                existing.add(fingerprint_id)
        return plan

    def _apply_fingerprint_import_plan(self, plan: dict) -> None:
        for row in plan["to_replace"]:
            self._conn.execute(
                """UPDATE speaker_fingerprints
                   SET embedding = ?, display_name = ?, sample_count = ?,
                       embedding_model = ?, embedding_dimension = ?,
                       status = ?, discard_reason = ?,
                       created_at = COALESCE(?, created_at),
                       updated_at = COALESCE(?, CURRENT_TIMESTAMP)
                   WHERE fingerprint_id = ?""",
                (
                    _vec_to_blob(row["embedding"]),
                    row.get("display_name"),
                    row.get("sample_count", 1),
                    row.get("embedding_model"),
                    row.get("embedding_dimension"),
                    row.get("status", "active"),
                    row.get("discard_reason"),
                    row.get("created_at"),
                    row.get("updated_at"),
                    row["fingerprint_id"],
                ),
            )
            self._relabel_speakers_for_fingerprint(row["fingerprint_id"], row.get("display_name"))
        for row in plan["to_insert"]:
            self._conn.execute(
                """INSERT INTO speaker_fingerprints
                   (fingerprint_id, embedding, display_name, sample_count,
                    embedding_model, embedding_dimension, status, discard_reason,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), COALESCE(?, CURRENT_TIMESTAMP))""",
                (
                    row["fingerprint_id"],
                    _vec_to_blob(row["embedding"]),
                    row.get("display_name"),
                    row.get("sample_count", 1),
                    row.get("embedding_model"),
                    row.get("embedding_dimension"),
                    row.get("status", "active"),
                    row.get("discard_reason"),
                    row.get("created_at"),
                    row.get("updated_at"),
                ),
            )
            self._relabel_speakers_for_fingerprint(row["fingerprint_id"], row.get("display_name"))

    def _relabel_speakers_for_fingerprint(self, fingerprint_id: str, display_name: str | None) -> None:
        self._conn.execute(
            "UPDATE speakers SET display_name = ? WHERE fingerprint_id = ?",
            (display_name, fingerprint_id),
        )

    def fingerprint_merge_plan(self, source: str, target: str) -> dict:
        plan = self._fingerprint_merge_plan_on_conn(source, target)
        plan.pop("merged_embedding", None)
        return plan

    def merge_fingerprints(self, source: str, target: str) -> dict:
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            plan = self._fingerprint_merge_plan_on_conn(source, target)
            self._apply_fingerprint_merge_plan(plan)
            self._conn.commit()
            plan["dry_run"] = False
            plan.pop("merged_embedding", None)
            return plan
        except Exception:
            self._conn.rollback()
            raise

    def _fingerprint_merge_plan_on_conn(self, source: str, target: str) -> dict:
        source_row = self._conn.execute(
            "SELECT * FROM speaker_fingerprints WHERE fingerprint_id = ?",
            (source,),
        ).fetchone()
        target_row = self._conn.execute(
            "SELECT * FROM speaker_fingerprints WHERE fingerprint_id = ?",
            (target,),
        ).fetchone()
        if source_row is None:
            raise ValueError(f"source fingerprint not found: {source}")
        if target_row is None:
            raise ValueError(f"target fingerprint not found: {target}")
        if source_row["status"] != "active":
            raise ValueError(f"source fingerprint is not active: {source}")
        if target_row["status"] != "active":
            raise ValueError(f"target fingerprint is not active: {target}")
        source_vec = _blob_to_vec(source_row["embedding"])
        target_vec = _blob_to_vec(target_row["embedding"])
        if source_row["embedding_model"] != target_row["embedding_model"]:
            raise ValueError("cannot merge fingerprints with different embedding models")
        if len(source_vec) != len(target_vec):
            raise ValueError("cannot merge fingerprints with different embedding dimensions")
        speaker_rows = self._conn.execute(
            "SELECT COUNT(*) AS count FROM speakers WHERE fingerprint_id = ?",
            (source,),
        ).fetchone()["count"]
        return {
            "operation": "fingerprint-merge",
            "dry_run": True,
            "source_fingerprint_id": source,
            "target_fingerprint_id": target,
            "source_display_name": source_row["display_name"],
            "target_display_name": target_row["display_name"] or source_row["display_name"],
            "source_sample_count": source_row["sample_count"],
            "target_sample_count": target_row["sample_count"],
            "embedding_model": target_row["embedding_model"],
            "embedding_dimension": target_row["embedding_dimension"],
            "speaker_rows_to_repoint": speaker_rows,
            "excerpts": self._fingerprint_excerpts(source),
            "merged_embedding": [
                (target_value * target_row["sample_count"] + source_value * source_row["sample_count"])
                / (target_row["sample_count"] + source_row["sample_count"])
                for target_value, source_value in zip(target_vec, source_vec)
            ],
        }

    def _apply_fingerprint_merge_plan(self, plan: dict) -> None:
        source = plan["source_fingerprint_id"]
        target = plan["target_fingerprint_id"]
        self._conn.execute(
            "UPDATE speakers SET fingerprint_id = ?, display_name = ? WHERE fingerprint_id = ?",
            (target, plan["target_display_name"], source),
        )
        self._conn.execute(
            "UPDATE speakers SET display_name = ? WHERE fingerprint_id = ?",
            (plan["target_display_name"], target),
        )
        self._conn.execute(
            """UPDATE speaker_fingerprints
               SET embedding = ?, display_name = ?, sample_count = ?,
                   embedding_model = ?, embedding_dimension = ?, updated_at = CURRENT_TIMESTAMP
               WHERE fingerprint_id = ?""",
            (
                _vec_to_blob(plan["merged_embedding"]),
                plan["target_display_name"],
                plan["source_sample_count"] + plan["target_sample_count"],
                plan["embedding_model"],
                plan["embedding_dimension"],
                target,
            ),
        )
        self._conn.execute(
            """INSERT OR IGNORE INTO fingerprint_sources
               (fingerprint_id, transcript_id, speaker_id, created_at)
               SELECT ?, transcript_id, speaker_id, created_at
               FROM fingerprint_sources
               WHERE fingerprint_id = ?""",
            (target, source),
        )
        self._conn.execute("DELETE FROM fingerprint_sources WHERE fingerprint_id = ?", (source,))
        self._conn.execute("DELETE FROM speaker_fingerprints WHERE fingerprint_id = ?", (source,))

    def fingerprint_status_counts(self) -> dict:
        columns = self._table_columns("speaker_fingerprints")
        if "status" in columns:
            rows = self._conn.execute(
                """SELECT status, COUNT(*) AS count
                   FROM speaker_fingerprints
                   GROUP BY status"""
            )
        else:
            rows = self._conn.execute(
                """SELECT 'active' AS status, COUNT(*) AS count
                   FROM speaker_fingerprints"""
            )
        by_status = {row["status"] or "active": row["count"] for row in rows}
        active = by_status.get("active", 0)
        discarded = by_status.get("discarded", 0)
        total = sum(by_status.values())
        return {
            "active": active,
            "discarded": discarded,
            "total": total,
            "by_status": [
                {"status": status, "count": count}
                for status, count in sorted(by_status.items())
            ],
        }

    def fingerprint_model_counts(self, active_model: str | None) -> dict:
        columns = self._table_columns("speaker_fingerprints")
        if "embedding_model" in columns:
            where = "WHERE status = 'active'" if "status" in columns else ""
            rows = self._conn.execute(
                f"""SELECT embedding_model, COUNT(*) AS count
                   FROM speaker_fingerprints
                   {where}
                   GROUP BY embedding_model"""
            )
        else:
            rows = self._conn.execute(
                """SELECT NULL AS embedding_model, COUNT(*) AS count
                   FROM speaker_fingerprints"""
            )
        legacy = 0
        compatible = 0
        incompatible = 0
        by_model = []
        for row in rows:
            model = row["embedding_model"]
            count = row["count"]
            by_model.append({"embedding_model": model, "count": count})
            if model is None:
                legacy += count
            elif model == active_model:
                compatible += count
            else:
                incompatible += count
        return {
            "active_embedding_model": active_model,
            "compatible": compatible,
            "legacy": legacy,
            "incompatible": incompatible,
            "by_model": by_model,
        }

    def fingerprint_adopt_model_plan(
        self,
        embedding_model: str,
        *,
        only_legacy: bool = True,
    ) -> dict:
        columns = self._table_columns("speaker_fingerprints")
        has_model_column = "embedding_model" in columns
        has_status_column = "status" in columns
        where = []
        if only_legacy and has_model_column:
            where.append("embedding_model IS NULL")
        if has_status_column:
            where.append("status = 'active'")
        where_sql = " AND ".join(where) if where else "1 = 1"
        model_projection = "embedding_model" if has_model_column else "NULL AS embedding_model"
        status_projection = "status" if has_status_column else "'active' AS status"
        rows = [
            dict(row)
            for row in self._conn.execute(
                f"""SELECT fingerprint_id, display_name, sample_count,
                           {model_projection}, {status_projection}
                    FROM speaker_fingerprints
                    WHERE {where_sql}
                    ORDER BY fingerprint_id"""
            )
        ]
        return {
            "operation": "fingerprint-adopt-model",
            "dry_run": True,
            "embedding_model": embedding_model,
            "only_legacy": only_legacy,
            "fingerprints_seen": len(rows),
            "fingerprints_to_update": len(rows),
            "updates": rows,
        }

    def adopt_fingerprint_model(self, embedding_model: str, *, only_legacy: bool = True) -> dict:
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            plan = self.fingerprint_adopt_model_plan(embedding_model, only_legacy=only_legacy)
            where = ["status = 'active'"]
            if only_legacy:
                where.append("embedding_model IS NULL")
            where_sql = " AND ".join(where)
            self._conn.execute(
                f"""UPDATE speaker_fingerprints
                    SET embedding_model = ?,
                        embedding_dimension = COALESCE(embedding_dimension, length(embedding) / 4),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE {where_sql}""",
                (embedding_model,),
            )
            self._conn.commit()
            plan["dry_run"] = False
            return plan
        except Exception:
            self._conn.rollback()
            raise

    def fingerprint_action_plan(
        self,
        fingerprint_id: str,
        *,
        action: str,
        reason: str | None = None,
    ) -> dict:
        row = self._conn.execute(
            """SELECT fingerprint_id, display_name, sample_count, embedding_model,
                      embedding_dimension, status, discard_reason
               FROM speaker_fingerprints
               WHERE fingerprint_id = ?""",
            (fingerprint_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"fingerprint not found: {fingerprint_id}")
        if action not in {"discard", "restore", "destroy"}:
            raise ValueError(f"unknown fingerprint action: {action}")
        current = dict(row)
        target_status = current["status"]
        target_reason = current["discard_reason"]
        will_write = False
        if action == "discard":
            clean_reason = (reason or "").strip()
            if not clean_reason:
                raise ValueError("fingerprint-discard requires a non-empty --reason")
            target_status = "discarded"
            target_reason = clean_reason
            will_write = current["status"] != target_status or current["discard_reason"] != target_reason
        elif action == "restore":
            target_status = "active"
            target_reason = None
            will_write = current["status"] != target_status or current["discard_reason"] is not None
        elif action == "destroy":
            will_write = True
        source_count = self._conn.execute(
            "SELECT COUNT(*) AS count FROM fingerprint_sources WHERE fingerprint_id = ?",
            (fingerprint_id,),
        ).fetchone()["count"]
        speaker_count = self._conn.execute(
            "SELECT COUNT(*) AS count FROM speakers WHERE fingerprint_id = ?",
            (fingerprint_id,),
        ).fetchone()["count"]
        return {
            "operation": f"fingerprint-{action}",
            "dry_run": True,
            "fingerprint_id": fingerprint_id,
            "action": action,
            "will_write": will_write,
            "current_status": current["status"],
            "target_status": None if action == "destroy" else target_status,
            "current_discard_reason": current["discard_reason"],
            "target_discard_reason": None if action == "destroy" else target_reason,
            "display_name": current["display_name"],
            "sample_count": current["sample_count"],
            "embedding_model": current["embedding_model"],
            "embedding_dimension": current["embedding_dimension"],
            "fingerprint_source_rows": source_count,
            "speaker_rows_referencing": speaker_count,
        }

    def discard_fingerprint(self, fingerprint_id: str, reason: str) -> dict:
        return self._apply_fingerprint_action(
            fingerprint_id,
            action="discard",
            reason=reason,
        )

    def restore_fingerprint(self, fingerprint_id: str) -> dict:
        return self._apply_fingerprint_action(fingerprint_id, action="restore")

    def destroy_fingerprint(self, fingerprint_id: str) -> dict:
        return self._apply_fingerprint_action(fingerprint_id, action="destroy")

    def _apply_fingerprint_action(
        self,
        fingerprint_id: str,
        *,
        action: str,
        reason: str | None = None,
    ) -> dict:
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            plan = self.fingerprint_action_plan(fingerprint_id, action=action, reason=reason)
            if action == "discard" and plan["will_write"]:
                self._conn.execute(
                    """UPDATE speaker_fingerprints
                       SET status = 'discarded', discard_reason = ?, updated_at = CURRENT_TIMESTAMP
                       WHERE fingerprint_id = ?""",
                    (plan["target_discard_reason"], fingerprint_id),
                )
            elif action == "restore" and plan["will_write"]:
                self._conn.execute(
                    """UPDATE speaker_fingerprints
                       SET status = 'active', discard_reason = NULL, updated_at = CURRENT_TIMESTAMP
                       WHERE fingerprint_id = ?""",
                    (fingerprint_id,),
                )
            elif action == "destroy":
                self._conn.execute(
                    "DELETE FROM speaker_fingerprints WHERE fingerprint_id = ?",
                    (fingerprint_id,),
                )
            self._conn.commit()
            plan["dry_run"] = False
            return plan
        except Exception:
            self._conn.rollback()
            raise

    def _fingerprint_excerpts(self, fingerprint_id: str) -> list[dict]:
        rows = self._conn.execute(
            """SELECT g.transcript_id, g.segment_id, g.speaker_id, g.start_ms, g.end_ms, g.text
               FROM speakers s
               JOIN segments g
                 ON g.transcript_id = s.transcript_id
                AND g.speaker_id = s.speaker_id
               WHERE s.fingerprint_id = ?
               ORDER BY g.transcript_id, g.start_ms
               LIMIT 3""",
            (fingerprint_id,),
        )
        return [dict(row) for row in rows]

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

    def stats(self, *, active_embedding_model: str | None = None) -> dict:
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
            "fingerprint_models": self.fingerprint_model_counts(active_embedding_model),
            "fingerprint_status": self.fingerprint_status_counts(),
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
            asr_confidence=row["asr_confidence"],
            diarization_quality=row["diarization_quality"],
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


def _normalize_fingerprint_row(row: dict, index: int) -> dict:
    if not isinstance(row, dict):
        raise ValueError(f"fingerprints[{index}] must be an object")
    fingerprint_id = row.get("fingerprint_id")
    if not isinstance(fingerprint_id, str) or not fingerprint_id.strip():
        raise ValueError(f"fingerprints[{index}].fingerprint_id is required")
    embedding = row.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise ValueError(f"fingerprints[{index}].embedding must be a non-empty number list")
    try:
        cleaned_embedding = [float(value) for value in embedding]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"fingerprints[{index}].embedding must contain only numbers") from exc
    if not all(math.isfinite(value) for value in cleaned_embedding):
        raise ValueError(f"fingerprints[{index}].embedding must contain only finite numbers")
    sample_count = row.get("sample_count", 1)
    if not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError(f"fingerprints[{index}].sample_count must be a positive integer")
    display_name = row.get("display_name")
    if display_name is not None and not isinstance(display_name, str):
        raise ValueError(f"fingerprints[{index}].display_name must be a string or null")
    created_at = _optional_timestamp(row, "created_at", index)
    updated_at = _optional_timestamp(row, "updated_at", index)
    embedding_model = row.get("embedding_model")
    if embedding_model is not None and (not isinstance(embedding_model, str) or not embedding_model.strip()):
        raise ValueError(f"fingerprints[{index}].embedding_model must be a non-empty string or null")
    embedding_dimension = row.get("embedding_dimension", len(cleaned_embedding))
    if not isinstance(embedding_dimension, int) or embedding_dimension != len(cleaned_embedding):
        raise ValueError(
            f"fingerprints[{index}].embedding_dimension must equal embedding length"
        )
    status = row.get("status", "active")
    if status not in {"active", "discarded"}:
        raise ValueError(f"fingerprints[{index}].status must be active or discarded")
    discard_reason = row.get("discard_reason")
    if discard_reason is not None and not isinstance(discard_reason, str):
        raise ValueError(f"fingerprints[{index}].discard_reason must be a string or null")
    return {
        "fingerprint_id": fingerprint_id,
        "embedding": cleaned_embedding,
        "display_name": display_name,
        "sample_count": sample_count,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "status": status,
        "discard_reason": discard_reason,
        "created_at": created_at,
        "updated_at": updated_at,
    }


def _optional_timestamp(row: dict, field: str, index: int) -> str | None:
    value = row.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"fingerprints[{index}].{field} must be a non-empty string or null")
    return value


def _normalize_fingerprint_rows(rows: list[dict]) -> list[dict]:
    normalized = [_normalize_fingerprint_row(row, index) for index, row in enumerate(rows)]
    seen: set[str] = set()
    for index, row in enumerate(normalized):
        fingerprint_id = row["fingerprint_id"]
        if fingerprint_id in seen:
            raise ValueError(f"duplicate fingerprint_id in import file: {fingerprint_id}")
        seen.add(fingerprint_id)
    return normalized
