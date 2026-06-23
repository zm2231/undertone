from __future__ import annotations

import hashlib
import logging
import sqlite3
import struct
from pathlib import Path

from undertone_audio.schema import Speaker

log = logging.getLogger(__name__)

DEFAULT_SIMILARITY_THRESHOLD = 0.78


def _vec_to_blob(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _blob_to_vec(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SpeakerFingerprintStore:
    """Persisted cross-recording voice embedding store."""

    def __init__(self, db_path: Path, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        self.db_path = Path(db_path)
        self.similarity_threshold = similarity_threshold

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def assign_fingerprints(
        self,
        speakers: list[Speaker],
        persist: bool = True,
    ) -> tuple[list[Speaker], "FingerprintAssignmentPlan"]:
        with self._conn() as conn:
            stored = [
                (
                    row["fingerprint_id"],
                    _blob_to_vec(row["embedding"]),
                    row["sample_count"],
                    row["display_name"],
                )
                for row in conn.execute(
                    "SELECT fingerprint_id, embedding, sample_count, display_name "
                    "FROM speaker_fingerprints"
                )
            ]

        plan = FingerprintAssignmentPlan(self)
        updated: list[Speaker] = []

        for speaker in speakers:
            if speaker.fingerprint_id:
                updated.append(speaker)
                continue

            if not speaker.embedding:
                if speaker.display_name:
                    name_lower = speaker.display_name.strip().lower()
                    name_match = next(
                        (
                            fingerprint_id
                            for fingerprint_id, _vector, _count, display_name in stored
                            if display_name and display_name.strip().lower() == name_lower
                        ),
                        None,
                    )
                    if name_match:
                        updated.append(speaker.model_copy(update={"fingerprint_id": name_match}))
                        continue
                updated.append(speaker)
                continue

            best_id = None
            best_similarity = 0.0
            best_name = None
            for fingerprint_id, vector, _count, display_name in stored:
                similarity = _cosine(speaker.embedding, vector)
                if similarity > best_similarity:
                    best_id = fingerprint_id
                    best_similarity = similarity
                    best_name = display_name

            if best_id and best_similarity >= self.similarity_threshold:
                plan.matches.append((best_id, list(speaker.embedding)))
                updated.append(
                    speaker.model_copy(
                        update={
                            "fingerprint_id": best_id,
                            "display_name": speaker.display_name or best_name,
                        }
                    )
                )
                log.info(
                    "matched %s -> %s (sim=%.3f)",
                    speaker.speaker_id,
                    best_id,
                    best_similarity,
                )
            else:
                fingerprint_id = self._mint_fingerprint(speaker.embedding)
                plan.inserts.append((fingerprint_id, list(speaker.embedding), speaker.display_name))
                updated.append(speaker.model_copy(update={"fingerprint_id": fingerprint_id}))
                stored.append((fingerprint_id, list(speaker.embedding), 1, speaker.display_name))
                log.info(
                    "new fingerprint %s for %s (best existing sim=%.3f)",
                    fingerprint_id,
                    speaker.speaker_id,
                    best_similarity,
                )

        if persist:
            plan.commit()
        return updated, plan

    def commit_plan(self, plan: "FingerprintAssignmentPlan") -> None:
        with self._conn() as conn:
            conn.execute("BEGIN")
            try:
                self.apply_plan_on_conn(conn, plan)
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def apply_plan_on_conn(
        self,
        conn: sqlite3.Connection,
        plan: "FingerprintAssignmentPlan",
    ) -> None:
        for fingerprint_id, embedding, display_name in plan.inserts:
            self._insert(conn, fingerprint_id, embedding, display_name)
        for fingerprint_id, embedding in plan.matches:
            self._update_mean(conn, fingerprint_id, embedding)

    def label(self, fingerprint_id: str, display_name: str) -> None:
        with self._conn() as conn:
            cursor = conn.execute(
                "UPDATE speaker_fingerprints SET display_name = ? WHERE fingerprint_id = ?",
                (display_name, fingerprint_id),
            )
            conn.commit()
            if cursor.rowcount == 0:
                raise ValueError(f"fingerprint not found: {fingerprint_id}")

    def list_all(self) -> list[tuple[str, str | None, int]]:
        with self._conn() as conn:
            return [
                (row["fingerprint_id"], row["display_name"], row["sample_count"])
                for row in conn.execute(
                    "SELECT fingerprint_id, display_name, sample_count "
                    "FROM speaker_fingerprints ORDER BY fingerprint_id"
                )
            ]

    def _mint_fingerprint(self, embedding: list[float]) -> str:
        digest = hashlib.blake2b(_vec_to_blob(embedding), digest_size=8).hexdigest()
        return f"VP-{digest}"

    def _insert(
        self,
        conn: sqlite3.Connection,
        fingerprint_id: str,
        embedding: list[float],
        display_name: str | None,
    ) -> None:
        conn.execute(
            """INSERT INTO speaker_fingerprints
               (fingerprint_id, embedding, display_name, sample_count)
               VALUES (?, ?, ?, 1)""",
            (fingerprint_id, _vec_to_blob(embedding), display_name),
        )

    def _update_mean(
        self,
        conn: sqlite3.Connection,
        fingerprint_id: str,
        new_embedding: list[float],
    ) -> None:
        row = conn.execute(
            "SELECT embedding, sample_count FROM speaker_fingerprints WHERE fingerprint_id = ?",
            (fingerprint_id,),
        ).fetchone()
        if not row:
            return
        old = _blob_to_vec(row["embedding"])
        count = row["sample_count"]
        if len(old) != len(new_embedding):
            return
        merged = [
            (old_value * count + new_value) / (count + 1)
            for old_value, new_value in zip(old, new_embedding)
        ]
        conn.execute(
            """UPDATE speaker_fingerprints
               SET embedding = ?, sample_count = sample_count + 1, updated_at = CURRENT_TIMESTAMP
               WHERE fingerprint_id = ?""",
            (_vec_to_blob(merged), fingerprint_id),
        )


class FingerprintAssignmentPlan:
    """Deferred fingerprint writes committed after the parent transcript saves."""

    def __init__(self, store: SpeakerFingerprintStore):
        self._store = store
        self.inserts: list[tuple[str, list[float], str | None]] = []
        self.matches: list[tuple[str, list[float]]] = []

    def commit(self) -> None:
        if self.inserts or self.matches:
            self._store.commit_plan(self)

    def discard(self) -> None:
        self.inserts.clear()
        self.matches.clear()
