from __future__ import annotations

import hashlib
import logging
import sqlite3
import struct
from pathlib import Path

from undertone_audio.schema import Speaker

log = logging.getLogger(__name__)

DEFAULT_SIMILARITY_THRESHOLD = 0.78
MARGIN_SIMILARITY_THRESHOLD = 0.62
MARGIN_DELTA = 0.15
MIN_ENROLL_DURATION_MS = 15_000
MIN_UPDATE_DURATION_MS = 3_000


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

    def __init__(
        self,
        db_path: Path,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        embedding_model: str | None = None,
    ):
        self.db_path = Path(db_path)
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def assign_fingerprints(
        self,
        speakers: list[Speaker],
        persist: bool = True,
        speaker_durations_ms: dict[str, int] | None = None,
    ) -> tuple[list[Speaker], "FingerprintAssignmentPlan"]:
        """Match speakers to stored voice fingerprints.

        ``speaker_durations_ms`` maps ``speaker_id`` to total talk time in this
        recording and drives the quality gates. When a duration is unknown the
        gates are skipped for that speaker (margin matching still applies).
        """
        durations = speaker_durations_ms or {}
        if self.embedding_model is None:
            query = (
                "SELECT fingerprint_id, embedding, sample_count, display_name "
                "FROM speaker_fingerprints "
                "WHERE embedding_model IS NULL AND status = 'active'"
            )
            params = ()
        else:
            query = (
                "SELECT fingerprint_id, embedding, sample_count, display_name "
                "FROM speaker_fingerprints "
                "WHERE embedding_model = ? AND status = 'active'"
            )
            params = (self.embedding_model,)
        with self._conn() as conn:
            stored = [
                (
                    row["fingerprint_id"],
                    _blob_to_vec(row["embedding"]),
                    row["sample_count"],
                    row["display_name"],
                )
                for row in conn.execute(query, params)
            ]
            existing_ids = {
                row["fingerprint_id"]
                for row in conn.execute("SELECT fingerprint_id FROM speaker_fingerprints")
            }

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
            second_similarity = 0.0
            for fingerprint_id, vector, _count, display_name in stored:
                similarity = _cosine(speaker.embedding, vector)
                if similarity > best_similarity:
                    second_similarity = best_similarity
                    best_id = fingerprint_id
                    best_similarity = similarity
                    best_name = display_name
                elif similarity > second_similarity:
                    second_similarity = similarity

            duration_ms = durations.get(speaker.speaker_id)
            strong_match = best_id is not None and best_similarity >= self.similarity_threshold
            margin_match = (
                best_id is not None
                and best_similarity >= MARGIN_SIMILARITY_THRESHOLD
                and (best_similarity - second_similarity) >= MARGIN_DELTA
            )

            if strong_match or margin_match:
                # Anti-drift: only fold a sample into the stored centroid when it is
                # a strong match AND long enough. A margin (cross-channel) match gets
                # the label but never pollutes the canonical print.
                long_enough = duration_ms is None or duration_ms >= MIN_UPDATE_DURATION_MS
                if strong_match and long_enough:
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
                    "matched %s -> %s (sim=%.3f, 2nd=%.3f, %s%s)",
                    speaker.speaker_id,
                    best_id,
                    best_similarity,
                    second_similarity,
                    "strong" if strong_match else "margin",
                    "" if (strong_match and long_enough) else ", no-fold",
                )
                continue

            # No acceptable match. Quality gate: do not mint a durable identity from
            # too little signal (the garbage-magnet source). Leave such speakers
            # unassigned rather than creating a junk fingerprint.
            if duration_ms is not None and duration_ms < MIN_ENROLL_DURATION_MS:
                updated.append(speaker)
                log.info(
                    "no-enroll %s (talk=%dms < %dms, best existing sim=%.3f)",
                    speaker.speaker_id,
                    duration_ms,
                    MIN_ENROLL_DURATION_MS,
                    best_similarity,
                )
                continue

            fingerprint_id = self._mint_available_fingerprint(speaker.embedding, existing_ids)
            existing_ids.add(fingerprint_id)
            plan.inserts.append(
                (
                    fingerprint_id,
                    list(speaker.embedding),
                    speaker.display_name,
                    self.embedding_model,
                    len(speaker.embedding),
                )
            )
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
        for fingerprint_id, embedding, display_name, embedding_model, embedding_dimension in plan.inserts:
            self._insert(conn, fingerprint_id, embedding, display_name, embedding_model, embedding_dimension)
        for fingerprint_id, embedding in plan.matches:
            self._update_mean(conn, fingerprint_id, embedding)
        for fingerprint_id, transcript_id, speaker_id in plan.sources:
            conn.execute(
                """INSERT OR IGNORE INTO fingerprint_sources
                   (fingerprint_id, transcript_id, speaker_id)
                   SELECT ?, ?, ?
                   WHERE EXISTS (
                       SELECT 1 FROM speaker_fingerprints
                       WHERE fingerprint_id = ? AND status = 'active'
                   )""",
                (fingerprint_id, transcript_id, speaker_id, fingerprint_id),
            )

    def label(self, fingerprint_id: str, display_name: str) -> None:
        with self._conn() as conn:
            cursor = conn.execute(
                "UPDATE speaker_fingerprints SET display_name = ? WHERE fingerprint_id = ?",
                (display_name, fingerprint_id),
            )
            conn.commit()
            if cursor.rowcount == 0:
                raise ValueError(f"fingerprint not found: {fingerprint_id}")

    def list_all(
        self,
        *,
        status: str = "active",
    ) -> list[tuple[str, str | None, int, str | None, str, str | None]]:
        if status not in {"active", "discarded", "all"}:
            raise ValueError(f"unknown fingerprint status filter: {status}")
        where = "" if status == "all" else "WHERE status = ?"
        params = () if status == "all" else (status,)
        with self._conn() as conn:
            return [
                (
                    row["fingerprint_id"],
                    row["display_name"],
                    row["sample_count"],
                    row["embedding_model"],
                    row["status"],
                    row["discard_reason"],
                )
                for row in conn.execute(
                    f"""SELECT fingerprint_id, display_name, sample_count,
                               embedding_model, status, discard_reason
                        FROM speaker_fingerprints
                        {where}
                        ORDER BY fingerprint_id""",
                    params,
                )
            ]

    def _mint_available_fingerprint(
        self,
        embedding: list[float],
        existing_ids: set[str],
    ) -> str:
        fingerprint_id = self._mint_fingerprint(embedding)
        salt = 1
        while fingerprint_id in existing_ids:
            fingerprint_id = self._mint_fingerprint(embedding, salt=salt)
            salt += 1
        return fingerprint_id

    def _mint_fingerprint(self, embedding: list[float], *, salt: int = 0) -> str:
        h = hashlib.blake2b(digest_size=8)
        h.update(_vec_to_blob(embedding))
        h.update(b"\0")
        h.update((self.embedding_model or "").encode("utf-8"))
        if salt:
            h.update(b"\0")
            h.update(str(salt).encode("ascii"))
        digest = h.hexdigest()
        return f"VP-{digest}"

    def _insert(
        self,
        conn: sqlite3.Connection,
        fingerprint_id: str,
        embedding: list[float],
        display_name: str | None,
        embedding_model: str | None,
        embedding_dimension: int,
    ) -> None:
        conn.execute(
            """INSERT INTO speaker_fingerprints
               (fingerprint_id, embedding, display_name, sample_count, embedding_model, embedding_dimension)
               VALUES (?, ?, ?, 1, ?, ?)""",
            (fingerprint_id, _vec_to_blob(embedding), display_name, embedding_model, embedding_dimension),
        )

    def _update_mean(
        self,
        conn: sqlite3.Connection,
        fingerprint_id: str,
        new_embedding: list[float],
    ) -> None:
        if self.embedding_model is None:
            row = conn.execute(
                """SELECT embedding, sample_count, embedding_model, embedding_dimension
                   FROM speaker_fingerprints
                   WHERE fingerprint_id = ? AND embedding_model IS NULL AND status = 'active'""",
                (fingerprint_id,),
            ).fetchone()
        else:
            row = conn.execute(
                """SELECT embedding, sample_count, embedding_model, embedding_dimension
                   FROM speaker_fingerprints
                   WHERE fingerprint_id = ? AND embedding_model = ? AND status = 'active'""",
                (fingerprint_id, self.embedding_model),
            ).fetchone()
        if not row:
            return
        old = _blob_to_vec(row["embedding"])
        count = row["sample_count"]
        if len(old) != len(new_embedding) or row["embedding_dimension"] != len(new_embedding):
            return
        merged = [
            (old_value * count + new_value) / (count + 1)
            for old_value, new_value in zip(old, new_embedding)
        ]
        if self.embedding_model is None:
            conn.execute(
                """UPDATE speaker_fingerprints
                   SET embedding = ?, sample_count = sample_count + 1, updated_at = CURRENT_TIMESTAMP
                   WHERE fingerprint_id = ? AND embedding_model IS NULL AND status = 'active'""",
                (_vec_to_blob(merged), fingerprint_id),
            )
        else:
            conn.execute(
                """UPDATE speaker_fingerprints
                   SET embedding = ?, sample_count = sample_count + 1, updated_at = CURRENT_TIMESTAMP
                   WHERE fingerprint_id = ? AND embedding_model = ? AND status = 'active'""",
                (_vec_to_blob(merged), fingerprint_id, self.embedding_model),
            )


class FingerprintAssignmentPlan:
    """Deferred fingerprint writes committed after the parent transcript saves."""

    def __init__(self, store: SpeakerFingerprintStore):
        self._store = store
        self.inserts: list[tuple[str, list[float], str | None, str | None, int]] = []
        self.matches: list[tuple[str, list[float]]] = []
        self.sources: list[tuple[str, str, str]] = []

    def commit(self) -> None:
        if self.inserts or self.matches:
            self._store.commit_plan(self)

    def discard(self) -> None:
        self.inserts.clear()
        self.matches.clear()
