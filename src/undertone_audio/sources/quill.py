from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from undertone_audio.audio import AudioPreprocessor

DEFAULT_QUILL_DB = Path.home() / "Library/Application Support/Quill/quill.db"
DEFAULT_MEETINGS_DIR = Path.home() / "Library/Application Support/Quill/meetings"


@dataclass(frozen=True)
class QuillMeeting:
    meeting_id: str
    title: str | None
    word_count: int
    combined: Path | None = None
    mic: Path | None = None
    system: Path | None = None

    @property
    def has_audio(self) -> bool:
        return self.combined is not None or (self.mic is not None and self.system is not None)


class QuillSource:
    """Discover Quill recordings without trusting Quill transcript diarization."""

    def __init__(
        self,
        db_path: Path = DEFAULT_QUILL_DB,
        meetings_dir: Path = DEFAULT_MEETINGS_DIR,
        preprocessor: AudioPreprocessor | None = None,
    ):
        self.db_path = Path(db_path)
        self.meetings_dir = Path(meetings_dir)
        self.preprocessor = preprocessor or AudioPreprocessor()

    def list_meetings(self, limit: int = 50) -> list[QuillMeeting]:
        rows = self._meeting_rows(limit)
        return [self.meeting(meeting_id=row[0], title=row[1], word_count=row[2] or 0) for row in rows]

    def meeting(
        self,
        meeting_id: str,
        title: str | None = None,
        word_count: int | None = None,
    ) -> QuillMeeting:
        if title is None or word_count is None:
            row = self._meeting_row(meeting_id)
            if row:
                title = title if title is not None else row[1]
                word_count = word_count if word_count is not None else row[2]
        return QuillMeeting(
            meeting_id=meeting_id,
            title=title,
            word_count=word_count or 0,
            combined=self.find_audio(meeting_id, "combined"),
            mic=self.find_audio(meeting_id, "mic"),
            system=self.find_audio(meeting_id, "system"),
        )

    def local_audio_for_meeting(self, meeting_id: str) -> tuple[Path, QuillMeeting]:
        if not self.db_path.exists():
            raise ValueError(
                f"Quill DB not found at {self.db_path}. "
                "Install/sign in to Quill, or pass --quill-db and --meetings-dir. "
                "Check readiness with `undertone doctor`."
            )
        if self._meeting_row(meeting_id) is None:
            raise ValueError(
                f"Quill meeting {meeting_id} was not found in {self.db_path}. "
                "Run `undertone quill-list` to see available meetings."
            )
        meeting = self.meeting(meeting_id)
        if meeting.combined:
            return self.preprocessor.normalize(meeting.combined), meeting
        if meeting.mic and meeting.system:
            return self.preprocessor.mix_to_wav(
                [meeting.mic, meeting.system],
                label=f"quill-{meeting_id}-mic-system",
            ), meeting
        raise ValueError(
            f"No Quill audio found for meeting {meeting_id}. "
            f"Looked under {self.meetings_dir}. "
            "Quill text/diarization is provenance only; undertone needs local recording audio."
        )

    def find_audio(self, meeting_id: str, variant: str) -> Path | None:
        if not self.meetings_dir.exists():
            return None
        for directory in self.meetings_dir.glob(f"*-{meeting_id}"):
            for path in directory.glob(f"*-{variant}.m4a"):
                return path
        return None

    def recorded_at(self, meeting: QuillMeeting) -> datetime | None:
        path = meeting.combined or meeting.mic or meeting.system
        if path is None:
            return None
        match = re.search(r"-FINAL-(\d+\.\d+)-\d+\.\d+-", path.name)
        if not match:
            return None
        return datetime.fromtimestamp(float(match.group(1))).astimezone()

    def source_metadata(self, meeting: QuillMeeting) -> dict:
        metadata = {
            "title": meeting.title,
            "source": "quill",
            "quill_meeting_id": meeting.meeting_id,
            "quill_word_count": meeting.word_count,
        }
        return {key: value for key, value in metadata.items() if value not in (None, "")}

    def _meeting_rows(self, limit: int) -> list[tuple[str, str | None, int | None]]:
        if not self.db_path.exists():
            return []
        with sqlite3.connect(self.db_path) as conn:
            return list(
                conn.execute(
                    """SELECT id, COALESCE(manualTitle, eventTitle, title, llmTitle), word_count
                       FROM Meeting
                       ORDER BY start DESC LIMIT ?""",
                    (limit,),
                )
            )

    def _meeting_row(self, meeting_id: str) -> tuple[str, str | None, int | None] | None:
        if not self.db_path.exists():
            return None
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                """SELECT id, COALESCE(manualTitle, eventTitle, title, llmTitle), word_count
                   FROM Meeting WHERE id = ?""",
                (meeting_id,),
            ).fetchone()
