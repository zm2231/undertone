import json
import sqlite3
from types import SimpleNamespace
from pathlib import Path

from undertone_audio.cli import main
from undertone_audio.dedupe import AudioSignature
from undertone_audio.engines.base import RawTranscript
from undertone_audio.schema import EnrichedTranscript, Segment, Speaker, TranscriptMetadata
from undertone_audio.sources.meet import MEET_ADC_COMMAND
from undertone_audio.sources.meet import MeetSource
from undertone_audio.sources.meet import download_recording_mp4
from undertone_audio.sources.quill import QuillSource
from undertone_audio.storage import TranscriptStore


class FakePreprocessor:
    def __init__(self, mixed_path=None):
        self.normalized = []
        self.mixed = []
        self.mixed_path = mixed_path

    def normalize(self, path):
        self.normalized.append(Path(path))
        return Path(path)

    def mix_to_wav(self, paths, label):
        self.mixed.append((list(paths), label))
        if self.mixed_path:
            self.mixed_path.write_bytes(b"mixed")
            return self.mixed_path
        return Path(paths[0])


def _quill_db(path):
    with sqlite3.connect(path) as conn:
        conn.execute(
            """CREATE TABLE Meeting (
                id TEXT PRIMARY KEY,
                manualTitle TEXT,
                eventTitle TEXT,
                title TEXT,
                llmTitle TEXT,
                word_count INTEGER,
                start INTEGER
            )"""
        )
        conn.execute(
            """INSERT INTO Meeting
               (id, manualTitle, eventTitle, title, llmTitle, word_count, start)
               VALUES ('m1', 'Manual', NULL, 'Title', 'LLM', 42, 2)"""
        )


def _seed_audio_duplicate(db_path, *, transcript_id="existing-audio", fingerprint="audio-fp"):
    store = TranscriptStore(db_path)
    try:
        store.save(
            EnrichedTranscript(
                transcript_id=transcript_id,
                metadata=TranscriptMetadata(
                    duration_ms=1000,
                    engine="fixture",
                    content_audio_fp=fingerprint,
                    content_audio_fp_algorithm="chromaprint-fpcalc-v1",
                ),
                speakers=[Speaker(speaker_id="S1")],
                segments=[
                    Segment(
                        segment_id="s1",
                        speaker_id="S1",
                        start_ms=0,
                        end_ms=1000,
                        text="existing audio",
                    )
                ],
            )
        )
    finally:
        store.close()


def test_quill_source_prefers_combined_audio_without_loading_transcript(tmp_path):
    db = tmp_path / "quill.db"
    meetings = tmp_path / "meetings"
    folder = meetings / "123-m1"
    folder.mkdir(parents=True)
    combined = folder / "m1-FINAL-1000.000000-1010.000000-combined.m4a"
    mic = folder / "m1-FINAL-1000.000000-1010.000000-mic.m4a"
    system = folder / "m1-FINAL-1000.000000-1010.000000-system.m4a"
    combined.write_bytes(b"combined")
    mic.write_bytes(b"mic")
    system.write_bytes(b"system")
    _quill_db(db)
    preprocessor = FakePreprocessor()

    source = QuillSource(db_path=db, meetings_dir=meetings, preprocessor=preprocessor)
    audio, meeting = source.local_audio_for_meeting("m1")

    assert audio == combined
    assert meeting.title == "Manual"
    assert preprocessor.normalized == [combined]
    assert preprocessor.mixed == []
    assert source.source_metadata(meeting)["quill_meeting_id"] == "m1"


def test_quill_source_mixes_mic_and_system_when_combined_missing(tmp_path):
    db = tmp_path / "quill.db"
    meetings = tmp_path / "meetings"
    folder = meetings / "123-m1"
    folder.mkdir(parents=True)
    mic = folder / "m1-FINAL-1000.000000-1010.000000-mic.m4a"
    system = folder / "m1-FINAL-1000.000000-1010.000000-system.m4a"
    mic.write_bytes(b"mic")
    system.write_bytes(b"system")
    mixed = tmp_path / "mixed.wav"
    _quill_db(db)
    preprocessor = FakePreprocessor(mixed_path=mixed)

    source = QuillSource(db_path=db, meetings_dir=meetings, preprocessor=preprocessor)
    audio, _meeting = source.local_audio_for_meeting("m1")

    assert audio == mixed
    assert preprocessor.mixed == [([mic, system], "quill-m1-mic-system")]


def test_meet_source_prefers_explicit_local_audio(tmp_path, monkeypatch):
    local = tmp_path / "mine.mp4"
    local.write_bytes(b"mp4")
    source = MeetSource(download_dir=tmp_path, preprocessor=FakePreprocessor())
    monkeypatch.setattr("undertone_audio.sources.meet.list_recordings", lambda conference_record: (_ for _ in ()).throw(AssertionError("should not list recordings")))

    selection = source.select("conferenceRecords/abc", local_audio=local)

    assert selection.audio_path == local
    assert selection.source_kind == "local-audio"
    assert selection.source_metadata["audio_priority"] == "explicit-local"


def test_meet_source_downloads_recording_before_text_fallback(tmp_path, monkeypatch):
    source = MeetSource(download_dir=tmp_path, preprocessor=FakePreprocessor())
    monkeypatch.setattr(
        "undertone_audio.sources.meet.list_recordings",
        lambda conference_record, **kwargs: [
            {
                "name": "recording",
                "state": "FILE_GENERATED",
                "driveDestination": {"file": "drive123", "exportUri": "https://example"},
            }
        ],
    )

    def fake_download(file_id, dest, **kwargs):
        dest.write_bytes(b"mp4")
        return dest

    monkeypatch.setattr("undertone_audio.sources.meet.download_recording_mp4", fake_download)
    monkeypatch.setattr(
        "undertone_audio.sources.meet.parse_meet_record",
        lambda conference_record, **kwargs: (_ for _ in ()).throw(AssertionError("should not parse text")),
    )

    selection = source.select("conferenceRecords/abc")

    assert selection.audio_path == tmp_path / "drive123.mp4"
    assert selection.source_kind == "meet-recording"
    assert selection.source_metadata["drive_file_id"] == "drive123"


def test_meet_recording_download_removes_partial_file(tmp_path, monkeypatch):
    class FailingResponse:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"partial"
            raise OSError("network dropped")

    monkeypatch.setattr("undertone_audio.sources.meet._headers", lambda **kwargs: {})
    monkeypatch.setattr(
        "undertone_audio.sources.meet.requests.get",
        lambda *args, **kwargs: FailingResponse(),
    )

    dest = tmp_path / "recording.mp4"
    try:
        download_recording_mp4("drive123", dest)
    except OSError as exc:
        assert "network dropped" in str(exc)
    else:
        raise AssertionError("download should fail")

    assert not dest.exists()
    assert not list(tmp_path.glob(".recording.mp4.*.tmp"))


def test_meet_source_text_fallback_when_no_audio(tmp_path, monkeypatch):
    raw = RawTranscript(
        duration_ms=1000,
        language="en",
        engine="meet-text-fallback",
        speakers=[Speaker(speaker_id="S1", display_name="Speaker")],
        segments=[Segment(segment_id="s1", speaker_id="S1", start_ms=0, end_ms=1000, text="hello")],
    )
    source = MeetSource(download_dir=tmp_path, preprocessor=FakePreprocessor())
    monkeypatch.setattr("undertone_audio.sources.meet.list_recordings", lambda conference_record, **kwargs: [])
    monkeypatch.setattr(
        "undertone_audio.sources.meet.parse_meet_record",
        lambda conference_record, **kwargs: (raw, {"source": "google-meet"}),
    )

    selection = source.select("conferenceRecords/abc")

    assert selection.audio_path is None
    assert selection.source_kind == "meet-text-fallback"
    assert selection.raw_fallback == raw
    assert selection.source_metadata["audio_priority"] == "meet-text-fallback"


def test_quill_list_cli_reports_audio_candidates(tmp_path, monkeypatch, capsys):
    from undertone_audio.cli import main
    from undertone_audio.sources.quill import QuillMeeting

    class FakeQuillSource:
        def __init__(self, db_path, meetings_dir, **kwargs):
            pass

        def list_meetings(self, limit):
            return [
                QuillMeeting(
                    meeting_id="m1",
                    title="Title",
                    word_count=5,
                    combined=Path("/tmp/combined.m4a"),
                )
            ]

    monkeypatch.setattr("undertone_audio.commands.sources.QuillSource", FakeQuillSource)

    db = tmp_path / "quill.db"
    db.write_bytes(b"")
    meetings_dir = tmp_path / "meetings"
    meetings_dir.mkdir()
    base = ["--quill-db", str(db), "--meetings-dir", str(meetings_dir)]

    assert main(["quill-list", *base, "--limit", "1", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["meeting_id"] == "m1"
    assert payload[0]["ingestable"] is True

    assert main(["quill-list", *base, "--limit", "1"]) == 0
    assert "Quill meetings" in capsys.readouterr().out


def test_quill_list_missing_db_names_fix(tmp_path, capsys):
    missing_db = tmp_path / "missing-quill.db"
    missing_meetings = tmp_path / "missing-meetings"

    assert (
        main(
            [
                "quill-list",
                "--quill-db",
                str(missing_db),
                "--meetings-dir",
                str(missing_meetings),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Quill DB not found" in output
    assert "--quill-db" in output


def test_quill_ingest_missing_db_names_fix(tmp_path, capsys):
    missing_db = tmp_path / "missing-quill.db"
    missing_meetings = tmp_path / "missing-meetings"

    assert (
        main(
            [
                "quill-ingest",
                "missing",
                "--quill-db",
                str(missing_db),
                "--meetings-dir",
                str(missing_meetings),
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Quill DB not found" in captured.err
    assert "undertone doctor" in captured.err


def test_quill_ingest_explicit_meeting_transcription_crash_exits_nonzero(tmp_path, monkeypatch, capsys):
    from undertone_audio.sources.quill import QuillMeeting

    audio = tmp_path / "m1.wav"
    audio.write_bytes(b"x")
    meeting = QuillMeeting(meeting_id="m1", title="Title", word_count=5, combined=audio)

    class FakeQuillSource:
        def __init__(self, db_path, meetings_dir, **kwargs):
            pass

        def meeting(self, meeting_id):
            return meeting

        def local_audio_for_meeting(self, meeting_id):
            return audio, meeting

        def recorded_at(self, hydrated):
            return None

        def source_metadata(self, hydrated):
            return {}

    class CrashEngine:
        name = "fake"

        async def healthcheck(self):
            return True

        async def transcribe(self, audio_path):
            raise RuntimeError("transcription crashed")

    monkeypatch.setattr("undertone_audio.commands.sources.QuillSource", FakeQuillSource)
    monkeypatch.setattr("undertone_audio.commands.sources.create_engine", lambda name, config: CrashEngine())
    monkeypatch.setenv("UNDERTONE_WEBHOOK_ENABLED", "0")

    assert main(["--db", str(tmp_path / "z.db"), "quill-ingest", "m1"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "transcription crashed" in captured.err


def test_quill_ingest_explicit_source_failure_progress_json_is_error(
    tmp_path,
    monkeypatch,
    capsys,
):
    from undertone_audio.sources.quill import QuillMeeting

    meeting = QuillMeeting(meeting_id="m1", title="Title", word_count=5)

    class FakeQuillSource:
        def __init__(self, db_path, meetings_dir, **kwargs):
            pass

        def meeting(self, meeting_id):
            return meeting

        def local_audio_for_meeting(self, meeting_id):
            raise RuntimeError("no local audio")

    monkeypatch.setattr("undertone_audio.commands.sources.QuillSource", FakeQuillSource)

    assert (
        main(
            [
                "--db",
                str(tmp_path / "z.db"),
                "quill-ingest",
                "m1",
                "--progress",
                "json",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["event"] == "error"
    assert payload["error"] == "no local audio"


def test_quill_ingest_explicit_meeting_keeps_report_off_stdout(tmp_path, monkeypatch, capsys):
    from undertone_audio.schema import Segment, Speaker
    from undertone_audio.sources.quill import QuillMeeting

    audio = tmp_path / "m1.wav"
    audio.write_bytes(b"x")
    meeting = QuillMeeting(meeting_id="m1", title="Title", word_count=5, combined=audio)

    class FakeQuillSource:
        def __init__(self, db_path, meetings_dir, **kwargs):
            pass

        def meeting(self, meeting_id):
            return meeting

        def local_audio_for_meeting(self, meeting_id):
            return audio, meeting

        def recorded_at(self, hydrated):
            return None

        def source_metadata(self, hydrated):
            return {"source": "quill"}

    class FakeEngine:
        name = "fake"

        async def transcribe(self, audio_path):
            from undertone_audio.engines.base import RawTranscript

            return RawTranscript(
                duration_ms=1000,
                language="en",
                engine="fixture",
                speakers=[Speaker(speaker_id="S1")],
                segments=[
                    Segment(
                        segment_id="s1",
                        speaker_id="S1",
                        start_ms=0,
                        end_ms=1000,
                        text="quill transcript",
                    )
                ],
            )

    monkeypatch.setattr("undertone_audio.commands.sources.QuillSource", FakeQuillSource)
    monkeypatch.setattr("undertone_audio.commands.sources.create_engine", lambda name, config: FakeEngine())
    monkeypatch.setenv("UNDERTONE_WEBHOOK_ENABLED", "0")

    assert (
        main(
            [
                "--db",
                str(tmp_path / "z.db"),
                "quill-ingest",
                "m1",
                "--json",
                "--voice-metrics",
                "off",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    transcript = json.loads(captured.out)
    assert transcript["transcript_id"] == "m1"
    assert transcript["segments"][0]["text"] == "quill transcript"
    assert "ingested" not in transcript


def test_quill_ingest_missing_db_graceful_even_when_engine_unavailable(tmp_path, monkeypatch, capsys):
    def boom(name, config):
        raise RuntimeError("engine should not be constructed")

    monkeypatch.setattr("undertone_audio.commands.sources.create_engine", boom)

    assert (
        main(
            [
                "--db",
                str(tmp_path / "z.db"),
                "quill-ingest",
                "missing",
                "--quill-db",
                str(tmp_path / "none.db"),
                "--meetings-dir",
                str(tmp_path / "none"),
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Quill DB not found" in captured.err


def test_quill_ingest_duplicate_refused_before_engine_construction(tmp_path, monkeypatch, capsys):
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(
        json.dumps(
            {
                "duration_ms": 1000,
                "language": "en",
                "engine": "fixture",
                "speakers": [{"speaker_id": "S1"}],
                "segments": [
                    {"segment_id": "s1", "speaker_id": "S1", "start_ms": 0, "end_ms": 1000, "text": "hi"}
                ],
            }
        )
    )
    db = tmp_path / "z.db"
    assert main(["--db", str(db), "finalize-json", str(raw_path), "--transcript-id", "dup"]) == 0
    capsys.readouterr()

    def boom(name, config):
        raise RuntimeError("engine should not be constructed")

    monkeypatch.setattr("undertone_audio.commands.sources.create_engine", boom)

    assert (
        main(
            [
                "--db",
                str(db),
                "quill-ingest",
                "dup",
                "--quill-db",
                str(tmp_path / "none.db"),
                "--meetings-dir",
                str(tmp_path / "none"),
            ]
        )
        == 1
    )
    assert "already exists" in capsys.readouterr().err


def test_quill_ingest_audio_duplicate_skips_before_engine_construction(
    tmp_path,
    monkeypatch,
    capsys,
):
    db = tmp_path / "z.db"
    _seed_audio_duplicate(db)
    quill_db = tmp_path / "quill.db"
    meetings = tmp_path / "meetings"
    folder = meetings / "123-m1"
    folder.mkdir(parents=True)
    audio_path = folder / "m1-FINAL-1000.000000-1010.000000-combined.m4a"
    audio_path.write_bytes(b"audio")
    _quill_db(quill_db)
    monkeypatch.setattr(
        "undertone_audio.commands.sources.AudioPreprocessor",
        lambda process_timeout_seconds=None: FakePreprocessor(),
    )
    monkeypatch.setattr(
        "undertone_audio.commands.common.audio_signature_for_path",
        lambda path, timeout_seconds: AudioSignature("audio-fp"),
    )

    def boom(name, config):
        raise RuntimeError("engine should not be constructed")

    monkeypatch.setattr("undertone_audio.commands.sources.create_engine", boom)

    assert (
        main(
            [
                "--db",
                str(db),
                "quill-ingest",
                "m1",
                "--quill-db",
                str(quill_db),
                "--meetings-dir",
                str(meetings),
                "--json",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["ingested"] == 0
    assert payload["skipped"][0]["meeting_id"] == "m1"
    assert payload["skipped"][0]["existing_transcript_id"] == "existing-audio"
    assert payload["skipped"][0]["match_type"] == "audio"


def test_quill_ingest_audio_duplicate_progress_json_stays_event_stream(
    tmp_path,
    monkeypatch,
    capsys,
):
    db = tmp_path / "z.db"
    _seed_audio_duplicate(db)
    quill_db = tmp_path / "quill.db"
    meetings = tmp_path / "meetings"
    folder = meetings / "123-m1"
    folder.mkdir(parents=True)
    audio_path = folder / "m1-FINAL-1000.000000-1010.000000-combined.m4a"
    audio_path.write_bytes(b"audio")
    _quill_db(quill_db)
    monkeypatch.setattr(
        "undertone_audio.commands.sources.AudioPreprocessor",
        lambda process_timeout_seconds=None: FakePreprocessor(),
    )
    monkeypatch.setattr(
        "undertone_audio.commands.common.audio_signature_for_path",
        lambda path, timeout_seconds: AudioSignature("audio-fp"),
    )

    def boom(name, config):
        raise RuntimeError("engine should not be constructed")

    monkeypatch.setattr("undertone_audio.commands.sources.create_engine", boom)

    assert (
        main(
            [
                "--db",
                str(db),
                "quill-ingest",
                "m1",
                "--quill-db",
                str(quill_db),
                "--meetings-dir",
                str(meetings),
                "--json",
                "--progress",
                "json",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    events = [json.loads(line) for line in captured.err.splitlines()]
    assert [event["event"] for event in events] == ["start", "skipped"]
    assert all("event" in event for event in events)
    assert events[1]["existing_transcript_id"] == "existing-audio"


def test_quill_ingest_skip_existing_does_not_construct_source_for_explicit_meeting(
    tmp_path,
    monkeypatch,
    capsys,
):
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(
        json.dumps(
            {
                "duration_ms": 1000,
                "language": "en",
                "engine": "fixture",
                "speakers": [{"speaker_id": "S1"}],
                "segments": [
                    {
                        "segment_id": "s1",
                        "speaker_id": "S1",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "text": "existing quill",
                    }
                ],
            }
        )
    )
    db = tmp_path / "z.db"
    assert main(["--db", str(db), "finalize-json", str(raw_path), "--transcript-id", "m1"]) == 0
    capsys.readouterr()

    class FailingQuillSource:
        def __init__(self, **kwargs):
            raise AssertionError("QuillSource should not be constructed")

    monkeypatch.setattr("undertone_audio.commands.sources.QuillSource", FailingQuillSource)

    assert main(["--db", str(db), "quill-ingest", "m1", "--skip-existing"]) == 0
    assert json.loads(capsys.readouterr().out)["skipped"] is True


def test_meet_discover_cli_reports_recording_and_transcript(monkeypatch, capsys):
    from undertone_audio.cli import main

    monkeypatch.setattr(
        "undertone_audio.commands.sources.list_recent_conferences",
        lambda limit, **kwargs: [{"name": "conferenceRecords/abc", "startTime": "2026-06-01T00:00:00Z"}],
    )
    monkeypatch.setattr(
        "undertone_audio.commands.sources.list_recordings",
        lambda name, **kwargs: [
            {
                "state": "FILE_GENERATED",
                "driveDestination": {"file": "drive123"},
            }
        ],
    )
    monkeypatch.setattr(
        "undertone_audio.commands.sources.list_transcripts",
        lambda name, **kwargs: [{"name": "transcript"}],
    )

    assert main(["meet-discover", "--limit", "1", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == [
        {
            "conference_record": "conferenceRecords/abc",
            "start_time": "2026-06-01T00:00:00Z",
            "end_time": None,
            "space": None,
            "has_recording": True,
            "has_transcript": True,
            "google_account": None,
        }
    ]

    assert main(["meet-discover", "--limit", "1"]) == 0
    assert "Google Meet conference records" in capsys.readouterr().out


def test_meet_ingest_skip_existing_does_not_select_source(tmp_path, monkeypatch, capsys):
    from undertone_audio.commands.sources import _stable_id

    conference_record = "conferenceRecords/abc"
    transcript_id = "meet-" + _stable_id(conference_record)
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(
        json.dumps(
            {
                "duration_ms": 1000,
                "language": "en",
                "engine": "fixture",
                "speakers": [{"speaker_id": "S1"}],
                "segments": [
                    {
                        "segment_id": "s1",
                        "speaker_id": "S1",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "text": "existing meet",
                    }
                ],
            }
        )
    )
    db = tmp_path / "z.db"
    assert main(["--db", str(db), "finalize-json", str(raw_path), "--transcript-id", transcript_id]) == 0
    capsys.readouterr()

    class FailingMeetSource:
        def __init__(self, **kwargs):
            raise AssertionError("MeetSource should not be constructed")

    monkeypatch.setattr("undertone_audio.commands.sources.MeetSource", FailingMeetSource)

    assert main(["--db", str(db), "meet-ingest", conference_record, "--skip-existing"]) == 0
    assert json.loads(capsys.readouterr().out)["skipped"] is True


def test_meet_ingest_audio_duplicate_skips_before_engine_construction(
    tmp_path,
    monkeypatch,
    capsys,
):
    db = tmp_path / "z.db"
    _seed_audio_duplicate(db)
    audio_path = tmp_path / "meet.mp4"
    audio_path.write_bytes(b"audio")

    class FakeMeetSource:
        def __init__(self, **kwargs):
            pass

        def select(self, conference_record, local_audio=None, allow_text_fallback=True):
            return SimpleNamespace(
                audio_path=audio_path,
                source_kind="local-audio",
                source_metadata={"source": "meet"},
                raw_fallback=None,
            )

    monkeypatch.setattr("undertone_audio.commands.sources.MeetSource", FakeMeetSource)
    monkeypatch.setattr(
        "undertone_audio.commands.common.audio_signature_for_path",
        lambda path, timeout_seconds: AudioSignature("audio-fp"),
    )

    def boom(name, config):
        raise RuntimeError("engine should not be constructed")

    monkeypatch.setattr("undertone_audio.commands.sources.create_engine", boom)

    assert (
        main(
            [
                "--db",
                str(db),
                "meet-ingest",
                "conferenceRecords/abc",
                "--transcript-id",
                "new-meet",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["transcript_id"] == "new-meet"
    assert payload["existing_transcript_id"] == "existing-audio"
    assert payload["match_type"] == "audio"


def test_meet_auth_fix_names_scoped_gcloud_command():
    from undertone_audio.sources.meet import _meet_auth_fix

    fix = _meet_auth_fix("Reauthentication is needed")

    assert MEET_ADC_COMMAND in fix
    assert "meetings.space.readonly" in fix
    assert "drive.meet.readonly" in fix
