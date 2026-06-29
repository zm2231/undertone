import json
import logging
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

from undertone_audio.config import Config
from undertone_audio import webhooks
from undertone_audio.commands import fingerprints as fingerprint_commands
from undertone_audio.commands.common import config_for_args
from undertone_audio.engines.base import RawTranscript
from undertone_audio.schema import EnrichedTranscript, Segment, Speaker, TranscriptMetadata
from undertone_audio.storage import TranscriptStore
from undertone_audio.cli import _configure_logging, _parser, main


def test_cli_finalize_load_search_and_emit(tmp_path, monkeypatch, capsys):
    calls = []

    class Response:
        status_code = 204

    def fake_post(url, *, data, headers, timeout):
        calls.append({"url": url, "data": data, "headers": headers, "timeout": timeout})
        return Response()

    monkeypatch.setattr(webhooks.requests, "post", fake_post)
    monkeypatch.setenv("UNDERTONE_WEBHOOK_URL", "https://zen.example/webhooks/workflow/ready")
    monkeypatch.setenv("UNDERTONE_WEBHOOK_SECRET", "shared-secret")

    raw_path = tmp_path / "raw.json"
    raw_path.write_text(
        json.dumps(
            {
                "duration_ms": 2000,
                "language": "en",
                "engine": "fixture",
                "speakers": [
                    {"speaker_id": "S1", "fingerprint_id": "VP-1", "embedding": [0.1, 0.2]}
                ],
                "segments": [
                    {
                        "segment_id": "seg1",
                        "speaker_id": "S1",
                        "start_ms": 0,
                        "end_ms": 2000,
                        "text": "operator path works",
                    }
                ],
            }
        )
    )
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({"title": "raw producer meeting"}))
    db = tmp_path / "state" / "undertone.db"

    assert (
        main(
            [
                "--db",
                str(db),
                "finalize-json",
                str(raw_path),
                "--transcript-id",
                "cli-1",
                "--source-metadata",
                str(metadata_path),
                "--diarization-state",
                "ok",
            ]
        )
        == 0
    )
    finalized = json.loads(capsys.readouterr().out)
    assert finalized["transcript_id"] == "cli-1"
    assert finalized["metadata"]["source_metadata"] == {"title": "raw producer meeting"}
    assert "scope" not in finalized["metadata"]
    assert len(calls) == 1

    assert main(["--db", str(db), "load", "cli-1"]) == 0
    loaded = json.loads(capsys.readouterr().out)
    assert loaded["speakers"][0]["fingerprint_id"] == "VP-1"
    assert loaded["segments"][0]["text"] == "operator path works"

    assert main(["--db", str(db), "search", "operator", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["transcript_id"] == "cli-1"
    assert rows[0]["segment_id"] == "seg1"

    assert main(["--db", str(db), "emit-ready", "cli-1", "--json"]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted == {"transcript_id": "cli-1", "emitted": True, "reason": "ok"}
    assert len(calls) == 2


def test_top_level_help_hides_aliases_without_blank_descriptions():
    help_text = _parser().format_help()

    assert "relabel" in help_text
    assert "resolve-names" not in help_text


def test_cli_operator_commands_for_saved_transcript(tmp_path, monkeypatch, capsys):
    calls = []

    class Response:
        status_code = 204

    monkeypatch.setattr(
        webhooks.requests,
        "post",
        lambda url, *, data, headers, timeout: calls.append(
            {"url": url, "data": data, "headers": headers, "timeout": timeout}
        )
        or Response(),
    )
    monkeypatch.setenv("UNDERTONE_WEBHOOK_URL", "https://zen.example/webhooks/workflow/ready")
    monkeypatch.setenv("UNDERTONE_WEBHOOK_SECRET", "shared-secret")

    raw_path = tmp_path / "raw.json"
    raw_path.write_text(
        json.dumps(
            {
                "duration_ms": 16000,
                "language": "en",
                "engine": "fixture",
                "speakers": [{"speaker_id": "S1", "embedding": [1.0, 0.0]}],
                "segments": [
                    {
                        "segment_id": "seg1",
                        "speaker_id": "S1",
                        "start_ms": 0,
                        "end_ms": 16000,
                        "text": "um operator commands work",
                    }
                ],
            }
        )
    )
    db = tmp_path / "undertone.db"

    assert main(["--db", str(db), "finalize-json", str(raw_path), "--transcript-id", "ops-1"]) == 0
    created = json.loads(capsys.readouterr().out)
    fingerprint_id = created["speakers"][0]["fingerprint_id"]

    assert main(["--db", str(db), "list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["transcript_id"] == "ops-1"
    assert listed[0]["speaker_count"] == 1

    assert main(["--db", str(db), "stats", "--json"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["transcript_count"] == 1
    assert stats["total_duration_ms"] == 16000

    assert (
        main(["--db", str(db), "fingerprint-label", fingerprint_id, "Alex Rivera", "--json"]) == 0
    )
    assert json.loads(capsys.readouterr().out)["display_name"] == "Alex Rivera"
    assert main(["--db", str(db), "fingerprints", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["display_name"] == "Alex Rivera"

    assert main(["--db", str(db), "webhook-preview", "ops-1", "--json"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["should_emit"] is True
    assert preview["signature_header"].startswith("sha256=")
    assert preview["payload"]["store_ref"].endswith("#ops-1")

    assert main(["--db", str(db), "reenrich", "ops-1", "--no-fillers"]) == 0
    refreshed = json.loads(capsys.readouterr().out)
    assert refreshed["transcript_id"] == "ops-1"
    assert refreshed["segments"][0]["enrichment"]["fillers"] == []

    assert main(["--db", str(db), "load", "ops-1", "--output-format", "csv"]) == 0
    csv_body = capsys.readouterr().out
    assert "transcript_id,speaker_id" in csv_body
    assert "ops-1" in csv_body

    assert main(["--db", str(db), "delete", "ops-1", "--yes", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["deleted"] is True
    assert main(["--db", str(db), "load", "ops-1"]) == 1


def test_relabel_restamps_saved_speaker_names_without_reenrich(tmp_path, capsys):
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(
        json.dumps(
            {
                "duration_ms": 16000,
                "language": "en",
                "engine": "fixture",
                "speakers": [{"speaker_id": "S1", "embedding": [1.0, 0.0]}],
                "segments": [
                    {
                        "segment_id": "seg1",
                        "speaker_id": "S1",
                        "start_ms": 0,
                        "end_ms": 16000,
                        "text": "name me later",
                    }
                ],
            }
        )
    )
    db = tmp_path / "undertone.db"

    assert main(["--db", str(db), "finalize-json", str(raw_path), "--transcript-id", "name-1"]) == 0
    created = json.loads(capsys.readouterr().out)
    fingerprint_id = created["speakers"][0]["fingerprint_id"]
    assert created["speakers"][0]["display_name"] is None
    assert created["speakers"][0]["match"]["kind"] == "new"

    assert main(["--db", str(db), "fingerprint-label", fingerprint_id, "Alex Rivera"]) == 0
    capsys.readouterr()
    assert main(["--db", str(db), "load", "name-1"]) == 0
    before = json.loads(capsys.readouterr().out)
    assert before["speakers"][0]["display_name"] is None
    before_match = before["speakers"][0]["match"]
    assert before_match["kind"] == "new"

    assert main(["--db", str(db), "relabel", "name-1", "--json"]) == 0
    relabel = json.loads(capsys.readouterr().out)
    assert relabel["speakers_updated"] == 1
    assert main(["--db", str(db), "load", "name-1"]) == 0
    after = json.loads(capsys.readouterr().out)
    assert after["speakers"][0]["display_name"] == "Alex Rivera"
    assert after["speakers"][0]["match"] == before_match

    assert main(["--db", str(db), "load", "name-1", "--output-format", "raw-json"]) == 0
    raw_payload = json.loads(capsys.readouterr().out)
    assert raw_payload["speakers"][0]["display_name"] == "Alex Rivera"
    assert raw_payload["speakers"][0]["match"] == before_match


def test_relabel_does_not_blank_name_when_fingerprint_row_missing(tmp_path, capsys):
    db = tmp_path / "undertone.db"
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(
        json.dumps(
            {
                "duration_ms": 1000,
                "language": "en",
                "engine": "fixture",
                "speakers": [
                    {
                        "speaker_id": "S1",
                        "fingerprint_id": "VP-missing",
                        "display_name": "Keep Me",
                    }
                ],
                "segments": [
                    {
                        "segment_id": "seg1",
                        "speaker_id": "S1",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "text": "orphan fingerprint",
                    }
                ],
            }
        )
    )

    assert main(["--db", str(db), "finalize-json", str(raw_path), "--transcript-id", "orphan"]) == 0
    capsys.readouterr()
    assert main(["--db", str(db), "relabel", "orphan", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["speakers_updated"] == 0
    assert main(["--db", str(db), "load", "orphan"]) == 0
    loaded = json.loads(capsys.readouterr().out)
    assert loaded["speakers"][0]["display_name"] == "Keep Me"


def test_relabel_rejects_transcript_id_with_all(tmp_path, capsys):
    assert main(["--db", str(tmp_path / "undertone.db"), "relabel", "meeting-1", "--all"]) == 1
    assert "either a transcript id or --all" in capsys.readouterr().err


def test_schema_command_and_progress_jsonl(tmp_path, capsys):
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
                        "segment_id": "seg1",
                        "speaker_id": "S1",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "text": "progress",
                        "words": [
                            {
                                "text": "progress",
                                "start_ms": 0,
                                "end_ms": 1000,
                                "confidence": 0.5,
                            }
                        ],
                    }
                ],
            }
        )
    )

    assert main(["schema", "connector-asset"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["properties"]["schema_version"]["const"] == "1"
    assert main(["schema", "transcript"]) == 0
    transcript_schema = json.loads(capsys.readouterr().out)
    assert "FingerprintMatch" in transcript_schema["$defs"]

    assert (
        main(
            [
                "--db",
                str(tmp_path / "undertone.db"),
                "finalize-json",
                str(raw_path),
                "--transcript-id",
                "progress-1",
                "--progress",
                "json",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    events = [json.loads(line)["event"] for line in captured.err.splitlines()]
    assert events == ["start", "finalizing", "saved"]
    payload = json.loads(captured.out)
    assert payload["segments"][0]["asr_confidence"] == 0.5

    assert (
        main(
            [
                "--db",
                str(tmp_path / "undertone.db"),
                "load",
                "progress-1",
                "--output-format",
                "jsonl",
                "--output-detail",
                "standard",
            ]
        )
        == 0
    )
    jsonl_row = json.loads(capsys.readouterr().out)
    assert jsonl_row["asr_confidence"] == 0.5
    assert "diarization_quality" in jsonl_row

    assert (
        main(
            [
                "--db",
                str(tmp_path / "undertone.db"),
                "load",
                "progress-1",
                "--output-format",
                "raw-json",
                "--output-detail",
                "standard",
            ]
        )
        == 0
    )
    raw_payload = json.loads(capsys.readouterr().out)
    assert raw_payload["segments"][0]["asr_confidence"] == 0.5
    assert "diarization_quality" in raw_payload["segments"][0]

    assert (
        main(
            [
                "--db",
                str(tmp_path / "undertone.db"),
                "reenrich",
                "progress-1",
                "--progress",
                "json",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    events = [json.loads(line)["event"] for line in captured.err.splitlines()]
    assert events == ["start", "finalizing", "saved"]
    assert json.loads(captured.out)["transcript_id"] == "progress-1"

    assert (
        main(
            [
                "--db",
                str(tmp_path / "undertone.db"),
                "reenrich",
                "missing",
                "--progress",
                "json",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["event"] == "error"
    assert "transcript not found: missing" in error["error"]


def test_progress_json_defaults_to_error_logging(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "undertone_audio.cli.logging.basicConfig",
        lambda **kwargs: calls.append(kwargs),
    )

    _configure_logging(Namespace(quiet=False, verbose=False, progress="json"))

    assert calls[0]["level"] == logging.ERROR
    assert calls[0]["force"] is True


def test_progress_json_errors_are_jsonl(tmp_path, capsys):
    assert (
        main(
            [
                "--db",
                str(tmp_path / "undertone.db"),
                "run-wav",
                str(tmp_path / "missing.wav"),
                "--transcript-id",
                "missing",
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
    assert payload["error_type"] == "ValueError"
    assert "audio file not found" in payload["error"]


def test_configure_logging_resets_between_in_process_calls():
    root = logging.getLogger()
    original_level = root.level
    original_handlers = root.handlers[:]
    try:
        _configure_logging(Namespace(quiet=False, verbose=False, progress="json"))
        assert root.level == logging.ERROR

        _configure_logging(Namespace(quiet=False, verbose=True, progress="off"))
        assert root.level == logging.INFO
    finally:
        logging.basicConfig(level=original_level, handlers=original_handlers, force=True)


def test_fingerprint_export_and_merge_require_confirmation_and_backup(tmp_path, capsys):
    db = tmp_path / "undertone.db"
    created_by_transcript = {}
    for transcript_id, embedding in [
        ("merge-source", [1.0, 0.0]),
        ("merge-target", [0.0, 1.0]),
    ]:
        raw_path = tmp_path / f"{transcript_id}.json"
        raw_path.write_text(
            json.dumps(
                {
                    "duration_ms": 16000,
                    "language": "en",
                    "engine": "fixture",
                    "speakers": [{"speaker_id": "S1", "embedding": embedding}],
                    "segments": [
                        {
                            "segment_id": "seg1",
                            "speaker_id": "S1",
                            "start_ms": 0,
                            "end_ms": 16000,
                            "text": transcript_id,
                        }
                    ],
                }
            )
        )
        assert (
            main(["--db", str(db), "finalize-json", str(raw_path), "--transcript-id", transcript_id])
            == 0
        )
        created_by_transcript[transcript_id] = json.loads(capsys.readouterr().out)["speakers"][0][
            "fingerprint_id"
        ]

    assert main(["--db", str(db), "fingerprints", "--format", "json"]) == 0
    fingerprints = json.loads(capsys.readouterr().out)
    source_id = created_by_transcript["merge-source"]
    target_id = created_by_transcript["merge-target"]
    assert {row["fingerprint_id"] for row in fingerprints} == {source_id, target_id}
    assert main(["--db", str(db), "fingerprint-label", target_id, "Canonical"]) == 0
    capsys.readouterr()

    export_path = tmp_path / "fingerprints.json"
    assert main(["--db", str(db), "fingerprint-export", "--output", str(export_path)]) == 0
    exported = json.loads(export_path.read_text())
    assert exported["schema_version"] == "1"
    assert "db_path" not in exported
    assert len(exported["fingerprints"]) == 2

    assert main(["--db", str(db), "fingerprint-merge", source_id, target_id]) == 1
    assert "requires --yes or --dry-run" in capsys.readouterr().err
    assert (
        main(["--db", str(db), "fingerprint-merge", source_id, target_id, "--dry-run", "--json"])
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["speaker_rows_to_repoint"] == 1

    assert (
        main(["--db", str(db), "fingerprint-merge", source_id, target_id, "--yes", "--json"])
        == 0
    )
    written = json.loads(capsys.readouterr().out)
    assert written["dry_run"] is False
    assert Path(written["backup_path"]).exists()
    assert "_000000Z" not in Path(written["backup_path"]).name

    assert main(["--db", str(db), "load", "merge-source"]) == 0
    loaded = json.loads(capsys.readouterr().out)
    assert loaded["speakers"][0]["fingerprint_id"] == target_id
    assert loaded["speakers"][0]["display_name"] == "Canonical"


def test_fingerprint_merge_preserves_source_name_when_target_is_unnamed(tmp_path, capsys):
    db = tmp_path / "undertone.db"
    created_by_transcript = {}
    for transcript_id, embedding in [
        ("named-source", [1.0, 0.0]),
        ("unnamed-target", [0.0, 1.0]),
    ]:
        raw_path = tmp_path / f"{transcript_id}.json"
        raw_path.write_text(
            json.dumps(
                {
                    "duration_ms": 16000,
                    "language": "en",
                    "engine": "fixture",
                    "speakers": [{"speaker_id": "S1", "embedding": embedding}],
                    "segments": [
                        {
                            "segment_id": "seg1",
                            "speaker_id": "S1",
                            "start_ms": 0,
                            "end_ms": 16000,
                            "text": transcript_id,
                        }
                    ],
                }
            )
        )
        assert (
            main(["--db", str(db), "finalize-json", str(raw_path), "--transcript-id", transcript_id])
            == 0
        )
        created_by_transcript[transcript_id] = json.loads(capsys.readouterr().out)["speakers"][0][
            "fingerprint_id"
        ]
    source_id = created_by_transcript["named-source"]
    target_id = created_by_transcript["unnamed-target"]
    assert main(["--db", str(db), "fingerprint-label", source_id, "Source Name"]) == 0
    capsys.readouterr()

    assert (
        main(["--db", str(db), "fingerprint-merge", source_id, target_id, "--yes", "--json"])
        == 0
    )
    written = json.loads(capsys.readouterr().out)
    assert written["target_display_name"] == "Source Name"
    assert main(["--db", str(db), "load", "named-source"]) == 0
    loaded = json.loads(capsys.readouterr().out)
    assert loaded["speakers"][0]["display_name"] == "Source Name"
    assert main(["--db", str(db), "load", "unnamed-target"]) == 0
    target_loaded = json.loads(capsys.readouterr().out)
    assert target_loaded["speakers"][0]["display_name"] == "Source Name"


def test_fingerprint_discard_restore_destroy_cli_status_and_backups(tmp_path, capsys):
    db = tmp_path / "undertone.db"
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(
        json.dumps(
            {
                "duration_ms": 16000,
                "language": "en",
                "engine": "fixture",
                "speakers": [{"speaker_id": "S1", "embedding": [1.0, 0.0]}],
                "segments": [
                    {
                        "segment_id": "seg1",
                        "speaker_id": "S1",
                        "start_ms": 0,
                        "end_ms": 16000,
                        "text": "fingerprint action target",
                    }
                ],
            }
        )
    )
    assert main(["--db", str(db), "finalize-json", str(raw_path), "--transcript-id", "fp-actions"]) == 0
    fingerprint_id = json.loads(capsys.readouterr().out)["speakers"][0]["fingerprint_id"]

    assert (
        main(
            [
                "--db",
                str(db),
                "fingerprint-discard",
                fingerprint_id,
                "--reason",
                "mixed speaker",
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )
    discard_plan = json.loads(capsys.readouterr().out)
    assert discard_plan["dry_run"] is True
    assert discard_plan["will_write"] is True
    assert discard_plan["target_status"] == "discarded"
    assert len(list(tmp_path.glob("undertone.db.*.bak"))) == 0

    assert (
        main(
            [
                "--db",
                str(db),
                "fingerprint-discard",
                fingerprint_id,
                "--reason",
                "mixed speaker",
                "--yes",
                "--json",
            ]
        )
        == 0
    )
    discarded = json.loads(capsys.readouterr().out)
    assert discarded["dry_run"] is False
    assert Path(discarded["backup_path"]).exists()

    assert main(["--db", str(db), "fingerprints", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []
    assert main(["--db", str(db), "fingerprints", "--status", "all", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["status"] == "discarded"
    assert rows[0]["discard_reason"] == "mixed speaker"

    assert (
        main(["--db", str(db), "fingerprint-restore", fingerprint_id, "--yes", "--json"])
        == 0
    )
    restored = json.loads(capsys.readouterr().out)
    assert restored["target_status"] == "active"
    assert restored["target_discard_reason"] is None
    assert Path(restored["backup_path"]).exists()

    assert (
        main(["--db", str(db), "fingerprint-restore", fingerprint_id, "--yes", "--json"])
        == 0
    )
    restore_noop = json.loads(capsys.readouterr().out)
    assert restore_noop["will_write"] is False
    assert "backup_path" not in restore_noop

    assert (
        main(["--db", str(db), "fingerprint-destroy", fingerprint_id, "--dry-run", "--json"])
        == 0
    )
    destroy_plan = json.loads(capsys.readouterr().out)
    assert destroy_plan["fingerprint_source_rows"] == 1
    assert destroy_plan["speaker_rows_referencing"] == 1
    assert destroy_plan["will_write"] is True

    assert main(["--db", str(db), "fingerprint-destroy", fingerprint_id, "--yes", "--json"]) == 0
    destroyed = json.loads(capsys.readouterr().out)
    assert Path(destroyed["backup_path"]).exists()
    assert main(["--db", str(db), "fingerprints", "--status", "all", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []

    from undertone_audio.storage import TranscriptStore

    store = TranscriptStore(db)
    try:
        source_count = store._conn.execute("SELECT COUNT(*) FROM fingerprint_sources").fetchone()[0]
        speaker_fingerprint = store._conn.execute(
            "SELECT fingerprint_id FROM speakers WHERE transcript_id = 'fp-actions'"
        ).fetchone()[0]
    finally:
        store.close()
    assert source_count == 0
    assert speaker_fingerprint == fingerprint_id


def test_fingerprint_action_dry_runs_plan_against_legacy_db_without_migrating_it(tmp_path, capsys):
    import sqlite3

    db = tmp_path / "legacy-actions.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """CREATE TABLE speaker_fingerprints (
                fingerprint_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                display_name TEXT,
                sample_count INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        conn.execute(
            "INSERT INTO speaker_fingerprints (fingerprint_id, embedding, sample_count) VALUES (?, ?, 1)",
            ("VP-legacy", b"\x00" * 8),
        )

    assert (
        main(
            [
                "--db",
                str(db),
                "fingerprint-discard",
                "VP-legacy",
                "--reason",
                "old db probe",
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )
    discard_plan = json.loads(capsys.readouterr().out)
    assert discard_plan["current_status"] == "active"
    assert discard_plan["target_status"] == "discarded"

    assert (
        main(["--db", str(db), "fingerprint-restore", "VP-legacy", "--dry-run", "--json"])
        == 0
    )
    restore_plan = json.loads(capsys.readouterr().out)
    assert restore_plan["will_write"] is False

    assert (
        main(["--db", str(db), "fingerprint-destroy", "VP-legacy", "--dry-run", "--json"])
        == 0
    )
    destroy_plan = json.loads(capsys.readouterr().out)
    assert destroy_plan["will_write"] is True

    with sqlite3.connect(db) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(speaker_fingerprints)")}
        row_count = conn.execute("SELECT COUNT(*) FROM speaker_fingerprints").fetchone()[0]
    assert "status" not in columns
    assert "discard_reason" not in columns
    assert row_count == 1
    assert list(tmp_path.glob("legacy-actions.db.*.bak")) == []


def test_fingerprint_export_import_preserves_discarded_status(tmp_path, capsys):
    db = tmp_path / "undertone.db"
    path = tmp_path / "fingerprints.json"
    path.write_text(
        json.dumps(
            {
                "fingerprints": [
                    {
                        "fingerprint_id": "VP-discarded",
                        "embedding": [1.0, 0.0],
                        "status": "discarded",
                        "discard_reason": "imported bad print",
                    }
                ]
            }
        )
    )

    assert main(["--db", str(db), "fingerprint-import", str(path), "--yes", "--json"]) == 0
    capsys.readouterr()
    assert main(["--db", str(db), "fingerprint-export"]) == 0
    exported = json.loads(capsys.readouterr().out)["fingerprints"][0]
    assert exported["status"] == "discarded"
    assert exported["discard_reason"] == "imported bad print"

    assert main(["--db", str(db), "fingerprints", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []
    assert main(["--db", str(db), "fingerprints", "--status", "discarded", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["fingerprint_id"] == "VP-discarded"


def test_fingerprint_status_counts_surface_in_operator_commands(tmp_path, capsys):
    db = tmp_path / "undertone.db"
    path = tmp_path / "fingerprints.json"
    path.write_text(
        json.dumps(
            {
                "fingerprints": [
                    {"fingerprint_id": "VP-active", "embedding": [1.0, 0.0]},
                    {
                        "fingerprint_id": "VP-discarded",
                        "embedding": [0.0, 1.0],
                        "status": "discarded",
                        "discard_reason": "bad print",
                    },
                ]
            }
        )
    )
    assert main(["--db", str(db), "fingerprint-import", str(path), "--yes"]) == 0
    capsys.readouterr()

    assert main(["--db", str(db), "stats", "--json"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["fingerprint_status"]["active"] == 1
    assert stats["fingerprint_status"]["discarded"] == 1
    assert stats["fingerprint_models"]["legacy"] == 1

    assert main(["--db", str(db), "models", "--json"]) == 0
    models = json.loads(capsys.readouterr().out)
    assert models["fingerprint_status"]["active"] == 1
    assert models["fingerprint_status"]["discarded"] == 1
    assert models["fingerprint_models"]["legacy"] == 1

    assert main(["--db", str(db), "doctor", "--json"]) == 1
    doctor = json.loads(capsys.readouterr().out)
    status_check = next(check for check in doctor["checks"] if check["name"] == "fingerprint_status")
    assert status_check["active"] == 1
    assert status_check["discarded"] == 1
    model_check = next(check for check in doctor["checks"] if check["name"] == "fingerprint_models")
    assert model_check["legacy"] == 1


def test_discarded_legacy_fingerprint_does_not_fail_model_compatibility(tmp_path, capsys, monkeypatch):
    class FakeEngine:
        cli_path = "/tmp/fluidaudiocli"

        async def healthcheck(self):
            return True

    monkeypatch.setattr(
        "undertone_audio.commands.ops.create_engine", lambda name, config: FakeEngine()
    )
    db = tmp_path / "undertone.db"
    path = tmp_path / "fingerprints.json"
    path.write_text(
        json.dumps(
            {
                "fingerprints": [
                    {
                        "fingerprint_id": "VP-discarded",
                        "embedding": [0.0, 1.0],
                        "status": "discarded",
                        "discard_reason": "bad print",
                    },
                ]
            }
        )
    )
    assert main(["--db", str(db), "fingerprint-import", str(path), "--yes"]) == 0
    capsys.readouterr()

    assert main(["--db", str(db), "doctor", "--json"]) == 0
    doctor = json.loads(capsys.readouterr().out)
    model_check = next(check for check in doctor["checks"] if check["name"] == "fingerprint_models")
    status_check = next(check for check in doctor["checks"] if check["name"] == "fingerprint_status")
    assert model_check["legacy"] == 0
    assert status_check["discarded"] == 1


def test_fingerprint_import_dry_run_validates_rows(tmp_path, capsys):
    bad = tmp_path / "bad-fingerprints.json"
    bad.write_text(json.dumps({"fingerprints": [{"fingerprint_id": "VP-bad"}]}))

    assert (
        main(
            [
                "--db",
                str(tmp_path / "undertone.db"),
                "fingerprint-import",
                str(bad),
                "--dry-run",
            ]
        )
        == 1
    )
    assert "embedding must be a non-empty number list" in capsys.readouterr().err

    bad.write_text(json.dumps([]))
    assert (
        main(
            [
                "--db",
                str(tmp_path / "undertone.db"),
                "fingerprint-import",
                str(bad),
                "--dry-run",
            ]
        )
        == 1
    )
    assert "must be an object with a fingerprints array" in capsys.readouterr().err

    dup = tmp_path / "dup-fingerprints.json"
    dup.write_text(
        json.dumps(
            {
                "fingerprints": [
                    {"fingerprint_id": "VP-dup", "embedding": [1.0, 0.0]},
                    {"fingerprint_id": "VP-dup", "embedding": [0.0, 1.0]},
                ]
            }
        )
    )
    assert (
        main(
            [
                "--db",
                str(tmp_path / "undertone.db"),
                "fingerprint-import",
                str(dup),
                "--dry-run",
            ]
        )
        == 1
    )
    assert "duplicate fingerprint_id in import file: VP-dup" in capsys.readouterr().err

    db = tmp_path / "invalid-write.db"
    assert (
        main(
            [
                "--db",
                str(db),
                "fingerprint-import",
                str(dup),
                "--yes",
            ]
        )
        == 1
    )
    assert "duplicate fingerprint_id in import file: VP-dup" in capsys.readouterr().err
    assert list(tmp_path.glob("invalid-write.db.*.bak")) == []


def test_fingerprint_merge_rejected_write_does_not_create_backup(tmp_path, capsys):
    db = tmp_path / "undertone.db"

    assert (
        main(
            [
                "--db",
                str(db),
                "fingerprint-merge",
                "VP-missing-source",
                "VP-missing-target",
                "--yes",
            ]
        )
        == 1
    )
    assert "source fingerprint not found" in capsys.readouterr().err
    assert list(tmp_path.glob("undertone.db.*.bak")) == []


def test_fingerprint_import_backups_do_not_collide(tmp_path, capsys):
    db = tmp_path / "undertone.db"
    for fingerprint_id, embedding in [
        ("VP-one", [1.0, 0.0]),
        ("VP-two", [0.0, 1.0]),
    ]:
        path = tmp_path / f"{fingerprint_id}.json"
        path.write_text(
            json.dumps(
                {
                    "fingerprints": [
                        {
                            "fingerprint_id": fingerprint_id,
                            "embedding": embedding,
                            "sample_count": 1,
                        }
                    ]
                }
            )
        )
        assert (
            main(
                [
                    "--db",
                    str(db),
                    "fingerprint-import",
                    str(path),
                    "--yes",
                    "--json",
                ]
            )
            == 0
        )
        backup_path = Path(json.loads(capsys.readouterr().out)["backup_path"])
        assert backup_path.exists()

    backups = sorted(tmp_path.glob("undertone.db.*.bak"))
    assert len(backups) == 2
    assert len({backup.name for backup in backups}) == 2


def test_fingerprint_import_noop_does_not_create_backup(tmp_path, capsys):
    db = tmp_path / "undertone.db"
    path = tmp_path / "fingerprints.json"
    path.write_text(
        json.dumps(
            {
                "fingerprints": [
                    {
                        "fingerprint_id": "VP-one",
                        "embedding": [1.0, 0.0],
                        "sample_count": 1,
                    }
                ]
            }
        )
    )

    assert (
        main(
            [
                "--db",
                str(db),
                "fingerprint-import",
                str(path),
                "--yes",
                "--json",
            ]
        )
        == 0
    )
    first = json.loads(capsys.readouterr().out)
    assert Path(first["backup_path"]).exists()
    assert len(list(tmp_path.glob("undertone.db.*.bak"))) == 1

    assert (
        main(
            [
                "--db",
                str(db),
                "fingerprint-import",
                str(path),
                "--yes",
                "--json",
            ]
        )
        == 0
    )
    second = json.loads(capsys.readouterr().out)
    assert second["dry_run"] is False
    assert second["to_insert"] == []
    assert second["to_replace"] == []
    assert second["skipped_existing"] == ["VP-one"]
    assert "backup_path" not in second
    assert len(list(tmp_path.glob("undertone.db.*.bak"))) == 1


def test_fingerprint_import_round_trips_exported_timestamps(tmp_path, capsys):
    db = tmp_path / "undertone.db"
    restore = tmp_path / "restore.json"
    restore.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "fingerprints": [
                    {
                        "fingerprint_id": "VP-one",
                        "embedding": [1.0, 0.0],
                        "display_name": "Original",
                        "sample_count": 2,
                        "created_at": "2026-06-01 10:00:00",
                        "updated_at": "2026-06-02 11:00:00",
                    }
                ],
            }
        )
    )

    assert (
        main(
            [
                "--db",
                str(db),
                "fingerprint-import",
                str(restore),
                "--yes",
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["--db", str(db), "fingerprint-export"]) == 0
    exported = json.loads(capsys.readouterr().out)["fingerprints"][0]
    assert exported["created_at"] == "2026-06-01 10:00:00"
    assert exported["updated_at"] == "2026-06-02 11:00:00"

    restore.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "fingerprints": [
                    {
                        "fingerprint_id": "VP-one",
                        "embedding": [0.0, 1.0],
                        "display_name": "Replacement",
                        "sample_count": 3,
                        "created_at": "2026-06-03 12:00:00",
                        "updated_at": "2026-06-04 13:00:00",
                    }
                ],
            }
        )
    )
    assert (
        main(
            [
                "--db",
                str(db),
                "fingerprint-import",
                str(restore),
                "--replace",
                "--yes",
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["--db", str(db), "fingerprint-export"]) == 0
    replaced = json.loads(capsys.readouterr().out)["fingerprints"][0]
    assert replaced["display_name"] == "Replacement"
    assert replaced["sample_count"] == 3
    assert replaced["created_at"] == "2026-06-03 12:00:00"
    assert replaced["updated_at"] == "2026-06-04 13:00:00"

    bad = tmp_path / "bad-schema.json"
    bad.write_text(json.dumps({"schema_version": "2", "fingerprints": []}))
    assert main(["--db", str(db), "fingerprint-import", str(bad), "--dry-run"]) == 1
    assert "schema_version must be 1" in capsys.readouterr().err


def test_fingerprint_import_replace_relabels_saved_transcript_speakers(tmp_path, capsys):
    db = tmp_path / "undertone.db"
    initial = tmp_path / "initial-fingerprint.json"
    initial.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "fingerprints": [
                    {
                        "fingerprint_id": "VP-one",
                        "embedding": [1.0, 0.0],
                        "display_name": "Old Name",
                    }
                ],
            }
        )
    )
    assert main(["--db", str(db), "fingerprint-import", str(initial), "--yes"]) == 0
    capsys.readouterr()

    raw_path = tmp_path / "raw.json"
    raw_path.write_text(
        json.dumps(
            {
                "duration_ms": 1000,
                "language": "en",
                "engine": "fixture",
                "speakers": [
                    {
                        "speaker_id": "S1",
                        "fingerprint_id": "VP-one",
                        "display_name": "Old Name",
                    }
                ],
                "segments": [
                    {
                        "segment_id": "seg1",
                        "speaker_id": "S1",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "text": "rename imported speaker",
                    }
                ],
            }
        )
    )
    assert main(["--db", str(db), "finalize-json", str(raw_path), "--transcript-id", "rename-1"]) == 0
    capsys.readouterr()

    replacement = tmp_path / "replacement-fingerprint.json"
    replacement.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "fingerprints": [
                    {
                        "fingerprint_id": "VP-one",
                        "embedding": [0.0, 1.0],
                        "display_name": "New Name",
                    }
                ],
            }
        )
    )
    assert (
        main(
            [
                "--db",
                str(db),
                "fingerprint-import",
                str(replacement),
                "--replace",
                "--yes",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["--db", str(db), "load", "rename-1"]) == 0
    loaded = json.loads(capsys.readouterr().out)
    assert loaded["speakers"][0]["display_name"] == "New Name"


def test_backup_path_does_not_collide_when_timestamp_is_identical(tmp_path, monkeypatch):
    class FrozenDateTime:
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 6, 27, 12, 0, 0, 123456, tzinfo=timezone.utc)

    db = tmp_path / "undertone.db"
    monkeypatch.setattr(fingerprint_commands, "datetime", FrozenDateTime)

    first = fingerprint_commands._backup_db(db)
    second = fingerprint_commands._backup_db(db)

    assert first != second
    assert first.exists()
    assert second.exists()


def test_cli_duplicate_controls_for_finalize_json(tmp_path, capsys):
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
                        "segment_id": "seg1",
                        "speaker_id": "S1",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "text": "first",
                    }
                ],
            }
        )
    )
    db = tmp_path / "undertone.db"

    assert main(["--db", str(db), "finalize-json", str(raw_path), "--transcript-id", "dup"]) == 0
    capsys.readouterr()
    assert main(["--db", str(db), "finalize-json", str(raw_path), "--transcript-id", "dup"]) == 1
    assert "already exists" in capsys.readouterr().err
    missing_raw = tmp_path / "missing.json"
    assert (
        main(
            [
                "--db",
                str(db),
                "finalize-json",
                str(missing_raw),
                "--transcript-id",
                "dup",
                "--skip-existing",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["skipped"] is True
    missing_audio = tmp_path / "missing.wav"
    assert (
        main(
            [
                "--db",
                str(db),
                "run-wav",
                str(missing_audio),
                "--transcript-id",
                "dup",
                "--skip-existing",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["skipped"] is True
    assert (
        main(
            [
                "--db",
                str(db),
                "finalize-json",
                str(raw_path),
                "--transcript-id",
                "dup",
                "--skip-existing",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["skipped"] is True
    output_path = tmp_path / "skipped-output.json"
    assert (
        main(
            [
                "--db",
                str(db),
                "finalize-json",
                str(raw_path),
                "--transcript-id",
                "dup",
                "--skip-existing",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["skipped"] is True
    assert not output_path.exists()
    assert (
        main(
            [
                "--db",
                str(db),
                "finalize-json",
                str(raw_path),
                "--transcript-id",
                "dup",
                "--skip-existing",
                "--output",
                str(output_path),
                "--progress",
                "json",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    events = [json.loads(line)["event"] for line in captured.err.splitlines()]
    assert events == ["skipped"]
    assert (
        main(
            [
                "--db",
                str(db),
                "finalize-json",
                str(raw_path),
                "--transcript-id",
                "dup",
                "--force",
            ]
        )
        == 0
    )


def test_cli_text_signature_does_not_skip_different_transcript_id(tmp_path, capsys):
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
                        "segment_id": "seg1",
                        "speaker_id": "S1",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "text": "same meeting agenda decisions blockers and next steps",
                    }
                ],
            }
        )
    )
    db = tmp_path / "undertone.db"

    assert main(["--db", str(db), "finalize-json", str(raw_path), "--transcript-id", "source-a"]) == 0
    capsys.readouterr()
    assert main(["--db", str(db), "finalize-json", str(raw_path), "--transcript-id", "source-b"]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["transcript_id"] == "source-b"
    assert second["metadata"]["content_text_simhash"]

    assert (
        main(
            [
                "--db",
                str(db),
                "finalize-json",
                str(raw_path),
                "--transcript-id",
                "source-c",
                "--progress",
                "json",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out)["transcript_id"] == "source-c"
    events = [json.loads(line) for line in captured.err.splitlines()]
    assert events[-1]["event"] == "saved"
    assert events[-1]["transcript_id"] == "source-c"


def test_cli_finalize_json_audio_duplicate_uses_source_path(tmp_path, monkeypatch, capsys):
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
                        "segment_id": "seg1",
                        "speaker_id": "S1",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "text": "audio duplicate source path",
                    }
                ],
            }
        )
    )
    db = tmp_path / "undertone.db"
    store = TranscriptStore(db)
    try:
        store.save(
            EnrichedTranscript(
                transcript_id="existing-audio",
                metadata=TranscriptMetadata(
                    duration_ms=1000,
                    engine="fixture",
                    content_audio_fp="audio-fp",
                    content_audio_fp_algorithm="chromaprint-fpcalc-v1",
                ),
                speakers=[Speaker(speaker_id="S1")],
                segments=[
                    Segment(
                        segment_id="seg1",
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

    class FakeSignature:
        value = "audio-fp"
        algorithm = "chromaprint-fpcalc-v1"

    monkeypatch.setattr("undertone_audio.commands.common.audio_signature_for_path", lambda *a, **k: FakeSignature())
    audio = tmp_path / "fixture.m4a"
    audio.write_bytes(b"audio")

    assert (
        main(
            [
                "--db",
                str(db),
                "finalize-json",
                str(raw_path),
                "--transcript-id",
                "new-audio",
                "--source-path",
                str(audio),
            ]
        )
        == 0
    )
    duplicate = json.loads(capsys.readouterr().out)
    assert duplicate["skipped"] is True
    assert duplicate["existing_transcript_id"] == "existing-audio"
    assert duplicate["match_type"] == "audio"

    assert (
        main(
            [
                "--db",
                str(db),
                "finalize-json",
                str(raw_path),
                "--transcript-id",
                "new-audio",
                "--source-path",
                str(audio),
                "--allow-duplicate",
            ]
        )
        == 0
    )
    transcript = json.loads(capsys.readouterr().out)
    assert transcript["metadata"]["content_audio_fp"] == "audio-fp"


def test_cli_run_wav_audio_duplicate_skips_before_engine(tmp_path, monkeypatch, capsys):
    db = tmp_path / "undertone.db"
    store = TranscriptStore(db)
    try:
        store.save(
            EnrichedTranscript(
                transcript_id="existing-audio",
                metadata=TranscriptMetadata(
                    duration_ms=1000,
                    engine="fixture",
                    content_audio_fp="audio-fp",
                    content_audio_fp_algorithm="chromaprint-fpcalc-v1",
                ),
                speakers=[Speaker(speaker_id="S1")],
                segments=[
                    Segment(
                        segment_id="seg1",
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

    class FakeSignature:
        value = "audio-fp"
        algorithm = "chromaprint-fpcalc-v1"

    def fail_create_engine(*args, **kwargs):
        raise AssertionError("duplicate audio should skip before engine construction")

    monkeypatch.setattr("undertone_audio.commands.common.audio_signature_for_path", lambda *a, **k: FakeSignature())
    monkeypatch.setattr("undertone_audio.commands.core.create_engine", fail_create_engine)
    monkeypatch.setenv("UNDERTONE_WEBHOOK_ENABLED", "0")
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"not a real wav")

    assert main(["--db", str(db), "run-wav", str(audio)]) == 0
    duplicate = json.loads(capsys.readouterr().out)
    assert duplicate["skipped"] is True
    assert duplicate["reason"] == "duplicate"
    assert duplicate["transcript_id"]
    assert duplicate["existing_transcript_id"] == "existing-audio"
    assert duplicate["match_type"] == "audio"


def test_cli_reports_missing_transcript(tmp_path, capsys):
    assert main(["--db", str(tmp_path / "undertone.db"), "load", "missing"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "transcript not found: missing" in captured.err


def test_cli_run_wav_uses_engine_and_assigns_fingerprint(tmp_path, monkeypatch, capsys):
    class FakeEngine:
        name = "fake"

        async def healthcheck(self):
            return True

        async def transcribe(self, audio_path: Path):
            assert audio_path.name == "fixture.wav"
            return RawTranscript(
                duration_ms=16000,
                language="en",
                engine="fluidaudio-hybrid",
                speakers=[Speaker(speaker_id="S1", embedding=[1.0, 0.0])],
                segments=[
                    Segment(
                        segment_id="seg1",
                        speaker_id="S1",
                        start_ms=0,
                        end_ms=16000,
                        text="wav path works",
                    )
                ],
            )

    def fake_create_engine(name, config):
        assert name == "fluidaudio-cli"
        return FakeEngine()

    monkeypatch.setattr("undertone_audio.commands.core.create_engine", fake_create_engine)
    monkeypatch.setenv("UNDERTONE_WEBHOOK_ENABLED", "0")
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"not a real wav")
    db = tmp_path / "undertone.db"

    assert (
        main(
            [
                "--db",
                str(db),
                "run-wav",
                str(audio),
                "--engine",
                "fluidaudio-cli",
                "--transcript-id",
                "wav-1",
            ]
        )
        == 0
    )
    transcript = json.loads(capsys.readouterr().out)
    assert transcript["transcript_id"] == "wav-1"
    assert transcript["metadata"]["asr_backend"] == "FluidAudio Parakeet TDT"
    assert transcript["metadata"]["audio_format"]["parse_error"] == "invalid-wav"
    assert transcript["speakers"][0]["fingerprint_id"].startswith("VP-")

    assert main(["--db", str(db), "fingerprints", "--json"]) == 0
    fingerprints = json.loads(capsys.readouterr().out)
    assert fingerprints[0]["sample_count"] == 1
    assert fingerprints[0]["embedding_model"] == "FluidAudio pyannote-derived speaker embeddings"

    from undertone_audio.storage import TranscriptStore

    store = TranscriptStore(db)
    try:
        sources = store._conn.execute(
            "SELECT fingerprint_id, transcript_id, speaker_id FROM fingerprint_sources"
        ).fetchall()
    finally:
        store.close()
    assert [(row["transcript_id"], row["speaker_id"]) for row in sources] == [("wav-1", "S1")]


def test_fingerprint_model_namespace_prevents_same_dimension_cross_model_match(
    tmp_path, monkeypatch, capsys
):
    db = tmp_path / "undertone.db"
    for transcript_id, model in [("model-a", "model-a"), ("model-b", "model-b")]:
        monkeypatch.setenv("UNDERTONE_EMBEDDING_MODEL", model)
        raw_path = tmp_path / f"{transcript_id}.json"
        raw_path.write_text(
            json.dumps(
                {
                    "duration_ms": 16000,
                    "language": "en",
                    "engine": "fixture",
                    "speakers": [{"speaker_id": "S1", "embedding": [1.0, 0.0]}],
                    "segments": [
                        {
                            "segment_id": "seg1",
                            "speaker_id": "S1",
                            "start_ms": 0,
                            "end_ms": 16000,
                            "text": transcript_id,
                        }
                    ],
                }
            )
        )
        assert (
            main(
                [
                    "--db",
                    str(db),
                    "finalize-json",
                    str(raw_path),
                    "--transcript-id",
                    transcript_id,
                ]
            )
            == 0
        )
        capsys.readouterr()

    assert main(["--db", str(db), "load", "model-a"]) == 0
    first = json.loads(capsys.readouterr().out)["speakers"][0]["fingerprint_id"]
    assert main(["--db", str(db), "load", "model-b"]) == 0
    second = json.loads(capsys.readouterr().out)["speakers"][0]["fingerprint_id"]
    assert first != second

    assert main(["--db", str(db), "fingerprints", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {row["embedding_model"] for row in rows} == {"model-a", "model-b"}


def test_finalize_json_uses_raw_engine_for_fingerprint_model_namespace(tmp_path, capsys):
    db = tmp_path / "undertone.db"
    raw_path = tmp_path / "pyannote-raw.json"
    raw_path.write_text(
        json.dumps(
            {
                "duration_ms": 16000,
                "language": "en",
                "engine": "fluidaudio-pyannote",
                "speakers": [{"speaker_id": "S1", "embedding": [1.0, 0.0]}],
                "segments": [
                    {
                        "segment_id": "seg1",
                        "speaker_id": "S1",
                        "start_ms": 0,
                        "end_ms": 16000,
                        "text": "pyannote raw",
                    }
                ],
            }
        )
    )

    assert main(["--db", str(db), "finalize-json", str(raw_path), "--transcript-id", "pyannote"]) == 0
    transcript = json.loads(capsys.readouterr().out)
    assert transcript["metadata"]["embedding_backend"] == "pyannote/speaker-diarization-community-1"

    assert main(["--db", str(db), "fingerprints", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["embedding_model"] == "pyannote/speaker-diarization-community-1"


def test_finalize_json_prefers_raw_embedded_embedding_model(tmp_path, capsys):
    db = tmp_path / "undertone.db"
    raw_path = tmp_path / "external-raw.json"
    raw_path.write_text(
        json.dumps(
            {
                "duration_ms": 16000,
                "language": "en",
                "engine": "fluidaudio-pyannote",
                "model_versions": {"embedding": "external/pyannote-model"},
                "speakers": [{"speaker_id": "S1", "embedding": [1.0, 0.0]}],
                "segments": [
                    {
                        "segment_id": "seg1",
                        "speaker_id": "S1",
                        "start_ms": 0,
                        "end_ms": 16000,
                        "text": "external raw",
                    }
                ],
            }
        )
    )

    assert (
        main(
            [
                "--db",
                str(db),
                "finalize-json",
                str(raw_path),
                "--transcript-id",
                "external",
                "--embedding-model",
                "local-fallback-model",
            ]
        )
        == 0
    )
    transcript = json.loads(capsys.readouterr().out)
    assert transcript["metadata"]["embedding_backend"] == "external/pyannote-model"
    assert transcript["metadata"]["model_versions"]["embedding"] == "external/pyannote-model"

    assert main(["--db", str(db), "fingerprints", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["embedding_model"] == "external/pyannote-model"


def test_finalize_json_embedding_model_flag_populates_model_versions(tmp_path, capsys):
    db = tmp_path / "undertone.db"
    raw_path = tmp_path / "flag-raw.json"
    raw_path.write_text(
        json.dumps(
            {
                "duration_ms": 16000,
                "language": "en",
                "engine": "fixture",
                "model_versions": {"embedding": "   ", "asr": "external-asr"},
                "speakers": [{"speaker_id": "S1", "embedding": [1.0, 0.0]}],
                "segments": [
                    {
                        "segment_id": "seg1",
                        "speaker_id": "S1",
                        "start_ms": 0,
                        "end_ms": 16000,
                        "text": "flag raw",
                    }
                ],
            }
        )
    )

    assert (
        main(
            [
                "--db",
                str(db),
                "finalize-json",
                str(raw_path),
                "--transcript-id",
                "flag",
                "--embedding-model",
                "flag-model",
            ]
        )
        == 0
    )
    transcript = json.loads(capsys.readouterr().out)
    assert transcript["metadata"]["embedding_backend"] == "flag-model"
    assert transcript["metadata"]["model_versions"]["embedding"] == "flag-model"
    assert transcript["metadata"]["model_versions"]["asr"] == "external-asr"

    assert main(["--db", str(db), "fingerprints", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["embedding_model"] == "flag-model"


def test_legacy_fingerprints_are_dormant_until_adopted(tmp_path, capsys):
    from undertone_audio.diarization.fingerprint import SpeakerFingerprintStore
    from undertone_audio.storage import TranscriptStore

    db = tmp_path / "undertone.db"
    TranscriptStore(db).close()
    legacy_store = SpeakerFingerprintStore(db, similarity_threshold=0.0)
    speakers, plan = legacy_store.assign_fingerprints(
        [Speaker(speaker_id="S1", embedding=[1.0, 0.0])],
        persist=False,
        speaker_durations_ms={"S1": 16000},
    )
    legacy_id = speakers[0].fingerprint_id
    plan.commit()

    raw_path = tmp_path / "new.json"
    raw_path.write_text(
        json.dumps(
            {
                "duration_ms": 16000,
                "language": "en",
                "engine": "fixture",
                "speakers": [{"speaker_id": "S1", "embedding": [1.0, 0.0]}],
                "segments": [
                    {
                        "segment_id": "seg1",
                        "speaker_id": "S1",
                        "start_ms": 0,
                        "end_ms": 16000,
                        "text": "new model path",
                    }
                ],
            }
        )
    )
    assert (
        main(["--db", str(db), "finalize-json", str(raw_path), "--transcript-id", "new"])
        == 0
    )
    captured = capsys.readouterr()
    assert "legacy and 0 incompatible voice fingerprints are dormant" in captured.err
    new_id = json.loads(captured.out)["speakers"][0]["fingerprint_id"]
    assert new_id != legacy_id

    assert main(["--db", str(db), "doctor", "--json"]) == 1
    doctor = json.loads(capsys.readouterr().out)
    model_check = next(check for check in doctor["checks"] if check["name"] == "fingerprint_models")
    assert model_check["legacy"] == 1
    assert "fingerprint-adopt-model" in model_check["fix"]

    assert main(["--db", str(db), "fingerprint-adopt-model", "--dry-run", "--json"]) == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run["fingerprints_to_update"] == 1
    assert dry_run["dry_run"] is True

    assert main(["--db", str(db), "fingerprint-adopt-model", "--yes", "--json"]) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["dry_run"] is False
    assert Path(applied["backup_path"]).exists()

    assert main(["--db", str(db), "fingerprints", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    legacy = next(row for row in rows if row["fingerprint_id"] == legacy_id)
    assert legacy["embedding_model"] == "FluidAudio pyannote-derived speaker embeddings"

    assert main(["--db", str(db), "stats", "--json"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["fingerprint_models"]["compatible"] == 2
    assert stats["fingerprint_models"]["legacy"] == 0


def test_progress_json_surfaces_dormant_fingerprint_warning(tmp_path, capsys):
    from undertone_audio.diarization.fingerprint import SpeakerFingerprintStore
    from undertone_audio.storage import TranscriptStore

    db = tmp_path / "undertone.db"
    TranscriptStore(db).close()
    legacy_store = SpeakerFingerprintStore(db, similarity_threshold=0.0)
    _speakers, plan = legacy_store.assign_fingerprints(
        [Speaker(speaker_id="S1", embedding=[1.0, 0.0])],
        persist=False,
        speaker_durations_ms={"S1": 16000},
    )
    plan.commit()

    raw_path = tmp_path / "raw.json"
    raw_path.write_text(
        json.dumps(
            {
                "duration_ms": 16000,
                "language": "en",
                "engine": "fixture",
                "speakers": [{"speaker_id": "S1", "embedding": [0.0, 1.0]}],
                "segments": [
                    {
                        "segment_id": "seg1",
                        "speaker_id": "S1",
                        "start_ms": 0,
                        "end_ms": 16000,
                        "text": "progress warning",
                    }
                ],
            }
        )
    )

    assert (
        main(
            [
                "--db",
                str(db),
                "finalize-json",
                str(raw_path),
                "--transcript-id",
                "progress-warning",
                "--progress",
                "json",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    events = [json.loads(line) for line in captured.err.splitlines()]
    warning = next(event for event in events if event["event"] == "warning")
    assert warning["warning"] == "fingerprint_models"
    assert warning["legacy"] == 1
    assert "fingerprint-adopt-model" in warning["fix"]


def test_fingerprint_dry_runs_plan_against_legacy_db_without_migrating_it(tmp_path, capsys):
    import sqlite3

    db = tmp_path / "legacy.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """CREATE TABLE speaker_fingerprints (
                fingerprint_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                display_name TEXT,
                sample_count INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        conn.execute(
            "INSERT INTO speaker_fingerprints (fingerprint_id, embedding, sample_count) VALUES (?, ?, 1)",
            ("VP-legacy", b"\x00" * 8),
        )
        conn.execute(
            "INSERT INTO speaker_fingerprints (fingerprint_id, embedding, sample_count) VALUES (?, ?, 2)",
            ("VP-other", b"\x01" * 8),
        )

    assert main(["--db", str(db), "fingerprint-adopt-model", "--dry-run", "--json"]) == 0
    adopt_plan = json.loads(capsys.readouterr().out)
    assert adopt_plan["fingerprints_to_update"] == 2

    assert (
        main(
            [
                "--db",
                str(db),
                "fingerprint-merge",
                "VP-legacy",
                "VP-other",
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )
    merge_plan = json.loads(capsys.readouterr().out)
    assert merge_plan["source_fingerprint_id"] == "VP-legacy"
    assert merge_plan["target_fingerprint_id"] == "VP-other"
    assert merge_plan["embedding_model"] is None

    import_path = tmp_path / "new-fingerprint.json"
    import_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "fingerprints": [
                    {
                        "fingerprint_id": "VP-new",
                        "embedding": [0.0, 1.0],
                        "sample_count": 1,
                    }
                ],
            }
        )
    )
    assert (
        main(
            [
                "--db",
                str(db),
                "fingerprint-import",
                str(import_path),
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )
    import_plan = json.loads(capsys.readouterr().out)
    assert [row["fingerprint_id"] for row in import_plan["to_insert"]] == ["VP-new"]

    with sqlite3.connect(db) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(speaker_fingerprints)")}
        row_count = conn.execute("SELECT COUNT(*) FROM speaker_fingerprints").fetchone()[0]
    assert "embedding_model" not in columns
    assert "embedding_dimension" not in columns
    assert row_count == 2


def test_cli_run_wav_output_format_and_model_flags(tmp_path, monkeypatch, capsys):
    class FakeEngine:
        name = "fake"

        async def healthcheck(self):
            return True

        async def transcribe(self, audio_path: Path):
            return RawTranscript(
                duration_ms=1000,
                language="en",
                engine="fluidaudio-hybrid",
                speakers=[Speaker(speaker_id="S1", embedding=[1.0, 0.0])],
                segments=[
                    Segment(
                        segment_id="seg1",
                        speaker_id="S1",
                        start_ms=0,
                        end_ms=1000,
                        text="output flags work",
                    )
                ],
            )

    monkeypatch.setattr(
        "undertone_audio.commands.core.create_engine", lambda name, config: FakeEngine()
    )
    monkeypatch.setenv("UNDERTONE_WEBHOOK_ENABLED", "0")
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"not a real wav")
    out = tmp_path / "out.md"

    assert (
        main(
            [
                "--db",
                str(tmp_path / "undertone.db"),
                "run-wav",
                str(audio),
                "--transcript-id",
                "wav-md",
                "--asr-model",
                "custom-asr",
                "--diarization-model",
                "custom-diar",
                "--voice-metrics",
                "off",
                "--output-format",
                "md",
                "--output-detail",
                "standard",
                "--output",
                str(out),
            ]
        )
        == 0
    )

    assert capsys.readouterr().out == ""
    body = out.read_text()
    assert "# Transcript wav-md" in body
    assert "output flags work" in body
    assert "jitter" not in body

    assert main(["--db", str(tmp_path / "undertone.db"), "load", "wav-md"]) == 0
    loaded = json.loads(capsys.readouterr().out)
    assert loaded["metadata"]["asr_backend"] == "custom-asr"
    assert loaded["metadata"]["diarization_backend"] == "custom-diar"


def test_cli_models_reports_effective_backend_selection(tmp_path, capsys):
    assert (
        main(
            [
                "--db",
                str(tmp_path / "undertone.db"),
                "models",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["engine"] == "fluidaudio-hybrid"
    assert payload["asr_model"] == "FluidAudio Parakeet TDT"
    assert payload["output_detail"] == "full"
    assert payload["pyannote_model"] == "pyannote/speaker-diarization-community-1"
    assert payload["pyannote_device"] == "auto"
    assert payload["features"]["linguistic"] is True
    assert payload["thresholds"]["clustering"] == 0.7045655
    assert payload["thresholds"]["fingerprint_similarity"] == 0.78


def test_cli_doctor_reports_checks(tmp_path, monkeypatch, capsys):
    class FakeEngine:
        cli_path = "/tmp/fluidaudiocli"

        async def healthcheck(self):
            return True

    monkeypatch.setattr(
        "undertone_audio.commands.ops.create_engine", lambda name, config: FakeEngine()
    )
    monkeypatch.setattr(
        "undertone_audio.commands.ops.shutil.which", lambda name: f"/usr/bin/{name}"
    )

    assert main(["--db", str(tmp_path / "undertone.db"), "doctor", "--check-yt-dlp", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert {check["name"] for check in payload["checks"]} == {
        "db_writable",
        "engine",
        "fingerprint_models",
        "fingerprint_status",
        "yt_dlp",
    }
    assert {source["source"] for source in payload["sources"]} == {
        "youtube",
        "podcast",
        "meet",
        "quill",
    }
    yt_dlp_check = next(check for check in payload["checks"] if check["name"] == "yt_dlp")
    assert yt_dlp_check["binary"] == "yt-dlp"
    assert yt_dlp_check["path"] == "/usr/bin/yt-dlp"


def test_cli_doctor_uses_custom_yt_dlp_binary(tmp_path, monkeypatch, capsys):
    seen = []

    class FakeEngine:
        cli_path = "/tmp/fluidaudiocli"

        async def healthcheck(self):
            return True

    def fake_which(name):
        seen.append(name)
        return "/custom/yt-dlp" if name == "custom-yt-dlp" else None

    monkeypatch.setattr(
        "undertone_audio.commands.ops.create_engine", lambda name, config: FakeEngine()
    )
    monkeypatch.setattr("undertone_audio.commands.ops.shutil.which", fake_which)

    assert (
        main(
            [
                "--db",
                str(tmp_path / "undertone.db"),
                "doctor",
                "--check-yt-dlp",
                "--yt-dlp-bin",
                "custom-yt-dlp",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    yt_dlp_check = next(check for check in payload["checks"] if check["name"] == "yt_dlp")
    assert seen[-1] == "custom-yt-dlp"
    assert yt_dlp_check["binary"] == "custom-yt-dlp"
    assert yt_dlp_check["path"] == "/custom/yt-dlp"


def test_cli_doctor_reports_pyannote_readiness(tmp_path, monkeypatch, capsys):
    class FakeEngine:
        cli_path = "/tmp/fluidaudiocli"

        async def healthcheck(self):
            return True

    monkeypatch.setattr(
        "undertone_audio.commands.ops.create_engine", lambda name, config: FakeEngine()
    )
    monkeypatch.setattr(
        "undertone_audio.commands.ops.pyannote_status",
        lambda model, device: {
            "ok": False,
            "model": model,
            "device": device,
            "error": "missing",
            "fix": "Install pyannote support.",
        },
    )

    assert (
        main(
            [
                "--db",
                str(tmp_path / "undertone.db"),
                "doctor",
                "--check-pyannote",
                "--json",
            ]
        )
        == 1
    )

    payload = json.loads(capsys.readouterr().out)
    pyannote = next(check for check in payload["checks"] if check["name"] == "pyannote")
    assert pyannote["ok"] is False
    assert pyannote["model"] == "pyannote/speaker-diarization-community-1"
    assert pyannote["device"] == "auto"
    assert pyannote["fix"] == "Install pyannote support."


def test_cli_numeric_zero_overrides_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("UNDERTONE_CLUSTERING_THRESHOLD", "0.7")
    monkeypatch.setenv("UNDERTONE_SPEAKER_MERGE_THRESHOLD", "0.8")
    args = Namespace(
        db=tmp_path / "undertone.db",
        fluidaudio_cli=None,
        engine=None,
        clustering_threshold=0.0,
        speaker_merge_threshold=0.0,
        min_talk_seconds=0.0,
        fingerprint_similarity_threshold=0.0,
        turn_gap_ms=0,
        pyannote_model=None,
        pyannote_device=None,
    )

    config = config_for_args(args)

    assert isinstance(config, Config)
    assert config.clustering_threshold == 0.0
    assert config.speaker_merge_threshold == 0.0
    assert config.min_talk_seconds == 0.0
    assert config.fingerprint_similarity_threshold == 0.0
    assert config.turn_gap_ms == 0


def test_cli_search_reports_invalid_fts_query(tmp_path, capsys):
    db = tmp_path / "undertone.db"

    assert main(["--db", str(db), "search", '"']) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "undertone:" in captured.err


def test_cli_human_readable_operator_outputs(tmp_path, monkeypatch, capsys):
    class FakeEngine:
        cli_path = "/tmp/fluidaudiocli"

        async def healthcheck(self):
            return True

    monkeypatch.setattr(
        "undertone_audio.commands.ops.create_engine", lambda name, config: FakeEngine()
    )
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(
        json.dumps(
            {
                "duration_ms": 16000,
                "language": "en",
                "engine": "fixture",
                "speakers": [{"speaker_id": "S1", "embedding": [1.0, 0.0]}],
                "segments": [
                    {
                        "segment_id": "seg1",
                        "speaker_id": "S1",
                        "start_ms": 0,
                        "end_ms": 16000,
                        "text": "human readable speaker line",
                    }
                ],
            }
        )
    )
    db = tmp_path / "undertone.db"
    assert (
        main(["--db", str(db), "finalize-json", str(raw_path), "--transcript-id", "human-1"]) == 0
    )
    created = json.loads(capsys.readouterr().out)
    fingerprint_id = created["speakers"][0]["fingerprint_id"]

    assert main(["--db", str(db), "doctor"]) == 0
    output = capsys.readouterr().out
    assert "undertone doctor" in output
    assert "Sources" in output

    assert main(["--db", str(db), "models"]) == 0
    assert "undertone models" in capsys.readouterr().out

    assert main(["--db", str(db), "stats"]) == 0
    assert "transcripts:" in capsys.readouterr().out

    assert main(["--db", str(db), "list"]) == 0
    assert "human-1" in capsys.readouterr().out

    assert main(["--db", str(db), "search", "speaker"]) == 0
    assert "Search results" in capsys.readouterr().out

    assert main(["--db", str(db), "fingerprints", "--unnamed", "--excerpts"]) == 0
    output = capsys.readouterr().out
    assert fingerprint_id in output
    assert "human readable speaker line" in output

    assert main(["--db", str(db), "fingerprint-label", fingerprint_id, "Alex Rivera"]) == 0
    assert f"Labeled {fingerprint_id} as Alex Rivera" in capsys.readouterr().out


def test_cli_no_command_prints_grouped_overview(capsys):
    import argparse

    from undertone_audio.cli import _COMMAND_GROUPS, _parser

    assert main([]) == 0
    out = capsys.readouterr().out
    assert "Commands:" in out
    assert "Ingest audio:" in out
    assert "Sources:" in out
    assert "{finalize-json" not in out

    parser = _parser()
    subparsers = next(
        action
        for action in parser._subparsers._group_actions
        if isinstance(action, argparse._SubParsersAction)
    )
    grouped = {name for _, group in _COMMAND_GROUPS for name in group}
    assert set(subparsers.choices) <= grouped


def test_cli_sources_reports_readiness(monkeypatch, capsys):
    monkeypatch.setattr(
        "undertone_audio.commands.ops.source_statuses",
        lambda check_meet=False: [
            {"source": "youtube", "state": "ready", "detail": "yt-dlp: /bin/yt-dlp", "fix": None},
            {
                "source": "meet",
                "state": "needs-auth",
                "detail": "reauth required",
                "fix": "Run gcloud auth application-default login.",
            },
        ],
    )

    assert main(["sources"]) == 0
    output = capsys.readouterr().out
    assert "youtube" in output
    assert "needs-auth" in output

    assert main(["sources", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sources"][1]["fix"].startswith("Run gcloud")


def test_install_skills_copies_router_and_references(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    assert main(["install-skills", "--target", "claude-project", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["installs"][0]["status"] == "installed"

    dest = tmp_path / ".claude" / "skills" / "undertone"
    assert (dest / "SKILL.md").is_file()
    assert (dest / "references" / "fingerprints.md").is_file()

    assert main(["install-skills", "--target", "claude-project", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["installs"][0]["status"] == "exists"

    sentinel = dest / "references" / "stale.md"
    sentinel.write_text("stale")
    assert main(["install-skills", "--target", "claude-project", "--force", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["installs"][0]["status"] == "installed"
    assert not sentinel.exists()
