import json
from argparse import Namespace
from pathlib import Path

from undertone_audio.config import Config
from undertone_audio import webhooks
from undertone_audio.commands.common import config_for_args
from undertone_audio.engines.base import RawTranscript
from undertone_audio.schema import Segment, Speaker
from undertone_audio.cli import main


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
    assert {check["name"] for check in payload["checks"]} == {"db_writable", "engine", "yt_dlp"}
    assert {source["source"] for source in payload["sources"]} == {
        "youtube",
        "podcast",
        "meet",
        "quill",
    }


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
