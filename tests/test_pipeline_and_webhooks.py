import hashlib
import hmac
import json
import sqlite3

from undertone_audio import AudioPipeline, Segment, Speaker
from undertone_audio.config import Config
from undertone_audio.diarization.fingerprint import SpeakerFingerprintStore
from undertone_audio.engines.base import RawTranscript
from undertone_audio.storage import TranscriptStore
from undertone_audio import webhooks


def test_pipeline_finalizes_raw_transcript_and_emits_readiness(monkeypatch, tmp_path):
    calls = []

    class Response:
        status_code = 204

    def fake_post(url, *, data, headers, timeout):
        calls.append({"url": url, "data": data, "headers": headers, "timeout": timeout})
        return Response()

    monkeypatch.setattr(webhooks.requests, "post", fake_post)

    store = TranscriptStore(tmp_path / "undertone.db")
    try:
        pipeline = AudioPipeline(
            store=store,
            config=Config(
                db_path=tmp_path / "undertone.db",
                webhook_url="https://zen.example/webhooks/workflow/ready",
                webhook_secret="shared-secret",
                webhook_enabled=True,
                webhook_accept_degraded=False,
            ),
        )
        transcript = pipeline.finalize_raw(
            RawTranscript(
                duration_ms=1000,
                language="en",
                engine="example",
                speakers=[Speaker(speaker_id="S1", fingerprint_id="VP-1")],
                segments=[
                    Segment(
                        segment_id="seg1",
                        speaker_id="S1",
                        start_ms=0,
                        end_ms=1000,
                        text="hello world",
                    )
                ],
            ),
            transcript_id="ready-1",
            diarization_state="ok",
        )

        assert transcript.transcript_id == "ready-1"
        assert store.load("ready-1") is not None
        assert len(calls) == 1
        body = calls[0]["data"]
        payload = json.loads(body.decode("utf-8"))
        assert payload == {
            "event": "meeting.transcript.ready",
            "transcript_id": "ready-1",
            "source": "undertone",
            "recorded_at": None,
            "store_ref": f"sqlite:{(tmp_path / 'undertone.db').resolve()}#ready-1",
        }
        expected = hmac.new(b"shared-secret", body, hashlib.sha256).hexdigest()
        assert calls[0]["headers"]["x-zen-signature-256"] == f"sha256={expected}"
    finally:
        store.close()


def test_webhook_suppresses_degraded_unless_explicit(tmp_path, monkeypatch):
    calls = []

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError("network down")

    monkeypatch.setattr(webhooks.requests, "post", fake_post)
    store = TranscriptStore(tmp_path / "undertone.db")
    try:
        pipeline = AudioPipeline(
            store=store,
            config=Config(
                db_path=tmp_path / "undertone.db",
                webhook_url="https://zen.example/webhooks/workflow/ready",
                webhook_secret="shared-secret",
                webhook_enabled=True,
                webhook_accept_degraded=False,
            ),
        )
        transcript = pipeline.finalize_raw(
            RawTranscript(
                duration_ms=1000,
                language="en",
                engine="example",
                speakers=[Speaker(speaker_id="S1")],
                segments=[Segment(segment_id="seg1", speaker_id="S1", start_ms=0, end_ms=1000, text="hi")],
            ),
            transcript_id="failed-1",
            diarization_state="failed",
        )
        assert transcript.transcript_id == "failed-1"
        assert calls == []

        assert webhooks.emit_transcript_ready(
            transcript,
            store.db_path,
            Config(
                db_path=tmp_path / "undertone.db",
                webhook_url="https://zen.example/webhooks/workflow/ready",
                webhook_secret="shared-secret",
                webhook_enabled=True,
                webhook_accept_degraded=True,
            ),
        ) is False
        assert len(calls) == 1
    finally:
        store.close()


def test_pipeline_rolls_back_transcript_when_fingerprint_commit_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("UNDERTONE_WEBHOOK_ENABLED", "0")

    class ExplodingFingerprintStore(SpeakerFingerprintStore):
        def apply_plan_on_conn(self, conn: sqlite3.Connection, plan):
            raise RuntimeError("fingerprint write failed")

    db = tmp_path / "undertone.db"
    store = TranscriptStore(db)
    try:
        pipeline = AudioPipeline(
            store=store,
            config=Config(db_path=db, webhook_enabled=False),
            fingerprint_store=ExplodingFingerprintStore(db),
        )

        raw = RawTranscript(
            duration_ms=1000,
            language="en",
            engine="fluidaudio-cli",
            speakers=[Speaker(speaker_id="S1", embedding=[1.0, 0.0])],
            segments=[
                Segment(segment_id="seg1", speaker_id="S1", start_ms=0, end_ms=1000, text="hi")
            ],
        )

        try:
            pipeline.finalize_raw(raw, transcript_id="atomic-failure")
        except RuntimeError as exc:
            assert "fingerprint write failed" in str(exc)
        else:
            raise AssertionError("expected fingerprint commit failure")

        assert store.load("atomic-failure") is None
        assert store._conn.execute("SELECT count(*) FROM speaker_fingerprints").fetchone()[0] == 0
    finally:
        store.close()
