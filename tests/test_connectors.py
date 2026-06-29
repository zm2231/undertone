import json
from pathlib import Path

from undertone_audio.cli import main
from undertone_audio.connectors.base import connector_for_ref, discover_connectors
from undertone_audio.connectors.podcast import PodcastConnector
from undertone_audio.connectors.web import WebMediaConnector
from undertone_audio.connectors.youtube import YouTubeConnector
from undertone_audio.engines.base import RawTranscript
from undertone_audio.schema import Segment, Speaker


def test_youtube_connector_uses_yt_dlp_args_without_shell(monkeypatch, tmp_path):
    commands = []

    monkeypatch.setattr("undertone_audio.connectors.youtube.ensure_binary", lambda binary: binary)
    monkeypatch.setattr(
        "undertone_audio.connectors.youtube.run_json",
        lambda cmd, **kwargs: {
            "id": "abc123",
            "title": "Interview",
            "webpage_url": "https://youtu.be/abc123",
            "channel": "Channel",
            "duration": 120,
        },
    )

    def fake_run_checked(cmd, **kwargs):
        commands.append(cmd)
        download_dir = Path(cmd[cmd.index("--paths") + 1])
        (download_dir / "abc123.wav").write_bytes(b"audio")

    monkeypatch.setattr("undertone_audio.connectors.youtube.run_checked", fake_run_checked)

    asset = YouTubeConnector(download_dir=tmp_path).fetch("https://youtu.be/abc123")

    assert asset.audio_path == tmp_path / "abc123.wav"
    assert asset.transcript_id_hint == "youtube-abc123"
    assert asset.metadata["audio_priority"] == "downloaded-youtube-audio"
    assert commands[0][0] == "yt-dlp"
    assert "https://youtu.be/abc123" in commands[0]


def test_youtube_connector_matches_only_real_youtube_hosts():
    connector = YouTubeConnector()

    assert connector.matches("https://youtube.com/watch?v=abc")
    assert connector.matches("https://www.youtube.com:443/watch?v=abc")
    assert connector.matches("https://music.youtube.com/watch?v=abc")
    assert connector.matches("https://youtu.be/abc")
    assert not connector.matches("https://notyoutube.com/watch?v=abc")
    assert not connector.matches("https://youtube.com.example/watch?v=abc")
    assert not connector.matches("https://notyoutu.be/abc")


def test_web_connector_is_force_only():
    assert not WebMediaConnector().matches("https://example.com/post")


def test_youtube_connector_does_not_publish_failed_download(monkeypatch, tmp_path):
    monkeypatch.setattr("undertone_audio.connectors.youtube.ensure_binary", lambda binary: binary)
    monkeypatch.setattr(
        "undertone_audio.connectors.youtube.run_json",
        lambda cmd, **kwargs: {"id": "abc123", "title": "Interview"},
    )

    def fake_run_checked(cmd, **kwargs):
        download_dir = Path(cmd[cmd.index("--paths") + 1])
        (download_dir / "abc123.wav").write_bytes(b"partial")
        raise RuntimeError("network dropped")

    monkeypatch.setattr("undertone_audio.connectors.youtube.run_checked", fake_run_checked)

    try:
        YouTubeConnector(download_dir=tmp_path).fetch("https://youtu.be/abc123")
    except RuntimeError as exc:
        assert "network dropped" in str(exc)
    else:
        raise AssertionError("fetch should fail")

    assert not (tmp_path / "abc123.wav").exists()
    assert not list(tmp_path.glob(".abc123.*"))


def test_youtube_connector_rejects_empty_download(monkeypatch, tmp_path):
    monkeypatch.setattr("undertone_audio.connectors.youtube.ensure_binary", lambda binary: binary)
    monkeypatch.setattr(
        "undertone_audio.connectors.youtube.run_json",
        lambda cmd, **kwargs: {"id": "abc123", "title": "Interview"},
    )

    def fake_run_checked(cmd, **kwargs):
        download_dir = Path(cmd[cmd.index("--paths") + 1])
        (download_dir / "abc123.wav").write_bytes(b"")

    monkeypatch.setattr("undertone_audio.connectors.youtube.run_checked", fake_run_checked)

    try:
        YouTubeConnector(download_dir=tmp_path).fetch("https://youtu.be/abc123")
    except RuntimeError as exc:
        assert "empty audio file" in str(exc)
    else:
        raise AssertionError("fetch should fail")

    assert not (tmp_path / "abc123.wav").exists()
    assert not list(tmp_path.glob(".abc123.*"))


def test_podcast_connector_lists_and_downloads_local_feed(tmp_path):
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    feed = tmp_path / "feed.xml"
    feed.write_text(
        f"""<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <item>
      <title>Episode One</title>
      <guid>ep-1</guid>
      <pubDate>Mon, 01 Jun 2026 12:00:00 GMT</pubDate>
      <enclosure url="{audio.as_uri()}" type="audio/mpeg" />
    </item>
  </channel>
</rss>
"""
    )

    connector = PodcastConnector(download_dir=tmp_path / "downloads")
    episodes = connector.list_episodes(feed.as_uri())
    asset = connector.fetch(feed.as_uri(), episode=0)

    assert episodes[0].title == "Episode One"
    assert asset.audio_path.read_bytes() == b"audio"
    assert asset.transcript_id_hint == "podcast-ep-1"
    assert asset.metadata["audio_priority"] == "downloaded-podcast-audio"


def test_podcast_connector_removes_partial_download(monkeypatch, tmp_path):
    class FailingResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, _size=-1):
            raise OSError("network dropped")

    monkeypatch.setattr(
        "undertone_audio.connectors.podcast.urllib.request.urlopen",
        lambda *args, **kwargs: FailingResponse(),
    )

    connector = PodcastConnector(download_dir=tmp_path)
    try:
        connector.fetch("https://example.com/episode.mp3")
    except OSError as exc:
        assert "network dropped" in str(exc)
    else:
        raise AssertionError("fetch should fail")

    assert not list(tmp_path.glob("*.mp3"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_podcast_list_cli(tmp_path, capsys):
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    feed = tmp_path / "feed.xml"
    feed.write_text(
        f"""<rss><channel><item><title>Episode</title><guid>g1</guid>
<enclosure url="{audio.as_uri()}" type="audio/mpeg" />
</item></channel></rss>"""
    )

    assert main(["podcast-list", feed.as_uri(), "--limit", "1", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["title"] == "Episode"
    assert payload[0]["guid"] == "g1"

    assert main(["podcast-list", feed.as_uri(), "--limit", "1"]) == 0
    assert "Podcast episodes" in capsys.readouterr().out


def test_youtube_ingest_cli_reruns_local_audio_pipeline(tmp_path, monkeypatch, capsys):
    class FakeConnector:
        def __init__(self, **kwargs):
            self.path = tmp_path / "yt.wav"
            self.path.write_bytes(b"audio")

        def fetch(self, url):
            from undertone_audio.connectors import ConnectorAsset

            return ConnectorAsset(
                audio_path=self.path,
                source_url=url,
                source_kind="youtube-audio",
                title="Interview",
                transcript_id_hint="youtube-abc",
                metadata={"source": "youtube", "audio_priority": "downloaded-youtube-audio"},
            )

    class FakeEngine:
        async def transcribe(self, audio_path: Path):
            assert audio_path.name == "yt.wav"
            return RawTranscript(
                duration_ms=1000,
                language="en",
                engine="fluidaudio-hybrid",
                speakers=[Speaker(speaker_id="S1", embedding=[1.0, 0.0])],
                segments=[
                    Segment(
                        segment_id="s1",
                        speaker_id="S1",
                        start_ms=0,
                        end_ms=1000,
                        text="speaker attributed audio",
                    )
                ],
            )

    monkeypatch.setattr("undertone_audio.commands.connectors.YouTubeConnector", FakeConnector)
    monkeypatch.setattr("undertone_audio.commands.connectors.create_engine", lambda name, config: FakeEngine())
    monkeypatch.setenv("UNDERTONE_WEBHOOK_ENABLED", "0")

    assert main(
        [
            "--db",
            str(tmp_path / "undertone.db"),
            "youtube-ingest",
            "https://youtu.be/abc",
            "--output-detail",
            "standard",
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["transcript_id"] == "youtube-abc"
    assert payload["metadata"]["source_url"] == "https://youtu.be/abc"
    assert payload["metadata"]["source_metadata"]["audio_priority"] == "downloaded-youtube-audio"


def test_connector_list_and_generic_ingest_use_discovered_connector(tmp_path, monkeypatch, capsys):
    class FakeConnector:
        name = "fixture"
        source_kind = "fixture-audio"

        def matches(self, ref):
            return ref.startswith("fixture:")

        def fetch(self, ref):
            from undertone_audio.connectors import ConnectorAsset

            path = tmp_path / "fixture.wav"
            path.write_bytes(b"audio")
            return ConnectorAsset(
                audio_path=path,
                source_url=ref,
                source_kind="fixture-audio",
                transcript_id_hint="fixture-1",
                metadata={"source": "fixture"},
            )

    class FakeEngine:
        async def transcribe(self, audio_path: Path):
            return RawTranscript(
                duration_ms=1000,
                language="en",
                engine="fixture-engine",
                speakers=[Speaker(speaker_id="S1")],
                segments=[
                    Segment(
                        segment_id="s1",
                        speaker_id="S1",
                        start_ms=0,
                        end_ms=1000,
                        text="generic connector",
                    )
                ],
            )

    monkeypatch.setattr(
        "undertone_audio.commands.connectors.discover_connectors",
        lambda: [FakeConnector()],
    )
    monkeypatch.setattr(
        "undertone_audio.commands.connectors.connector_for_ref",
        lambda ref, preferred=None: FakeConnector(),
    )
    monkeypatch.setattr("undertone_audio.commands.connectors.create_engine", lambda name, config: FakeEngine())
    monkeypatch.setenv("UNDERTONE_WEBHOOK_ENABLED", "0")

    assert main(["connector-list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == [
        {"name": "fixture", "source_kind": "fixture-audio"}
    ]

    assert (
        main(
            [
                "--db",
                str(tmp_path / "undertone.db"),
                "connector-ingest",
                "fixture:one",
                "--progress",
                "json",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert [json.loads(line)["event"] for line in captured.err.splitlines()] == [
        "fetching",
        "start",
        "transcribed",
        "finalizing",
        "saved",
    ]
    payload = json.loads(captured.out)
    assert payload["transcript_id"] == "fixture-1"
    assert payload["metadata"]["source_metadata"]["source"] == "fixture"


def test_connector_ingest_skip_existing_does_not_fetch_when_transcript_id_known(
    tmp_path,
    monkeypatch,
    capsys,
):
    db = tmp_path / "undertone.db"
    _save_existing_transcript(tmp_path, db, "existing-connector")
    capsys.readouterr()

    def fail_connector_for_ref(ref, preferred=None):
        raise AssertionError("connector should not be resolved")

    monkeypatch.setattr(
        "undertone_audio.commands.connectors.connector_for_ref",
        fail_connector_for_ref,
    )

    assert (
        main(
            [
                "--db",
                str(db),
                "connector-ingest",
                "fixture:one",
                "--transcript-id",
                "existing-connector",
                "--skip-existing",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["skipped"] is True


def test_connector_ingest_skip_existing_does_not_fetch_for_known_youtube_ref(
    tmp_path,
    monkeypatch,
    capsys,
):
    db = tmp_path / "undertone.db"
    _save_existing_transcript(tmp_path, db, "youtube-abc123")
    capsys.readouterr()

    def fail_connector_for_ref(ref, preferred=None):
        raise AssertionError("connector should not be resolved")

    monkeypatch.setattr(
        "undertone_audio.commands.connectors.connector_for_ref",
        fail_connector_for_ref,
    )

    assert (
        main(
            [
                "--db",
                str(db),
                "connector-ingest",
                "https://www.youtube.com/watch?v=abc123",
                "--skip-existing",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["skipped"] is True


def test_connector_ingest_skip_existing_does_not_fetch_for_preferred_youtube_ref(
    tmp_path,
    monkeypatch,
    capsys,
):
    db = tmp_path / "undertone.db"
    _save_existing_transcript(tmp_path, db, "youtube-abc123")
    capsys.readouterr()

    def fail_connector_for_ref(ref, preferred=None):
        raise AssertionError("connector should not be resolved")

    monkeypatch.setattr(
        "undertone_audio.commands.connectors.connector_for_ref",
        fail_connector_for_ref,
    )

    assert (
        main(
            [
                "--db",
                str(db),
                "connector-ingest",
                "https://www.youtube.com/watch?v=abc123",
                "--connector",
                "youtube",
                "--skip-existing",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["skipped"] is True


def test_youtube_ingest_rejects_non_youtube_url_before_fetch(capsys):
    assert main(["youtube-ingest", "https://example.com/post", "--dry-run"]) == 1
    err = capsys.readouterr().err
    assert "youtube-ingest only accepts youtube.com or youtu.be URLs" in err
    assert "web-ingest" in err


def test_connector_ingest_preferred_youtube_rejects_non_youtube_url(capsys):
    assert main(["connector-ingest", "https://example.com/post", "--connector", "youtube"]) == 1
    err = capsys.readouterr().err
    assert "connector 'youtube' only accepts youtube.com or youtu.be URLs" in err
    assert "web-ingest" in err


def test_connector_ingest_rejects_preferred_web_connector_before_fetch(monkeypatch, capsys):
    def fail_fetch(self, ref):
        raise AssertionError("connector-ingest should not fetch with the web connector")

    monkeypatch.setattr("undertone_audio.connectors.web.WebMediaConnector.fetch", fail_fetch)

    assert main(["connector-ingest", "https://example.com/post", "--connector", "web"]) == 1
    err = capsys.readouterr().err
    assert "connector 'web' is only available" in err
    assert "web-ingest" in err


def test_connector_ingest_does_not_apply_youtube_hint_to_non_youtube_query(
    tmp_path,
    monkeypatch,
    capsys,
):
    db = tmp_path / "undertone.db"
    _save_existing_transcript(tmp_path, db, "youtube-abc123")
    capsys.readouterr()

    class FakeConnector:
        name = "fixture"
        source_kind = "fixture-audio"

        def matches(self, ref):
            return True

        def fetch(self, ref):
            from undertone_audio.connectors import ConnectorAsset

            path = tmp_path / "fixture.wav"
            path.write_bytes(b"audio")
            return ConnectorAsset(
                audio_path=path,
                source_url=ref,
                source_kind="fixture-audio",
                transcript_id_hint="fixture-abc123",
                metadata={"source": "fixture"},
            )

    class FakeEngine:
        async def transcribe(self, audio_path: Path):
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
                        text="not youtube",
                    )
                ],
            )

    monkeypatch.setattr(
        "undertone_audio.commands.connectors.connector_for_ref",
        lambda ref, preferred=None: FakeConnector(),
    )
    monkeypatch.setattr("undertone_audio.commands.connectors.create_engine", lambda name, config: FakeEngine())
    monkeypatch.setenv("UNDERTONE_WEBHOOK_ENABLED", "0")

    assert (
        main(
            [
                "--db",
                str(db),
                "connector-ingest",
                "https://example.com/file.mp3?v=abc123",
                "--skip-existing",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["transcript_id"] == "fixture-abc123"


def test_connector_ingest_preferred_connector_disables_builtin_early_hint(
    tmp_path,
    monkeypatch,
    capsys,
):
    db = tmp_path / "undertone.db"
    _save_existing_transcript(tmp_path, db, "youtube-abc123")
    capsys.readouterr()

    class FakeConnector:
        name = "fixture"
        source_kind = "fixture-audio"

        def fetch(self, ref):
            from undertone_audio.connectors import ConnectorAsset

            path = tmp_path / "fixture.wav"
            path.write_bytes(b"audio")
            return ConnectorAsset(
                audio_path=path,
                source_url=ref,
                source_kind="fixture-audio",
                transcript_id_hint="fixture-youtube-url",
                metadata={"source": "fixture"},
            )

    class FakeEngine:
        async def transcribe(self, audio_path: Path):
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
                        text="preferred",
                    )
                ],
            )

    monkeypatch.setattr(
        "undertone_audio.commands.connectors.connector_for_ref",
        lambda ref, preferred=None: FakeConnector(),
    )
    monkeypatch.setattr("undertone_audio.commands.connectors.create_engine", lambda name, config: FakeEngine())
    monkeypatch.setenv("UNDERTONE_WEBHOOK_ENABLED", "0")

    assert (
        main(
            [
                "--db",
                str(db),
                "connector-ingest",
                "https://www.youtube.com/watch?v=abc123",
                "--connector",
                "fixture",
                "--skip-existing",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["transcript_id"] == "fixture-youtube-url"


def test_connector_ingest_skip_existing_does_not_fetch_for_known_direct_podcast_ref(
    tmp_path,
    monkeypatch,
    capsys,
):
    from undertone_audio.connectors.podcast import _stable_id

    url = "https://example.com/episode.mp3"
    db = tmp_path / "undertone.db"
    _save_existing_transcript(tmp_path, db, f"podcast-{_stable_id(url)}")
    capsys.readouterr()

    def fail_connector_for_ref(ref, preferred=None):
        raise AssertionError("connector should not be resolved")

    monkeypatch.setattr(
        "undertone_audio.commands.connectors.connector_for_ref",
        fail_connector_for_ref,
    )

    assert (
        main(
            [
                "--db",
                str(db),
                "connector-ingest",
                url,
                "--skip-existing",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["skipped"] is True


def test_youtube_ingest_skip_existing_does_not_fetch_when_video_id_known(
    tmp_path,
    monkeypatch,
    capsys,
):
    db = tmp_path / "undertone.db"
    _save_existing_transcript(tmp_path, db, "youtube-abc123")
    capsys.readouterr()

    class FailingYouTubeConnector:
        def __init__(self, **kwargs):
            raise AssertionError("youtube connector should not be constructed")

    monkeypatch.setattr(
        "undertone_audio.commands.connectors.YouTubeConnector",
        FailingYouTubeConnector,
    )

    assert (
        main(
            [
                "--db",
                str(db),
                "youtube-ingest",
                "https://www.youtube.com/watch?v=abc123",
                "--skip-existing",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["skipped"] is True


def test_podcast_ingest_skip_existing_does_not_fetch_when_transcript_id_known(
    tmp_path,
    monkeypatch,
    capsys,
):
    db = tmp_path / "undertone.db"
    _save_existing_transcript(tmp_path, db, "existing-podcast")
    capsys.readouterr()

    class FailingPodcastConnector:
        def __init__(self, **kwargs):
            raise AssertionError("podcast connector should not be constructed")

    monkeypatch.setattr(
        "undertone_audio.commands.connectors.PodcastConnector",
        FailingPodcastConnector,
    )

    assert (
        main(
            [
                "--db",
                str(db),
                "podcast-ingest",
                "https://example.com/feed.xml",
                "--transcript-id",
                "existing-podcast",
                "--skip-existing",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["skipped"] is True


def test_podcast_direct_url_skip_existing_does_not_fetch_when_hint_known(
    tmp_path,
    monkeypatch,
    capsys,
):
    from undertone_audio.connectors.podcast import _stable_id

    url = "https://example.com/episode.mp3"
    db = tmp_path / "undertone.db"
    _save_existing_transcript(tmp_path, db, f"podcast-{_stable_id(url)}")
    capsys.readouterr()

    class FailingPodcastConnector:
        def __init__(self, **kwargs):
            raise AssertionError("podcast connector should not be constructed")

    monkeypatch.setattr(
        "undertone_audio.commands.connectors.PodcastConnector",
        FailingPodcastConnector,
    )

    assert (
        main(
            [
                "--db",
                str(db),
                "podcast-ingest",
                url,
                "--skip-existing",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["skipped"] is True


def test_connector_discovery_skips_broken_entry_point(monkeypatch):
    class BrokenEntryPoint:
        name = "broken"

        def load(self):
            raise RuntimeError("bad plugin")

    monkeypatch.setattr(
        "undertone_audio.connectors.base.entry_points",
        lambda group=None: [BrokenEntryPoint()] if group == "undertone.connectors" else [],
    )

    names = [connector.name for connector in discover_connectors()]
    assert "youtube" in names
    assert "podcast" in names


def test_connector_discovery_keeps_builtins_when_valid_third_party_exists(monkeypatch):
    class ThirdPartyConnector:
        name = "third"
        source_kind = "third-party"

        def matches(self, ref):
            return ref.startswith("third:")

        def fetch(self, ref):
            raise AssertionError("not used")

    class ThirdPartyEntryPoint:
        name = "third"

        def load(self):
            return ThirdPartyConnector

    monkeypatch.setattr(
        "undertone_audio.connectors.base.entry_points",
        lambda group=None: [ThirdPartyEntryPoint()] if group == "undertone.connectors" else [],
    )

    names = [connector.name for connector in discover_connectors()]
    assert names == ["youtube", "podcast", "web", "third"]

    connector = connector_for_ref("https://www.youtube.com/watch?v=abc123")
    assert connector.name == "youtube"


def test_connector_discovery_rejects_duplicate_names(monkeypatch):
    class DuplicateConnector:
        name = "youtube"
        source_kind = "duplicate"

        def matches(self, ref):
            return False

        def fetch(self, ref):
            raise AssertionError("not used")

    class DuplicateEntryPoint:
        def __init__(self, name):
            self.name = name

        def load(self):
            return DuplicateConnector

    monkeypatch.setattr(
        "undertone_audio.connectors.base.entry_points",
        lambda group=None: [DuplicateEntryPoint("one"), DuplicateEntryPoint("two")]
        if group == "undertone.connectors"
        else [],
    )

    try:
        discover_connectors()
    except Exception as exc:
        assert "duplicate connector name: youtube" in str(exc)
    else:
        raise AssertionError("duplicate connector names should fail")


def test_connector_ingest_rejects_bad_fetch_return(monkeypatch, tmp_path, capsys):
    class BadConnector:
        name = "bad"
        source_kind = "bad"

        def matches(self, ref):
            return True

        def fetch(self, ref):
            return {"audio_path": "not-an-asset"}

    monkeypatch.setattr(
        "undertone_audio.commands.connectors.connector_for_ref",
        lambda ref, preferred=None: BadConnector(),
    )

    assert (
        main(
            [
                "--db",
                str(tmp_path / "undertone.db"),
                "connector-ingest",
                "bad:anything",
            ]
        )
        == 1
    )
    assert "expected ConnectorAsset" in capsys.readouterr().err


def test_connector_resolve_lists_ranked_stable_web_candidates(monkeypatch, capsys):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    monkeypatch.setattr(
        "undertone_audio.connectors.web.run_json",
        lambda cmd, **kwargs: {
            "extractor_key": "Substack",
            "title": "Article",
            "entries": [
                {
                    "id": "voice",
                    "extractor_key": "Substack",
                    "webpage_url": "https://example.com/post",
                    "title": "AI voiceover",
                    "duration": 300,
                    "ext": "mp3",
                },
                {
                    "id": "yt-real",
                    "extractor_key": "Youtube",
                    "webpage_url": "https://www.youtube.com/watch?v=real",
                    "title": "Real interview",
                    "duration": 1255,
                },
            ],
        },
    )

    assert main(["connector-resolve", "https://example.com/post", "--json"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert [row["kind"] for row in first] == ["external-video", "page-voiceover"]
    assert first[0]["title"] == "Real interview"

    assert main(["connector-resolve", "https://example.com/post", "--json"]) == 0
    second = json.loads(capsys.readouterr().out)
    assert [row["candidate_id"] for row in first] == [row["candidate_id"] for row in second]


def test_web_candidate_id_ignores_rotating_signed_media_url(monkeypatch):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    responses = [
        {
            "id": "episode-1",
            "extractor_key": "Substack",
            "webpage_url": "https://example.com/post",
            "url": "https://cdn.example.com/audio.mp3?token=one",
        },
        {
            "id": "episode-1",
            "extractor_key": "Substack",
            "webpage_url": "https://example.com/post",
            "url": "https://cdn.example.com/audio.mp3?token=two",
        },
    ]

    def fake_run_json(cmd, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr("undertone_audio.connectors.web.run_json", fake_run_json)
    connector = WebMediaConnector()
    first = connector.resolve("https://example.com/post")[0]
    second = connector.resolve("https://example.com/post")[0]
    assert first.candidate_id == second.candidate_id


def test_web_resolver_coalesces_duplicate_signed_url_candidates(monkeypatch):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    monkeypatch.setattr(
        "undertone_audio.connectors.web.run_json",
        lambda cmd, **kwargs: {
            "entries": [
                {
                    "id": "episode-1",
                    "extractor_key": "Substack",
                    "webpage_url": "https://example.com/post",
                    "url": "https://cdn.example.com/audio.mp3?token=one",
                    "title": "Episode",
                    "duration": 120,
                },
                {
                    "id": "episode-1",
                    "extractor_key": "Substack",
                    "webpage_url": "https://example.com/post",
                    "url": "https://cdn.example.com/audio.mp3?token=two",
                    "title": "Episode",
                    "duration": 120,
                },
            ]
        },
    )

    rows = WebMediaConnector().resolve("https://example.com/post")
    assert len(rows) == 1
    assert len({row.candidate_id for row in rows}) == len(rows)


def test_web_candidate_id_distinguishes_stable_query_identity(monkeypatch):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    monkeypatch.setattr(
        "undertone_audio.connectors.web.run_json",
        lambda cmd, **kwargs: {
            "entries": [
                {
                    "id": "dup",
                    "extractor_key": "Substack",
                    "webpage_url": "https://example.com/post",
                    "url": "https://cdn.example.com/audio.mp3?episode=one&token=SECRET",
                    "title": "One",
                },
                {
                    "id": "dup",
                    "extractor_key": "Substack",
                    "webpage_url": "https://example.com/post",
                    "url": "https://cdn.example.com/audio.mp3?episode=two&token=SECRET",
                    "title": "Two",
                },
            ]
        },
    )

    rows = WebMediaConnector().resolve("https://example.com/post")
    by_url = {row.url: row.candidate_id for row in rows}
    assert len(set(by_url.values())) == 2


def test_web_candidate_id_preserves_non_secret_query_identity(monkeypatch):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    monkeypatch.setattr(
        "undertone_audio.connectors.web.run_json",
        lambda cmd, **kwargs: {
            "entries": [
                {
                    "id": "dup",
                    "extractor_key": "Substack",
                    "webpage_url": "https://example.com/post",
                    "url": "https://cdn.example.com/audio.mp3?file=one&token=SECRET",
                    "title": "One",
                },
                {
                    "id": "dup",
                    "extractor_key": "Substack",
                    "webpage_url": "https://example.com/post",
                    "url": "https://cdn.example.com/audio.mp3?file=two&token=SECRET",
                    "title": "Two",
                },
            ]
        },
    )

    rows = WebMediaConnector().resolve("https://example.com/post")
    assert {row.title for row in rows} == {"One", "Two"}
    assert len({row.candidate_id for row in rows}) == 2


def test_web_candidate_id_with_media_id_ignores_signed_cdn_query_churn(monkeypatch):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    responses = [
        {
            "id": "stable-media",
            "extractor_key": "Substack",
            "webpage_url": "https://example.com/post",
            "url": "https://cdn.example.com/audio.mp3?se=2026-06-29T16%3A00Z&sp=r&sv=2024-11-04&sig=one",
            "title": "Episode",
        },
        {
            "id": "stable-media",
            "extractor_key": "Substack",
            "webpage_url": "https://example.com/post",
            "url": "https://cdn.example.com/audio.mp3?se=2026-06-30T16%3A00Z&sp=r&sv=2024-11-04&sig=two",
            "title": "Episode",
        },
    ]

    def fake_run_json(cmd, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr("undertone_audio.connectors.web.run_json", fake_run_json)
    connector = WebMediaConnector()
    first = connector.resolve("https://example.com/post")[0]
    second = connector.resolve("https://example.com/post")[0]
    assert first.candidate_id == second.candidate_id


def test_web_candidate_id_with_media_id_ignores_cloudfront_query_churn(monkeypatch):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    responses = [
        {
            "id": "stable-media",
            "extractor_key": "Substack",
            "webpage_url": "https://example.com/post",
            "url": "https://cdn.example.com/audio.mp3?Expires=111&Policy=one&Signature=aaa&file=one",
            "title": "Episode",
        },
        {
            "id": "stable-media",
            "extractor_key": "Substack",
            "webpage_url": "https://example.com/post",
            "url": "https://cdn.example.com/audio.mp3?Expires=222&Policy=two&Signature=bbb&file=two",
            "title": "Episode",
        },
    ]

    def fake_run_json(cmd, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr("undertone_audio.connectors.web.run_json", fake_run_json)
    connector = WebMediaConnector()
    first = connector.resolve("https://example.com/post")[0]
    second = connector.resolve("https://example.com/post")[0]
    assert first.candidate_id == second.candidate_id


def test_web_candidate_ids_are_order_independent_for_duplicate_media_ids(monkeypatch):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    responses = [
        {
            "entries": [
                {
                    "id": "dup",
                    "extractor_key": "Substack",
                    "webpage_url": "https://example.com/post",
                    "url": "https://cdn.example.com/one.mp3",
                    "title": "One",
                },
                {
                    "id": "dup",
                    "extractor_key": "Substack",
                    "webpage_url": "https://example.com/post",
                    "url": "https://cdn.example.com/two.mp3",
                    "title": "Two",
                },
            ]
        },
        {
            "entries": [
                {
                    "id": "dup",
                    "extractor_key": "Substack",
                    "webpage_url": "https://example.com/post",
                    "url": "https://cdn.example.com/two.mp3",
                    "title": "Two",
                },
                {
                    "id": "dup",
                    "extractor_key": "Substack",
                    "webpage_url": "https://example.com/post",
                    "url": "https://cdn.example.com/one.mp3",
                    "title": "One",
                },
            ]
        },
    ]

    def fake_run_json(cmd, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr("undertone_audio.connectors.web.run_json", fake_run_json)
    connector = WebMediaConnector()
    first = {row.url: row.candidate_id for row in connector.resolve("https://example.com/post")}
    second = {row.url: row.candidate_id for row in connector.resolve("https://example.com/post")}
    assert first == second
    assert len(set(first.values())) == 2


def test_web_candidate_ids_coalesce_same_id_rows_without_download_targets(monkeypatch):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    responses = [
        {
            "entries": [
                {
                    "id": "dup",
                    "extractor_key": "Generic",
                    "webpage_url": "https://example.com/post",
                    "title": "One",
                },
                {
                    "id": "dup",
                    "extractor_key": "Generic",
                    "webpage_url": "https://example.com/post",
                    "title": "Two",
                },
            ]
        },
        {
            "entries": [
                {
                    "id": "dup",
                    "extractor_key": "Generic",
                    "webpage_url": "https://example.com/post",
                    "title": "Two",
                },
                {
                    "id": "dup",
                    "extractor_key": "Generic",
                    "webpage_url": "https://example.com/post",
                    "title": "One",
                },
            ]
        },
    ]

    def fake_run_json(cmd, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr("undertone_audio.connectors.web.run_json", fake_run_json)
    connector = WebMediaConnector()
    first = connector.resolve("https://example.com/post")
    second = connector.resolve("https://example.com/post")
    assert [row.candidate_id for row in first] == [row.candidate_id for row in second]
    assert len(first) == 1
    assert first[0].availability == "found-but-unavailable"


def test_web_duplicate_candidate_ids_ignore_mutable_display_fields(monkeypatch):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    responses = [
        {
            "entries": [
                {
                    "id": "dup",
                    "extractor_key": "Substack",
                    "webpage_url": "https://example.com/post",
                    "url": "https://cdn.example.com/one.mp3",
                    "title": "One",
                    "duration": 60,
                },
                {
                    "id": "dup",
                    "extractor_key": "Substack",
                    "webpage_url": "https://example.com/post",
                    "url": "https://cdn.example.com/two.mp3",
                    "title": "Two",
                    "duration": 120,
                },
            ]
        },
        {
            "entries": [
                {
                    "id": "dup",
                    "extractor_key": "Substack",
                    "webpage_url": "https://example.com/post",
                    "url": "https://cdn.example.com/one.mp3",
                    "title": "One renamed",
                    "duration": 600,
                },
                {
                    "id": "dup",
                    "extractor_key": "Substack",
                    "webpage_url": "https://example.com/post",
                    "url": "https://cdn.example.com/two.mp3",
                    "title": "Two renamed",
                    "duration": 1200,
                },
            ]
        },
    ]

    def fake_run_json(cmd, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr("undertone_audio.connectors.web.run_json", fake_run_json)
    connector = WebMediaConnector()
    first = {row.url: row.candidate_id for row in connector.resolve("https://example.com/post")}
    second = {row.url: row.candidate_id for row in connector.resolve("https://example.com/post")}
    assert first == second


def test_web_duplicate_candidate_ids_ignore_title_derived_kind_changes(monkeypatch):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    responses = [
        {
            "entries": [
                {
                    "id": "dup",
                    "extractor_key": "Generic",
                    "webpage_url": "https://example.com/post",
                    "url": "https://cdn.example.com/one.mp3",
                    "title": "AI voiceover",
                },
                {
                    "id": "dup",
                    "extractor_key": "Generic",
                    "webpage_url": "https://example.com/post",
                    "url": "https://cdn.example.com/two.mp3",
                    "title": "Two",
                },
            ]
        },
        {
            "entries": [
                {
                    "id": "dup",
                    "extractor_key": "Generic",
                    "webpage_url": "https://example.com/post",
                    "url": "https://cdn.example.com/one.mp3",
                    "title": "Renamed",
                },
                {
                    "id": "dup",
                    "extractor_key": "Generic",
                    "webpage_url": "https://example.com/post",
                    "url": "https://cdn.example.com/two.mp3",
                    "title": "Two",
                },
            ]
        },
    ]

    def fake_run_json(cmd, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr("undertone_audio.connectors.web.run_json", fake_run_json)
    connector = WebMediaConnector()
    first = {row.url: row.candidate_id for row in connector.resolve("https://example.com/post")}
    second = {row.url: row.candidate_id for row in connector.resolve("https://example.com/post")}
    assert first == second


def test_web_candidate_identity_tolerates_malformed_candidate_url(monkeypatch):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    monkeypatch.setattr(
        "undertone_audio.connectors.web.run_json",
        lambda cmd, **kwargs: {
            "id": "bad",
            "extractor_key": "Generic",
            "url": "https://example.com:bad/audio.mp3",
            "title": "Malformed URL",
        },
    )

    rows = WebMediaConnector().resolve("https://example.org/post")
    assert len(rows) == 1
    assert rows[0].candidate_id
    assert rows[0].url == "https://example.com:bad/audio.mp3"


def test_web_entries_without_concrete_download_target_are_not_downloadable(monkeypatch):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    monkeypatch.setattr(
        "undertone_audio.connectors.web.run_json",
        lambda cmd, **kwargs: {
            "entries": [
                {
                    "id": "one",
                    "extractor_key": "Generic",
                    "webpage_url": "https://example.com/post",
                    "title": "One",
                },
                {
                    "id": "two",
                    "extractor_key": "Generic",
                    "webpage_url": "https://example.com/post",
                    "title": "Two",
                },
            ]
        },
    )

    rows = WebMediaConnector().resolve("https://example.com/post")
    assert {row.availability for row in rows} == {"found-but-unavailable"}
    assert all("not directly downloadable" in (row.reason or "") for row in rows)


def test_web_entries_with_url_equal_to_article_are_not_downloadable(monkeypatch):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    monkeypatch.setattr(
        "undertone_audio.connectors.web.run_json",
        lambda cmd, **kwargs: {
            "entries": [
                {
                    "id": "one",
                    "extractor_key": "Generic",
                    "webpage_url": "https://example.com/post",
                    "url": "https://example.com/post",
                    "title": "One",
                },
                {
                    "id": "two",
                    "extractor_key": "Generic",
                    "webpage_url": "https://example.com/post",
                    "url": "https://example.com/post",
                    "title": "Two",
                },
            ]
        },
    )

    rows = WebMediaConnector().resolve("https://example.com/post")
    assert {row.availability for row in rows} == {"found-but-unavailable"}
    assert all("not directly downloadable" in (row.reason or "") for row in rows)


def test_connector_candidate_schema_redacts_media_url_tokens():
    from undertone_audio.connectors import ConnectorCandidate

    candidate = ConnectorCandidate(
        candidate_id="c1",
        original_url="https://example.com/post?ref=secret",
        extractor="extractor https://user:pass@example.com/ext.mp3?token=EXT#frag",
        extractor_key="key https://user:pass@example.com/key.mp3?token=KEY#frag",
        webpage_url="https://user:pass@cdn.example.com/page.mp3?token=page#frag",
        url="https://user:pass@cdn.example.com/audio.mp3?token=secret#frag",
        media_id="media https://user:pass@example.com/media.mp3?token=MEDIA#frag",
        format_id="format https://user:pass@example.com/format.mp3?token=FORMAT#frag",
        title="listen https://cdn.example.com/title.mp3?token=title#frag",
        reason="failed https://user:pass@media.example/a.mp3?token=reason",
        metadata={
            "https://user:pass@example.com/key.mp3?token=KEY#frag": "key value",
            "debug_url": "https://user:pass@cdn.example.com/debug.mp3?sig=meta#frag",
            "nested": ["see https://cdn.example.com/nested.mp3?token=nested"],
            "tuple": ("https://user:pass@example.com/m.mp3?token=tuple#frag",),
        },
    )

    payload = candidate.to_schema().model_dump(mode="json")
    assert payload["original_url"] == "https://example.com/post"
    assert payload["extractor"] == "extractor https://example.com/ext.mp3"
    assert payload["extractor_key"] == "key https://example.com/key.mp3"
    assert payload["webpage_url"] == "https://cdn.example.com/page.mp3"
    assert payload["url"] == "https://cdn.example.com/audio.mp3"
    assert payload["media_id"] == "media https://example.com/media.mp3"
    assert payload["format_id"] == "format https://example.com/format.mp3"
    assert payload["title"] == "listen https://cdn.example.com/title.mp3"
    assert payload["reason"] == "failed https://media.example/a.mp3"
    assert "https://example.com/key.mp3" in payload["metadata"]
    assert payload["metadata"]["debug_url"] == "https://cdn.example.com/debug.mp3"
    assert payload["metadata"]["nested"] == ["see https://cdn.example.com/nested.mp3"]
    assert payload["metadata"]["tuple"] == ["https://example.com/m.mp3"]
    assert "secret" not in json.dumps(payload)
    assert "user:pass" not in json.dumps(payload)
    assert "reason" in payload
    assert "token=nested" not in json.dumps(payload)
    assert "token=tuple" not in json.dumps(payload)
    assert "token=title" not in json.dumps(payload)
    assert "token=KEY" not in json.dumps(payload)
    assert "token=EXT" not in json.dumps(payload)
    assert "token=MEDIA" not in json.dumps(payload)
    assert "token=FORMAT" not in json.dumps(payload)


def test_connector_candidate_schema_redacts_jwt_and_opaque_path_segments():
    from undertone_audio.connectors import ConnectorCandidate

    jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkphbmUifQ."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    opaque = "AbCDefGhIjKlMnOpQrStUvWxYz1234567890"
    candidate = ConnectorCandidate(
        candidate_id="c1",
        original_url="https://example.com/post",
        webpage_url=f"https://cdn.example.com/{jwt}/file.mp3",
        url=f"https://cdn.example.com/{opaque}/file.mp3",
        title=f"signed path https://cdn.example.com/{jwt}/file.mp3",
        reason=f"failed https://cdn.example.com/{opaque}/file.mp3",
        metadata={"cdn": f"https://cdn.example.com/{jwt}/file.mp3"},
    )

    payload = candidate.to_schema().model_dump(mode="json")
    blob = json.dumps(payload)
    assert jwt not in blob
    assert opaque not in blob
    assert "https://cdn.example.com/[redacted]/file.mp3" in blob


def test_redaction_removes_bare_bearer_and_api_tokens():
    from undertone_audio.connectors.base import redact_url_values

    payload = redact_url_values(
        {
            "message": "Authorization: Bearer sk_live_SECRET123",
            "key": "sk_test_SECRET456",
        }
    )
    blob = json.dumps(payload)
    assert "SECRET123" not in blob
    assert "SECRET456" not in blob
    assert "Bearer [redacted]" in blob
    assert "sk_[redacted]" in blob


def test_redaction_preserves_benign_long_slugs_and_youtube_ids():
    from undertone_audio.connectors.base import redact_url_values

    payload = redact_url_values(
        {
            "episode": "https://cdn.example.com/episode-142-the-future-of-ai-2024.mp3?sig=SECRET",
            "policy": "https://cdn.example.com/foreign-policy-roundtable-episode-142.mp3?sig=SECRET",
            "design": "https://cdn.example.com/design-token-system-episode.mp3?sig=SECRET",
            "sign": "https://cdn.example.com/sign-language-interview.mp3?sig=SECRET",
            "briefing": "https://cdn.example.com/policy-briefing.mp3?sig=SECRET",
            "youtube": "https://www.youtube.com/watch?v=abc123&sig=SECRET",
        }
    )

    assert payload["episode"] == "https://cdn.example.com/episode-142-the-future-of-ai-2024.mp3"
    assert payload["policy"] == "https://cdn.example.com/foreign-policy-roundtable-episode-142.mp3"
    assert payload["design"] == "https://cdn.example.com/design-token-system-episode.mp3"
    assert payload["sign"] == "https://cdn.example.com/sign-language-interview.mp3"
    assert payload["briefing"] == "https://cdn.example.com/policy-briefing.mp3"
    assert payload["youtube"] == "https://www.youtube.com/watch?v=abc123"
    assert "SECRET" not in json.dumps(payload)


def test_redaction_handles_malformed_ports_without_raising():
    from undertone_audio.connectors.base import redact_url_values

    message = redact_url_values(
        "download failed https://user:pass@example.com:bad/audio.mp3?token=SECRET#frag"
    )

    assert "download failed" in message
    assert "https://example.com/audio.mp3" in message
    assert "user:pass" not in message
    assert "SECRET" not in message
    assert "#frag" not in message


def test_connector_asset_schema_redacts_source_url_and_metadata_tokens(tmp_path):
    from undertone_audio.connectors import ConnectorAsset

    asset = ConnectorAsset(
        audio_path=tmp_path / "a.wav",
        source_url="https://example.com/post?ref=SECRET",
        source_kind="web-media-audio",
        title="asset https://cdn.example.com/title.mp3?token=ASSET#frag",
        metadata={
            "https://user:pass@example.com/key.mp3?token=KEY#frag": "key value",
            "webpage_url": "https://user:pass@cdn.example.com/page.mp3?token=PAGE#frag",
            "nested": {"url": "https://cdn.example.com/a.mp3?sig=ABC"},
            "tuple": ("https://user:pass@example.com/m.mp3?token=tuple#frag",),
        },
    )

    payload = asset.to_schema().model_dump(mode="json")
    blob = json.dumps(payload)
    assert payload["source_url"] == "https://example.com/post"
    assert payload["title"] == "asset https://cdn.example.com/title.mp3"
    assert "https://example.com/key.mp3" in payload["metadata"]
    assert payload["metadata"]["webpage_url"] == "https://cdn.example.com/page.mp3"
    assert payload["metadata"]["nested"]["url"] == "https://cdn.example.com/a.mp3"
    assert payload["metadata"]["tuple"] == ["https://example.com/m.mp3"]
    assert "SECRET" not in blob
    assert "PAGE" not in blob
    assert "ABC" not in blob
    assert "ASSET" not in blob
    assert "KEY" not in blob
    assert "tuple" in payload["metadata"]
    assert "user:pass" not in blob


def test_web_resolver_ranks_same_page_tts_voiceover_below_real_audio(monkeypatch):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    monkeypatch.setattr(
        "undertone_audio.connectors.web.run_json",
        lambda cmd, **kwargs: {
            "entries": [
                {
                    "id": "voiceover",
                    "extractor_key": "Substack",
                    "webpage_url": "https://example.com/post/audio.mp3",
                    "title": "AI voiceover",
                    "duration": 3000,
                    "ext": "mp3",
                },
                {
                    "id": "real",
                    "extractor_key": "Substack",
                    "webpage_url": "https://media.example.com/episode.mp3",
                    "title": "Real human episode",
                    "duration": 600,
                    "ext": "mp3",
                },
            ]
        },
    )

    rows = WebMediaConnector().resolve("https://example.com/post")
    assert [(row.media_id, row.kind) for row in rows] == [
        ("real", "podcast-enclosure"),
        ("voiceover", "page-voiceover"),
    ]


def test_web_resolver_demotes_youtube_hosted_tts_voiceover(monkeypatch):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    monkeypatch.setattr(
        "undertone_audio.connectors.web.run_json",
        lambda cmd, **kwargs: {
            "entries": [
                {
                    "id": "voiceover",
                    "extractor_key": "Youtube",
                    "webpage_url": "https://www.youtube.com/watch?v=voice",
                    "title": "AI voiceover",
                    "duration": 3000,
                },
                {
                    "id": "real",
                    "extractor_key": "Substack",
                    "webpage_url": "https://media.example.com/episode.mp3",
                    "title": "Real human episode",
                    "duration": 600,
                    "ext": "mp3",
                },
            ]
        },
    )

    rows = WebMediaConnector().resolve("https://example.com/post")
    assert [(row.media_id, row.kind) for row in rows] == [
        ("real", "podcast-enclosure"),
        ("voiceover", "page-voiceover"),
    ]


def test_web_resolver_ranks_longest_real_media_above_short_external_promo(monkeypatch):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    monkeypatch.setattr(
        "undertone_audio.connectors.web.run_json",
        lambda cmd, **kwargs: {
            "entries": [
                {
                    "id": "promo",
                    "extractor_key": "Youtube",
                    "webpage_url": "https://www.youtube.com/watch?v=promo",
                    "title": "Promo clip",
                    "duration": 30,
                },
                {
                    "id": "full",
                    "extractor_key": "Substack",
                    "webpage_url": "https://media.example.com/full-interview.mp3",
                    "title": "Full interview",
                    "duration": 7200,
                    "ext": "mp3",
                },
            ]
        },
    )

    rows = WebMediaConnector().resolve("https://example.com/post")
    assert [(row.media_id, row.kind, row.duration) for row in rows] == [
        ("full", "podcast-enclosure", 7200.0),
        ("promo", "external-video", 30.0),
    ]


def test_connector_resolve_human_output_redacts_media_url_tokens(monkeypatch, capsys):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    monkeypatch.setattr(
        "undertone_audio.connectors.web.run_json",
        lambda cmd, **kwargs: {
            "id": "episode-1",
            "extractor_key": "Substack",
            "webpage_url": None,
            "url": "https://user:pass@cdn.example.com/audio.mp3?token=SECRET#frag",
            "title": "Signed audio",
        },
    )

    assert main(["connector-resolve", "https://example.com/post"]) == 0
    out = capsys.readouterr().out
    assert "https://cdn.example.com/audio.mp3" in out
    assert "SECRET" not in out
    assert "user:pass" not in out
    assert "#frag" not in out


def test_connector_resolve_human_output_redacts_webpage_url_tokens(monkeypatch, capsys):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    monkeypatch.setattr(
        "undertone_audio.connectors.web.run_json",
        lambda cmd, **kwargs: {
            "id": "episode-1",
            "extractor_key": "Substack",
            "webpage_url": "https://user:pass@cdn.example.com/page.mp3?token=SECRET#frag",
            "url": "https://cdn.example.com/audio.mp3?token=OTHER",
            "title": "Signed page",
        },
    )

    assert main(["connector-resolve", "https://example.com/post"]) == 0
    out = capsys.readouterr().out
    assert "https://cdn.example.com/page.mp3" in out
    assert "SECRET" not in out
    assert "OTHER" not in out
    assert "user:pass" not in out
    assert "#frag" not in out


def test_connector_resolve_output_redacts_title_urls(monkeypatch, capsys):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    monkeypatch.setattr(
        "undertone_audio.connectors.web.run_json",
            lambda cmd, **kwargs: {
                "id": "episode-1",
                "extractor_key": "Substack",
                "webpage_url": "https://example.com/post",
                "url": "https://cdn.example.com/audio.mp3",
                "title": "listen https://user:pass@cdn.example.com/title.mp3?token=TITLE#frag",
            },
    )

    assert main(["connector-resolve", "https://example.com/post", "--json"]) == 0
    json_out = capsys.readouterr().out
    assert "https://cdn.example.com/title.mp3" in json_out
    assert "TITLE" not in json_out
    assert "user:pass" not in json_out

    assert main(["connector-resolve", "https://example.com/post"]) == 0
    human_out = capsys.readouterr().out
    assert "https://cdn.example.com/title.mp3" in human_out
    assert "TITLE" not in human_out
    assert "user:pass" not in human_out


def test_connector_resolve_human_output_redacts_extractor_fields(monkeypatch, capsys):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    monkeypatch.setattr(
        "undertone_audio.connectors.web.run_json",
            lambda cmd, **kwargs: {
                "id": "episode-1",
                "extractor": "Extractor https://user:pass@extractor.example/e.mp3?token=EXT#frag",
                "extractor_key": "Key https://user:pass@extractor.example/k.mp3?token=KEY#frag",
                "webpage_url": "https://example.com/post",
                "url": "https://cdn.example.com/audio.mp3",
                "title": "Extractor field",
            },
    )

    assert main(["connector-resolve", "https://example.com/post"]) == 0
    out = capsys.readouterr().out
    assert "https://extractor.example/k.mp3" in out
    assert "KEY" not in out
    assert "EXT" not in out
    assert "user:pass" not in out
    assert "#frag" not in out


def test_connector_resolve_reason_redacts_external_error_urls(monkeypatch, capsys):
    from undertone_audio.connectors import ConnectorError

    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)

    def fail_json(cmd, **kwargs):
        raise ConnectorError(
            "failed https://user:pass@media.example/audio.mp3?token=SECRET&sig=abc"
        )

    monkeypatch.setattr("undertone_audio.connectors.web.run_json", fail_json)

    assert main(["connector-resolve", "https://example.com/post", "--json"]) == 1
    blob = capsys.readouterr().out
    assert "https://media.example/audio.mp3" in blob
    assert "SECRET" not in blob
    assert "sig=abc" not in blob
    assert "user:pass" not in blob


def test_run_json_redacts_external_failure_urls(monkeypatch):
    from undertone_audio.connectors.base import run_json

    def fail_process(*args, **kwargs):
        raise RuntimeError(
            "failed https://user:pass@example.com/a.mp3?token=SECRET#frag"
        )

    monkeypatch.setattr("undertone_audio.connectors.base.run_process_sync", fail_process)

    try:
        run_json(["yt-dlp"])
    except Exception as exc:
        message = str(exc)
    else:
        raise AssertionError("run_json should fail")
    assert "https://example.com/a.mp3" in message
    assert "SECRET" not in message
    assert "user:pass" not in message
    assert "#frag" not in message


def test_connector_no_match_redacts_ref_in_errors(capsys):
    url = "https://user:pass@example.com/post?token=SECRET&sig=abc#frag"
    assert main(["connector-ingest", url]) == 1
    err = capsys.readouterr().err
    assert "https://example.com/post" in err
    assert "SECRET" not in err
    assert "sig=abc" not in err
    assert "user:pass" not in err
    assert "#frag" not in err


def test_connector_no_match_progress_json_redacts_ref(capsys):
    url = "https://user:pass@example.com/post?token=SECRET&sig=abc#frag"
    assert main(["connector-ingest", url, "--progress", "json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert "https://example.com/post" in payload["error"]
    assert "SECRET" not in captured.err
    assert "sig=abc" not in captured.err
    assert "user:pass" not in captured.err
    assert "#frag" not in captured.err


def test_connector_ingest_progress_redacts_matched_ref(tmp_path, monkeypatch, capsys):
    class FixtureConnector:
        name = "fixture"
        source_kind = "fixture-audio"

        def matches(self, ref):
            return True

        def fetch(self, ref):
            from undertone_audio.connectors import ConnectorAsset

            path = tmp_path / "fixture.wav"
            path.write_bytes(b"audio")
            return ConnectorAsset(
                audio_path=path,
                source_url=ref,
                source_kind="fixture-audio",
                transcript_id_hint="fixture-1",
            )

    class FakeEngine:
        async def transcribe(self, audio_path: Path):
            return RawTranscript(
                duration_ms=1000,
                language="en",
                engine="fixture-engine",
                speakers=[Speaker(speaker_id="S1")],
                segments=[
                    Segment(
                        segment_id="s1",
                        speaker_id="S1",
                        start_ms=0,
                        end_ms=1000,
                        text="progress",
                    )
                ],
            )

    monkeypatch.setattr(
        "undertone_audio.commands.connectors.connector_for_ref",
        lambda ref, preferred=None: FixtureConnector(),
    )
    monkeypatch.setattr("undertone_audio.commands.connectors.create_engine", lambda name, config: FakeEngine())
    monkeypatch.setenv("UNDERTONE_WEBHOOK_ENABLED", "0")

    ref = "https://user:pass@example.com/audio.mp3?token=SECRET#frag"
    assert (
        main(
            [
                "--db",
                str(tmp_path / "undertone.db"),
                "connector-ingest",
                ref,
                "--progress",
                "json",
            ]
        )
        == 0
    )
    first = json.loads(capsys.readouterr().err.splitlines()[0])
    assert first["event"] == "fetching"
    assert first["ref"] == "https://example.com/audio.mp3"
    assert "SECRET" not in json.dumps(first)
    assert "user:pass" not in json.dumps(first)


def test_connector_resolve_human_output_redacts_reason_urls():
    from undertone_audio.commands.connectors import _render_candidates
    from undertone_audio.connectors import ConnectorCandidate

    candidate = ConnectorCandidate(
        candidate_id="c1",
        original_url="https://example.com/post",
        reason="failed https://user:pass@media.example/a.mp3?token=SECRET#frag",
        metadata={"debug_url": "https://user:pass@cdn.example/a.mp3?sig=SECRET"},
    )

    rendered = _render_candidates([candidate])
    assert "https://media.example/a.mp3" in rendered
    assert "SECRET" not in rendered
    assert "user:pass" not in rendered
    assert "#frag" not in rendered


def test_web_ingest_refuses_ambiguous_non_tty_without_selection(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    monkeypatch.setattr(
        "undertone_audio.connectors.web.run_json",
        lambda cmd, **kwargs: {
            "entries": [
                {"id": "a", "extractor_key": "Youtube", "webpage_url": "https://youtu.be/a", "duration": 10},
                {"id": "b", "extractor_key": "Youtube", "webpage_url": "https://youtu.be/b", "duration": 20},
            ]
        },
    )

    assert (
        main(
            [
                "--db",
                str(tmp_path / "undertone.db"),
                "web-ingest",
                "https://example.com/post",
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "multiple downloadable candidates" in err
    assert "--select" in err


def test_web_ingest_refuses_ambiguous_yes_without_selection(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    monkeypatch.setattr(
        "undertone_audio.connectors.web.run_json",
        lambda cmd, **kwargs: {
            "entries": [
                {"id": "a", "extractor_key": "Substack", "webpage_url": "https://a.example.com/a.mp3", "duration": 10},
                {"id": "b", "extractor_key": "Substack", "webpage_url": "https://b.example.com/b.mp3", "duration": 20},
            ]
        },
    )

    assert (
        main(
            [
                "--db",
                str(tmp_path / "undertone.db"),
                "web-ingest",
                "https://example.com/post",
                "--yes",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "multiple downloadable candidates" in captured.err
    assert "--select" in captured.err
    assert "--yes" not in captured.err


def test_web_ingest_refuses_single_non_tty_without_yes_or_selection(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    monkeypatch.setattr(
        "undertone_audio.connectors.web.run_json",
        lambda cmd, **kwargs: {
            "id": "single",
            "extractor_key": "Substack",
            "webpage_url": "https://media.example.com/single.mp3",
            "title": "Single",
            "duration": 120,
            "ext": "mp3",
        },
    )

    def fail_run_checked(cmd, **kwargs):
        raise AssertionError("web-ingest should require explicit consent before fetching")

    monkeypatch.setattr("undertone_audio.connectors.web.run_checked", fail_run_checked)

    assert (
        main(
            [
                "--db",
                str(tmp_path / "undertone.db"),
                "web-ingest",
                "https://example.com/post",
                "--dry-run",
                "--json",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "requires --yes or --select" in captured.err


def test_web_ingest_yes_allows_single_downloadable_candidate(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    monkeypatch.setattr(
        "undertone_audio.connectors.web.run_json",
        lambda cmd, **kwargs: {
            "id": "single",
            "extractor_key": "Substack",
            "webpage_url": "https://media.example.com/single.mp3",
            "title": "Single",
            "duration": 120,
            "ext": "mp3",
        },
    )

    def fake_run_checked(cmd, **kwargs):
        download_dir = Path(cmd[cmd.index("--paths") + 1])
        stem = cmd[cmd.index("-o") + 1].split(".%")[0]
        (download_dir / f"{stem}.wav").write_bytes(b"audio")

    monkeypatch.setattr("undertone_audio.connectors.web.run_checked", fake_run_checked)

    assert (
        main(
            [
                "web-ingest",
                "https://example.com/post",
                "--yes",
                "--dry-run",
                "--json",
                "--download-dir",
                str(tmp_path / "downloads"),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_kind"] == "web-media-audio"
    assert payload["metadata"]["candidate_id"]


def test_web_ingest_selects_second_disambiguated_candidate(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)

    def fake_run_json(cmd, **kwargs):
        return {
            "entries": [
                    {
                        "id": "dup",
                        "extractor_key": "Generic",
                        "webpage_url": "https://example.com/post",
                        "url": "https://cdn.example.com/one.mp3",
                        "title": "One",
                    },
                    {
                        "id": "dup",
                        "extractor_key": "Generic",
                        "webpage_url": "https://example.com/post",
                        "url": "https://cdn.example.com/two.mp3",
                        "title": "Two",
                    },
            ]
        }

    def fake_run_checked(cmd, **kwargs):
        download_dir = Path(cmd[cmd.index("--paths") + 1])
        stem = cmd[cmd.index("-o") + 1].split(".%")[0]
        (download_dir / f"{stem}.wav").write_bytes(b"audio")

    monkeypatch.setattr("undertone_audio.connectors.web.run_json", fake_run_json)
    monkeypatch.setattr("undertone_audio.connectors.web.run_checked", fake_run_checked)

    selected = next(row for row in WebMediaConnector().resolve("https://example.com/post") if row.title == "Two")
    assert (
        main(
            [
                "web-ingest",
                "https://example.com/post",
                "--select",
                selected.candidate_id,
                "--dry-run",
                "--json",
                "--download-dir",
                str(tmp_path / "downloads"),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["title"] == "Two"
    assert payload["metadata"]["candidate_id"] == selected.candidate_id


def test_web_ingest_skip_existing_transcript_id_does_not_resolve(
    tmp_path,
    monkeypatch,
    capsys,
):
    db = tmp_path / "undertone.db"
    _save_existing_transcript(tmp_path, db, "existing-web")
    capsys.readouterr()

    def fail_web_connector_from_args(args, config=None):
        raise AssertionError("web connector should not be constructed")

    monkeypatch.setattr(
        "undertone_audio.commands.connectors._web_connector_from_args",
        fail_web_connector_from_args,
    )

    assert (
        main(
            [
                "--db",
                str(db),
                "web-ingest",
                "https://example.com/post",
                "--transcript-id",
                "existing-web",
                "--skip-existing",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["skipped"] is True


def test_web_ingest_skip_existing_selected_id_does_not_resolve(
    tmp_path,
    monkeypatch,
    capsys,
):
    db = tmp_path / "undertone.db"
    _save_existing_transcript(tmp_path, db, "web-knowncandidate")
    capsys.readouterr()

    def fail_web_connector_from_args(args, config=None):
        raise AssertionError("web connector should not be constructed")

    monkeypatch.setattr(
        "undertone_audio.commands.connectors._web_connector_from_args",
        fail_web_connector_from_args,
    )

    assert (
        main(
            [
                "--db",
                str(db),
                "web-ingest",
                "https://example.com/post",
                "--select",
                "knowncandidate",
                "--skip-existing",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["skipped"] is True


def test_web_ingest_selected_candidate_runs_local_pipeline(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)

    def fake_run_json(cmd, **kwargs):
        return {
            "entries": [
                {
                    "id": "voice",
                    "extractor_key": "Substack",
                    "webpage_url": "https://example.com/post?token=VOICE",
                    "title": "AI voiceover",
                    "duration": 300,
                    "ext": "mp3",
                },
                {
                    "id": "yt-real",
                    "extractor_key": "Youtube",
                    "webpage_url": "https://www.youtube.com/watch?v=real&token=SECRET",
                    "title": "Real interview",
                    "duration": 1255,
                },
            ]
        }

    def fake_run_checked(cmd, **kwargs):
        download_dir = Path(cmd[cmd.index("--paths") + 1])
        stem = cmd[cmd.index("-o") + 1].split(".%")[0]
        (download_dir / f"{stem}.wav").write_bytes(b"audio")

    class FakeEngine:
        async def transcribe(self, audio_path: Path):
            assert audio_path.suffix == ".wav"
            return RawTranscript(
                duration_ms=1000,
                language="en",
                engine="fixture-engine",
                speakers=[Speaker(speaker_id="S1")],
                segments=[
                    Segment(
                        segment_id="s1",
                        speaker_id="S1",
                        start_ms=0,
                        end_ms=1000,
                        text="web media",
                    )
                ],
            )

    monkeypatch.setattr("undertone_audio.connectors.web.run_json", fake_run_json)
    monkeypatch.setattr("undertone_audio.connectors.web.run_checked", fake_run_checked)
    monkeypatch.setattr("undertone_audio.commands.connectors.create_engine", lambda name, config: FakeEngine())
    monkeypatch.setenv("UNDERTONE_WEBHOOK_ENABLED", "0")

    selected = WebMediaConnector().resolve("https://example.com/post")[0]

    assert (
        main(
            [
                "--db",
                str(tmp_path / "undertone.db"),
                "web-ingest",
                "https://example.com/post",
                "--select",
                selected.candidate_id,
                "--download-dir",
                str(tmp_path / "downloads"),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["transcript_id"] == f"web-{selected.candidate_id}"
    metadata = payload["metadata"]["source_metadata"]
    assert payload["metadata"]["source_url"] == "https://example.com/post"
    assert metadata["source"] == "web"
    assert metadata["source_kind"] == "web-media-audio"
    assert metadata["candidate_kind"] == "external-video"
    assert metadata["webpage_url"] == "https://www.youtube.com/watch?v=real"
    assert "media_url" not in metadata
    assert "SECRET" not in json.dumps(payload)


def test_web_fetch_candidate_downloads_selected_direct_media_url(monkeypatch, tmp_path):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    selected_refs = []

    monkeypatch.setattr(
        "undertone_audio.connectors.web.run_json",
        lambda cmd, **kwargs: {
            "entries": [
                {
                    "id": "voice",
                    "extractor_key": "Substack",
                    "webpage_url": "https://example.com/post",
                    "url": "https://cdn.example.com/voice.mp3?sig=voice",
                    "title": "AI voiceover",
                    "duration": 300,
                    "ext": "mp3",
                },
                {
                    "id": "real",
                    "extractor_key": "Substack",
                    "webpage_url": "https://example.com/post",
                    "url": "https://cdn.example.com/real.mp3?sig=real",
                    "title": "Real audio",
                    "duration": 1200,
                    "ext": "mp3",
                },
            ]
        },
    )

    def fake_run_checked(cmd, **kwargs):
        selected_refs.append(cmd[-1])
        download_dir = Path(cmd[cmd.index("--paths") + 1])
        stem = cmd[cmd.index("-o") + 1].split(".%")[0]
        (download_dir / f"{stem}.wav").write_bytes(b"audio")

    monkeypatch.setattr("undertone_audio.connectors.web.run_checked", fake_run_checked)

    connector = WebMediaConnector(download_dir=tmp_path / "downloads")
    candidate = next(row for row in connector.resolve("https://example.com/post") if row.media_id == "real")
    asset = connector.fetch_candidate(candidate)

    assert asset.audio_path.exists()
    assert selected_refs == ["https://cdn.example.com/real.mp3?sig=real"]
    assert selected_refs[0] != "https://example.com/post"


def test_web_fetch_candidate_filename_uses_candidate_id_not_media_id(monkeypatch, tmp_path):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    stems = []

    def fake_run_checked(cmd, **kwargs):
        stem = cmd[cmd.index("-o") + 1].split(".%")[0]
        stems.append(stem)
        download_dir = Path(cmd[cmd.index("--paths") + 1])
        (download_dir / f"{stem}.wav").write_bytes(b"audio")

    monkeypatch.setattr("undertone_audio.connectors.web.run_checked", fake_run_checked)

    from undertone_audio.connectors import ConnectorCandidate

    candidate = ConnectorCandidate(
        candidate_id="safe-candidate",
        original_url="https://example.com/post",
        url="https://cdn.example.com/audio.mp3",
        media_id="AbCDefGhIjKlMnOpQrStUvWxYz1234567890",
    )
    asset = WebMediaConnector(download_dir=tmp_path / "downloads").fetch_candidate(candidate)

    assert stems == ["safe-candidate"]
    assert asset.audio_path.name == "safe-candidate.wav"
    assert asset.metadata["media_id"] == "[redacted]"


def test_web_fetch_candidate_redacts_token_like_ids_in_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)

    def fake_run_checked(cmd, **kwargs):
        stem = cmd[cmd.index("-o") + 1].split(".%")[0]
        download_dir = Path(cmd[cmd.index("--paths") + 1])
        (download_dir / f"{stem}.wav").write_bytes(b"audio")

    monkeypatch.setattr("undertone_audio.connectors.web.run_checked", fake_run_checked)

    from undertone_audio.connectors import ConnectorCandidate

    candidate = ConnectorCandidate(
        candidate_id="safe-candidate",
        original_url="https://example.com/post",
        url="https://cdn.example.com/audio.mp3",
        media_id="AbCDefGhIjKlMnOpQrStUvWxYz1234567890",
        format_id=(
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkphbmUifQ."
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        ),
    )
    asset = WebMediaConnector(download_dir=tmp_path / "downloads").fetch_candidate(candidate)

    blob = json.dumps(asset.metadata)
    assert "AbCDef" not in blob
    assert "eyJhbGci" not in blob
    assert asset.metadata["media_id"] == "[redacted]"
    assert asset.metadata["format_id"] == "[redacted]"


def test_web_resolver_assigns_unique_ids_when_yt_dlp_duplicates_media_ids(monkeypatch):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    monkeypatch.setattr(
        "undertone_audio.connectors.web.run_json",
        lambda cmd, **kwargs: {
            "entries": [
                {
                    "id": "dup",
                    "extractor_key": "Substack",
                    "webpage_url": "https://example.com/post",
                    "url": "https://cdn.example.com/one.mp3",
                    "title": "One",
                },
                {
                    "id": "dup",
                    "extractor_key": "Substack",
                    "webpage_url": "https://example.com/post",
                    "url": "https://cdn.example.com/two.mp3",
                    "title": "Two",
                },
            ]
        },
    )

    ids = [row.candidate_id for row in WebMediaConnector().resolve("https://example.com/post")]
    assert len(ids) == 2
    assert len(set(ids)) == 2


def test_web_fetch_external_video_prefers_extractor_webpage_url(monkeypatch, tmp_path):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    selected_refs = []

    def fake_run_checked(cmd, **kwargs):
        selected_refs.append(cmd[-1])
        download_dir = Path(cmd[cmd.index("--paths") + 1])
        stem = cmd[cmd.index("-o") + 1].split(".%")[0]
        (download_dir / f"{stem}.wav").write_bytes(b"audio")

    monkeypatch.setattr("undertone_audio.connectors.web.run_checked", fake_run_checked)

    from undertone_audio.connectors import ConnectorCandidate

    candidate = ConnectorCandidate(
        candidate_id="yt",
        original_url="https://example.com/post",
        extractor_key="Youtube",
        webpage_url="https://www.youtube.com/watch?v=real",
        url="https://rr1---sn.example.com/videoplayback?expire=1&sig=SECRET",
        media_id="real",
        kind="external-video",
    )
    WebMediaConnector(download_dir=tmp_path / "downloads").fetch_candidate(candidate)

    assert selected_refs == ["https://www.youtube.com/watch?v=real"]


def test_web_ingest_dry_run_redacts_asset_urls(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    monkeypatch.setattr(
        "undertone_audio.connectors.web.run_json",
        lambda cmd, **kwargs: {
            "id": "episode-1",
            "extractor_key": "Substack",
            "webpage_url": "https://user:pass@cdn.example.com/page.mp3?token=PAGE#frag",
            "url": "https://cdn.example.com/audio.mp3?token=MEDIA",
            "title": "Signed https://cdn.example.com/title.mp3?token=ASSET#frag",
        },
    )

    def fake_run_checked(cmd, **kwargs):
        download_dir = Path(cmd[cmd.index("--paths") + 1])
        stem = cmd[cmd.index("-o") + 1].split(".%")[0]
        (download_dir / f"{stem}.wav").write_bytes(b"audio")

    monkeypatch.setattr("undertone_audio.connectors.web.run_checked", fake_run_checked)

    assert (
        main(
            [
                "web-ingest",
                "https://example.com/post?ref=SECRET",
                "--yes",
                "--dry-run",
                "--json",
                "--download-dir",
                str(tmp_path / "downloads"),
            ]
        )
        == 0
    )
    blob = capsys.readouterr().out
    assert "https://example.com/post" in blob
    assert "https://cdn.example.com/page.mp3" in blob
    assert "SECRET" not in blob
    assert "PAGE" not in blob
    assert "MEDIA" not in blob
    assert "ASSET" not in blob
    assert "user:pass" not in blob


def test_web_ingest_download_error_redacts_signed_url(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    monkeypatch.setattr(
        "undertone_audio.connectors.web.run_json",
        lambda cmd, **kwargs: {
            "id": "episode-1",
            "extractor_key": "Substack",
            "webpage_url": "https://example.com/post",
            "url": "https://cdn.example.com/audio.mp3",
            "title": "Download failure",
        },
    )

    def fail_process(cmd, **kwargs):
        raise RuntimeError(
            "download failed at https://user:pass@example.com/audio.mp3?token=DOWNLOADSECRET#frag"
        )

    monkeypatch.setattr("undertone_audio.connectors.base.run_process_sync", fail_process)

    assert (
        main(
            [
                "web-ingest",
                "https://example.com/post",
                "--yes",
                "--dry-run",
                "--download-dir",
                str(tmp_path / "downloads"),
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "https://example.com/audio.mp3" in err
    assert "DOWNLOADSECRET" not in err
    assert "user:pass" not in err
    assert "#frag" not in err


def test_connector_ingest_persisted_metadata_redacts_tuple_urls(monkeypatch, tmp_path, capsys):
    class TupleConnector:
        name = "tuple"
        source_kind = "tuple-audio"

        def matches(self, ref):
            return ref == "tuple:audio"

        def fetch(self, ref):
            from undertone_audio.connectors import ConnectorAsset

            path = tmp_path / "tuple.wav"
            path.write_bytes(b"audio")
            return ConnectorAsset(
                audio_path=path,
                source_url="https://example.com/post?token=SOURCE",
                source_kind="tuple-audio",
                transcript_id_hint="tuple-1",
                metadata={
                    "tuple": ("https://user:pass@example.com/m.mp3?token=SECRET#frag",),
                    "https://user:pass@example.com/key.mp3?token=KEY#frag": "key value",
                },
            )

    class FakeEngine:
        async def transcribe(self, audio_path: Path):
            return RawTranscript(
                duration_ms=1000,
                language="en",
                engine="fixture-engine",
                speakers=[Speaker(speaker_id="S1")],
                segments=[
                    Segment(
                        segment_id="s1",
                        speaker_id="S1",
                        start_ms=0,
                        end_ms=1000,
                        text="tuple metadata",
                    )
                ],
            )

    monkeypatch.setattr(
        "undertone_audio.commands.connectors.connector_for_ref",
        lambda ref, preferred=None: TupleConnector(),
    )
    monkeypatch.setattr("undertone_audio.commands.connectors.create_engine", lambda name, config: FakeEngine())
    monkeypatch.setenv("UNDERTONE_WEBHOOK_ENABLED", "0")

    db = tmp_path / "undertone.db"
    assert main(["--db", str(db), "connector-ingest", "tuple:audio"]) == 0
    payload = json.loads(capsys.readouterr().out)
    blob = json.dumps(payload)
    assert payload["metadata"]["source_url"] == "https://example.com/post"
    assert payload["metadata"]["source_metadata"]["tuple"] == ["https://example.com/m.mp3"]
    assert "https://example.com/key.mp3" in payload["metadata"]["source_metadata"]
    assert "SECRET" not in blob
    assert "SOURCE" not in blob
    assert "KEY" not in blob
    assert "user:pass" not in blob


def test_web_connector_fetch_refuses_ambiguous_url(monkeypatch):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)
    monkeypatch.setattr(
        "undertone_audio.connectors.web.run_json",
        lambda cmd, **kwargs: {
            "entries": [
                {"id": "a", "extractor_key": "Youtube", "webpage_url": "https://youtu.be/a", "duration": 10},
                {"id": "b", "extractor_key": "Youtube", "webpage_url": "https://youtu.be/b", "duration": 20},
            ]
        },
    )

    try:
        WebMediaConnector().fetch("https://example.com/post")
    except Exception as exc:
        assert "ambiguous" in str(exc)
        assert "connector-resolve" in str(exc)
    else:
        raise AssertionError("ambiguous web fetch should fail")


def test_web_connector_rejects_private_hosts(monkeypatch):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    try:
        WebMediaConnector().resolve("http://127.0.0.1/audio")
    except Exception as exc:
        assert "private/local host" in str(exc)
    else:
        raise AssertionError("private web URL should be rejected")


def test_web_connector_rejects_non_global_dns(monkeypatch):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr(
        "undertone_audio.connectors.web.socket.getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, ("100.64.0.1", 443))],
    )

    try:
        WebMediaConnector().resolve("https://example.com/post")
    except Exception as exc:
        assert "100.64.0.1" in str(exc)
    else:
        raise AssertionError("non-global DNS target should be rejected")


def test_web_connector_rejects_private_resolved_media_url(monkeypatch, tmp_path):
    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)

    def fake_getaddrinfo(host, *args, **kwargs):
        ip = "127.0.0.1" if host == "127.0.0.1" else "93.184.216.34"
        return [(None, None, None, None, (ip, 443))]

    monkeypatch.setattr("undertone_audio.connectors.web.socket.getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(
        "undertone_audio.connectors.web.run_json",
        lambda cmd, **kwargs: {
            "id": "bad",
            "extractor_key": "Generic",
            "webpage_url": "https://example.com/post",
            "url": "http://127.0.0.1/private.mp3",
            "title": "Bad",
            "duration": 60,
        },
    )
    called = False

    def fail_if_called(cmd, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("undertone_audio.connectors.web.run_checked", fail_if_called)
    candidate = WebMediaConnector(download_dir=tmp_path).resolve("https://example.com/post")[0]

    try:
        WebMediaConnector(download_dir=tmp_path).fetch_candidate(candidate)
    except Exception as exc:
        assert "private/local host" in str(exc)
    else:
        raise AssertionError("private resolved media URL should be rejected")
    assert called is False


def test_web_max_download_size_flag_overrides_env(monkeypatch):
    monkeypatch.setenv("UNDERTONE_MAX_DOWNLOAD_SIZE", "9M")
    connector = WebMediaConnector(max_download_size="1M")
    cmd = connector._download_cmd("yt-dlp", "https://example.com/audio", "a", Path("/tmp"))
    assert cmd[cmd.index("--max-filesize") + 1] == "1M"


def test_web_yt_dlp_commands_ignore_ambient_config():
    connector = WebMediaConnector()
    info_cmd = connector._info_cmd("yt-dlp", "https://example.com/post")
    download_cmd = connector._download_cmd("yt-dlp", "https://example.com/audio", "a", Path("/tmp"))

    assert "--ignore-config" in info_cmd
    assert "--ignore-config" in download_cmd
    assert info_cmd.index("--ignore-config") < info_cmd.index("--dump-single-json")
    assert download_cmd.index("--ignore-config") < download_cmd.index("-f")


def test_connector_resolve_reports_unsupported_candidate(monkeypatch, capsys):
    from undertone_audio.connectors import ConnectorError

    monkeypatch.setattr("undertone_audio.connectors.web.ensure_binary", lambda binary: binary)
    monkeypatch.setattr("undertone_audio.connectors.web._validate_public_url", lambda value: None)

    def fail_json(cmd, **kwargs):
        raise ConnectorError("unsupported url")

    monkeypatch.setattr("undertone_audio.connectors.web.run_json", fail_json)

    assert main(["connector-resolve", "https://example.com/post", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["kind"] == "unsupported"
    assert payload[0]["availability"] == "unsupported"
    assert payload[0]["reason"] == "unsupported url"


def test_connector_candidate_schema_command(capsys):
    assert main(["schema", "connector-candidate"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "candidate_id" in payload["properties"]
    assert "availability" in payload["properties"]


def _save_existing_transcript(tmp_path, db: Path, transcript_id: str) -> None:
    raw_path = tmp_path / f"{transcript_id}.json"
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
                        "text": "existing",
                    }
                ],
            }
        )
    )
    assert main(["--db", str(db), "finalize-json", str(raw_path), "--transcript-id", transcript_id]) == 0


def test_connector_asset_rejects_unsupported_schema_version(tmp_path):
    from undertone_audio.connectors import ConnectorAsset, ConnectorError

    asset = ConnectorAsset(
        audio_path=tmp_path / "audio.wav",
        source_url="fixture:audio",
        source_kind="fixture",
        schema_version="2",
    )

    try:
        asset.to_schema()
    except ConnectorError as exc:
        assert "unsupported ConnectorAsset schema_version" in str(exc)
    else:
        raise AssertionError("unsupported connector asset version should fail")


def test_youtube_dry_run_passes_cli_process_timeout(monkeypatch, capsys):
    seen = {}

    class FakeConnector:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def fetch(self, url):
            from undertone_audio.connectors import ConnectorAsset

            return ConnectorAsset(
                audio_path=Path("/tmp/yt.wav"),
                source_url=url,
                source_kind="youtube-audio",
                transcript_id_hint="youtube-abc",
                metadata={"source": "youtube"},
            )

    monkeypatch.setattr("undertone_audio.commands.connectors.YouTubeConnector", FakeConnector)

    assert (
        main(
            [
                "youtube-ingest",
                "https://youtu.be/abc",
                "--dry-run",
                "--json",
                "--process-timeout-seconds",
                "9",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["transcript_id_hint"] == "youtube-abc"
    assert seen["process_timeout_seconds"] == 9


def test_youtube_ingest_missing_binary_names_fix(capsys):
    assert (
        main(
            [
                "youtube-ingest",
                "https://www.youtube.com/watch?v=abc",
                "--yt-dlp-bin",
                "definitely-not-yt-dlp",
                "--dry-run",
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "pip install -e '.[connectors]'" in err
    assert "doctor --check-yt-dlp" in err
