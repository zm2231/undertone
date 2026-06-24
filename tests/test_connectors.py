import json
from pathlib import Path

from undertone_audio.cli import main
from undertone_audio.connectors.podcast import PodcastConnector
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
