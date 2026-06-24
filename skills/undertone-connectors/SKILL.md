---
name: undertone-connectors
description: Use when installing or operating Undertone sidecar connectors for YouTube, podcasts, RSS feeds, direct media URLs, or other external media sources that should download audio and rerun local Undertone diarization.
---

# Undertone Connectors

Use this skill for sidecar source connectors. Connectors acquire audio and then hand that audio to the normal Undertone pipeline.

## Install

```bash
pip install -e '.[connectors,voice]'
```

`yt-dlp` is required for YouTube. Podcasts use Python standard-library RSS and download support. FluidAudio is still required for local ASR/diarization.

## Paths

Do not hardcode local machine paths. Use one of:

- `--download-dir /path/to/cache`
- `UNDERTONE_DOWNLOAD_DIR=/path/to/cache`
- default cache path from `XDG_CACHE_HOME/undertone/downloads` or `~/.cache/undertone/downloads`

Downloads publish atomically. A failed YouTube or podcast transfer should not leave a reusable media file in the cache. External downloader/process calls use `UNDERTONE_PROCESS_TIMEOUT_SECONDS`; set it only when long media needs a different bound.

## YouTube

```bash
undertone --db ./undertone.db youtube-ingest 'https://www.youtube.com/watch?v=...' \
  --engine fluidaudio-hybrid \
  --voice-metrics optional
```

Useful flags:

- `--download-dir`: explicit connector cache/output directory.
- `--yt-dlp-bin`: non-default `yt-dlp` binary path or name.
- `--audio-format wav`: audio format passed to `yt-dlp`.
- `--include-playlist`: allow playlist processing instead of forcing a single video.
- `--dry-run`: download/select audio and print the connector asset without ingesting.
- `--json`: print machine-readable dry-run output.

## Podcasts

```bash
undertone podcast-list 'https://example.com/feed.xml' --limit 20
undertone --db ./undertone.db podcast-ingest 'https://example.com/feed.xml' --episode 0
undertone --db ./undertone.db podcast-ingest 'https://example.com/feed.xml' --title-contains 'interview'
undertone --db ./undertone.db podcast-ingest 'https://cdn.example.com/episode.mp3'
```

Podcast feeds are selected by zero-based `--episode` index or first title match. Direct audio URLs skip RSS parsing.
`podcast-list` is human-readable by default. For agents and scripts, add `--json` for machine-readable output.

## Duplicate IDs

Connector ingest fails on a duplicate transcript id rather than overwriting. Pass `--force` to overwrite or `--skip-existing` to no-op.

## Quality Rule

Connectors should not use captions, feed notes, or external speaker labels as the transcript population path when audio is available. Download audio, then run Undertone local ASR, diarization, embeddings, fingerprinting, and enrichment.
