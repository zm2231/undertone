# Undertone Connectors

Use this skill for sidecar source connectors. Connectors acquire audio and then hand that audio to the normal Undertone pipeline.

## Install

```bash
pip install -e '.[connectors,voice]'
```

`yt-dlp` is required for YouTube and web media resolution. Podcasts use Python standard-library RSS and download support. FluidAudio is still required for local ASR/diarization.

## Plugin Contract

Connectors can be installed as Python entry-point plugins under `undertone.connectors`. A connector implements:

- `matches(ref) -> bool`
- `fetch(ref) -> ConnectorAsset`

`ConnectorAsset` is a versioned contract. Inspect it with:

```bash
undertone schema connector-asset
undertone schema connector-candidate
undertone connector-list
undertone --db ./undertone.db connector-ingest 'https://example.com/audio.mp3'
```

Keep source-specific credentials and cursors outside Undertone unless the connector explicitly owns them. Undertone should receive a ref and return audio plus source metadata.
Third-party connectors are additive. Built-in YouTube and podcast connectors stay available, and connector name collisions fail loudly instead of shadowing a built-in.

## Paths

Do not hardcode local machine paths. Use one of:

- `UNDERTONE_DOWNLOAD_DIR=/path/to/cache` for built-in connector defaults.
- `--download-dir /path/to/cache` on first-party source commands that expose it, such as `youtube-ingest` and `podcast-ingest`.
- default cache path from `XDG_CACHE_HOME/undertone/downloads` or `~/.cache/undertone/downloads`

Downloads publish atomically. A failed YouTube or podcast transfer should not leave a reusable media file in the cache. External downloader/process calls use `UNDERTONE_PROCESS_TIMEOUT_SECONDS`; set it only when long media needs a different bound.

## Web Media Resolver

Use this for article/content pages where the audio source must be resolved before ingest, such as a newsletter page with an embedded player plus links to YouTube or podcast platforms.

```bash
undertone connector-resolve 'https://example.com/article-with-audio'
undertone connector-resolve 'https://example.com/article-with-audio' --json
undertone --db ./undertone.db web-ingest 'https://example.com/article-with-audio' --list
undertone --db ./undertone.db web-ingest 'https://example.com/article-with-audio' --select <candidate-id>
undertone --db ./undertone.db web-ingest 'https://example.com/article-with-audio' --yes
```

`connector-resolve` and `web-ingest --list` are true metadata-only previews: they call yt-dlp's info path and do not download media. Candidates have stable `candidate_id` values for scripts. Ranking prefers non-voiceover real media, then the longest duration. If more than one downloadable candidate is found, `web-ingest` requires `--select <candidate-id>`; `--yes` only skips confirmation for a single downloadable candidate.

Only candidates with a concrete media URL or extractor URL are downloadable. If yt-dlp reports multiple entries that all point back to the original article page, treat them as provenance only; Undertone marks them not directly downloadable so `--select` cannot silently fetch the wrong recording.

`web-ingest` is explicit and force-only. The built-in web connector does not auto-match arbitrary URLs, so it cannot steal URLs from YouTube, podcast, or third-party connectors. `youtube-ingest` accepts only YouTube hosts; use `web-ingest` for article pages.

Security defaults for arbitrary web URLs:

- localhost, `.local`, loopback, link-local, and non-global user URLs are refused before yt-dlp runs.
- selected candidate media URLs are preflight-validated again before download.
- selected downloads pass `--max-filesize` to yt-dlp; set `--max-download-size` or `UNDERTONE_MAX_DOWNLOAD_SIZE` when needed.
- cookies are explicit only: `--cookies` or `--cookies-from-browser`; Undertone passes `--ignore-config` so ambient yt-dlp config is not loaded.

This is not a full network sandbox for hostile pages: yt-dlp performs its own DNS, redirects, extractor logic, and network I/O after Undertone's preflight checks. Use `web-ingest` as a local trusted-URL tool; do not expose it as a hosted/server-side fetch endpoint without an external egress sandbox or network policy.

## YouTube

```bash
undertone --db ./undertone.db youtube-ingest 'https://www.youtube.com/watch?v=...' \
  --engine fluidaudio-hybrid \
  --voice-metrics optional \
  --progress json
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
