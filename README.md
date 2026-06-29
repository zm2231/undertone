<div align="center">

# 🎙️ Undertone

**On-device audio transcripts that capture the words and how they were said.**

[![PyPI](https://img.shields.io/pypi/v/undertone-audio?logo=pypi&logoColor=white&color=3775A9)](https://pypi.org/project/undertone-audio/)
[![CI](https://github.com/zm2231/undertone/actions/workflows/ci.yml/badge.svg)](https://github.com/zm2231/undertone/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-000000?logo=apple&logoColor=white)
![License](https://img.shields.io/badge/license-Apache--2.0-D22128)
![Audio](https://img.shields.io/badge/audio-on--device-2ea44f)

</div>

A standard transcript gives you the words. It misses the undertones: where the pauses and overlaps fall, who interrupted whom, filler words, speaking pace, talk balance, and per-speaker voice quality such as pitch, jitter, and shimmer. Undertone captures both, and stores them as one structured, speaker-attributed transcript.

It ingests audio from local files or source connectors (YouTube, podcasts, Quill, Google Meet), then runs transcription, diarization, speaker embeddings, and enrichment locally through FluidAudio. Results are stored in SQLite, exportable as JSON, Markdown, text, or CSV, and can trigger a webhook when a transcript is ready. Audio never leaves the machine.

## What You Get

Each transcript is stored with three layers.

**Words and speakers.** Diarized, speaker-attributed text with per-segment and optional per-word timings, plus stable cross-recording speaker fingerprints.

**Confidence.** FluidAudio word ASR confidence is preserved on `words[].confidence`, and Undertone derives `segments[].asr_confidence` as the average confidence of words in that segment. `segments[].diarization_quality` is nullable: `fluidaudio-cli` preserves FluidAudio process `qualityScore`; `fluidaudio-hybrid` overlap-maps process `qualityScore` onto Sortformer spans when possible; `fluidaudio-pyannote` reports `null` because pyannote's public diarization output does not expose per-span confidence.

**Per-segment enrichment** (`SegmentEnrichment`):

- `is_interruption`, `overlap_with_prev_ms`: who cut in, and by how much
- `gap_before_ms`: the silence before a segment
- `fillers`: counted "um", "uh", and similar
- `sentiment`, `tone_tags`, `linguistic`: text-derived enrichment

**Per-speaker metrics** (`SpeakerVoiceMetrics`):

- `talk_ratio`, `talk_time_ms`, `word_count`, `wpm`: how much each speaker held the floor and how fast
- `pause_count`, `avg_pause_ms`: hesitation profile
- `interruptions_made`, `interruptions_received`: turn dynamics
- `filler_count`, `filler_rate`: disfluency
- `f0_mean_hz`, `f0_stdev_hz`, `jitter_local`, `shimmer_local`, `voiced_duration_s`, `articulation_rate`: acoustic voice quality, when voice metrics are enabled

## Requirements

- macOS on Apple Silicon
- Python 3.11+
- `fluidaudiocli`, built from [FluidInference/FluidAudio](https://github.com/FluidInference/FluidAudio) (see [Install](#install)). FluidAudio is a Swift SDK for on-device audio AI using Core ML and the Apple Neural Engine. Undertone does not vendor it; it shells out to the CLI that FluidAudio builds.
- `yt-dlp`, for the YouTube connector and arbitrary web media resolution
- Google Application Default Credentials, only for Google Meet
- A local Quill database and recordings, only for Quill ingest

## Install

### 1. Build the FluidAudio CLI

```bash
git clone https://github.com/FluidInference/FluidAudio.git
cd FluidAudio
swift build -c release --product fluidaudiocli
```

Then put it on `PATH`:

```bash
mkdir -p "$HOME/bin"
ln -sf "$PWD/.build/release/fluidaudiocli" "$HOME/bin/fluidaudiocli"
export PATH="$HOME/bin:$PATH"
```

or point Undertone at it directly:

```bash
export UNDERTONE_FLUIDAUDIO_CLI="$PWD/.build/release/fluidaudiocli"
```

FluidAudio downloads model assets from Hugging Face on first run. Upstream honors `REGISTRY_URL` / `MODEL_REGISTRY_URL` for mirrors and `https_proxy` for proxy routing, which matters on locked-down networks.

### 2. Install Undertone

From [PyPI](https://pypi.org/project/undertone-audio/):

```bash
pip install undertone-audio
```

Optional extras:

```bash
pip install 'undertone-audio[voice]'       # Parselmouth acoustic metrics
pip install 'undertone-audio[pyannote]'    # Optional pyannote diarization backend
pip install 'undertone-audio[meet]'        # Google Meet auth helpers
pip install 'undertone-audio[connectors]'  # YouTube + web media resolution via yt-dlp
pip install 'undertone-audio[voice,pyannote,meet,connectors]'
```

Or from source, for development:

```bash
pip install -e '.[dev]'
pip install -e '.[dev,voice,pyannote,meet,connectors]'
```

### 3. Verify

```bash
command -v fluidaudiocli
undertone --help
undertone models
undertone doctor
undertone doctor --check-yt-dlp --yt-dlp-bin /path/to/yt-dlp
```

If you installed the optional pyannote backend, check that its Python dependency imports:

```bash
undertone doctor --check-pyannote
```

This does not download or load a gated Hugging Face model; model access is verified when the backend runs.

## Quick Start

Run a local audio file:

```bash
UNDERTONE_WEBHOOK_ENABLED=0 undertone --db ./undertone.db run-wav ./meeting.wav \
  --transcript-id meeting-1 \
  --engine fluidaudio-hybrid \
  --voice-metrics optional \
  --output-format json \
  --output-detail standard \
  --output ./meeting.json
```

Load it later:

```bash
undertone --db ./undertone.db load meeting-1 --output-format text --output-detail minimal
undertone --db ./undertone.db list --limit 20
```

Operator browse/status commands print human-readable output by default. For agents and scripts, add `--json` for machine-readable output.

Search transcript text:

```bash
undertone --db ./undertone.db search "next steps"
undertone --db ./undertone.db search "next steps" --json
```

List persisted speaker fingerprints:

```bash
undertone --db ./undertone.db fingerprints
undertone --db ./undertone.db fingerprints --format json
undertone --db ./undertone.db fingerprints --unnamed --excerpts
undertone --db ./undertone.db fingerprints --status all
undertone --db ./undertone.db fingerprint-label VP-abc123 "Speaker Name"
undertone --db ./undertone.db relabel --all
undertone --db ./undertone.db fingerprint-adopt-model --dry-run
```

`relabel` (alias `resolve-names`) re-stamps saved transcript speaker names from the voice fingerprint DB without ASR, diarization, or enrichment. Use it after naming a voice to update old exports.

Inspect effective model and backend selections:

```bash
undertone --db ./undertone.db models
undertone --db ./undertone.db doctor
undertone --db ./undertone.db doctor --all
undertone --db ./undertone.db doctor --check-pyannote
```

Source commands are always visible. There is no per-source enable switch; a source becomes ready when its optional dependency, credentials, or local data exists. `doctor` shows optional source readiness, and source commands print fix-oriented messages when a dependency is missing.

Connectors can also be installed as Python entry-point plugins under the `undertone.connectors` group. A connector implements `matches(ref) -> bool` and `fetch(ref) -> ConnectorAsset`; Undertone discovers it at runtime:

```bash
undertone connector-list
undertone --db ./undertone.db connector-ingest 'https://www.youtube.com/watch?v=...'
undertone connector-resolve 'https://example.com/article-with-audio' --json
undertone --db ./undertone.db web-ingest 'https://example.com/article-with-audio' --list
```

First-party compatibility commands such as `youtube-ingest` and `podcast-ingest` remain available.
Third-party connectors are additive: built-in YouTube and podcast connectors stay discoverable, and connector name collisions fail loudly instead of shadowing a built-in.

Common maintenance commands:

```bash
undertone --db ./undertone.db reenrich meeting-1 --turn-gap-ms 600
undertone --db ./undertone.db webhook-preview meeting-1
undertone --db ./undertone.db fingerprint-label VP-abc123 "Speaker Name"
undertone --db ./undertone.db fingerprint-export --output ./voiceprints.json
undertone --db ./undertone.db fingerprint-adopt-model --dry-run
undertone --db ./undertone.db fingerprint-adopt-model --yes
undertone --db ./undertone.db fingerprint-merge VP-old VP-canonical --dry-run
undertone --db ./undertone.db fingerprint-merge VP-old VP-canonical --yes
undertone --db ./undertone.db fingerprint-discard VP-bad --reason "mixed speaker" --dry-run
undertone --db ./undertone.db fingerprint-discard VP-bad --reason "mixed speaker" --yes
undertone --db ./undertone.db fingerprint-restore VP-bad --dry-run
undertone --db ./undertone.db fingerprint-restore VP-bad --yes
undertone --db ./undertone.db fingerprint-destroy VP-bad --dry-run
undertone --db ./undertone.db fingerprint-destroy VP-bad --yes
undertone --db ./undertone.db stats
undertone --db ./undertone.db delete meeting-1 --yes
```

Fingerprint import, merge, model adoption, discard, restore, and destroy are explicit write operations. Use `--dry-run` first; writes require `--yes`. Before a mutating write, Undertone creates a timestamped `.bak` copy beside the SQLite DB.

Discard is the reversible corrective action for a bad voiceprint. A discarded print is hidden from normal `fingerprints` output, ignored by future matching, and visible with `fingerprints --status discarded` or `--status all`. Restore makes it matchable again. Destroy permanently deletes the fingerprint row and cascades its fingerprint-source rows; saved transcript speaker rows keep their historical fingerprint id.

Speaker fingerprints are namespaced by the embedding model that produced their vectors. Changing `UNDERTONE_EMBEDDING_MODEL` or switching to the pyannote backend does not compare new embeddings against old model spaces. Legacy fingerprints from older Undertone DBs are dormant until explicitly adopted:

```bash
undertone --db ./undertone.db doctor
undertone --db ./undertone.db fingerprint-adopt-model --dry-run
undertone --db ./undertone.db fingerprint-adopt-model --yes
```

`fingerprint-adopt-model` is a provenance assertion for existing vectors, not a cross-model conversion. Use it only when the stored vectors were actually produced by the target model. A true model migration requires rerunning audio to rebuild embeddings.

Ingest commands fail instead of silently overwriting an existing transcript id. Pass `--force` to overwrite or `--skip-existing` to no-op when the target id already exists.

Long-running ingest commands support JSON progress events on stderr:

```bash
undertone --db ./undertone.db run-wav ./meeting.wav --progress json --output ./meeting.json
```

Stdout and `--output` remain reserved for transcript output.

## Schemas

Undertone publishes JSON Schemas for the transcript contract, connector asset contract, and connector candidate contract:

```bash
undertone schema transcript --output ./undertone-transcript.schema.json
undertone schema connector-asset --output ./undertone-connector-asset.schema.json
undertone schema connector-candidate --output ./undertone-connector-candidate.schema.json
```

The transcript schema is versioned by `schema_version`. Connector plugins exchange `ConnectorAsset` shape version `1`; web media resolution exposes `ConnectorCandidate` shape version `1`.

## Engines

`fluidaudio-hybrid` is the default. It runs FluidAudio transcription, FluidAudio processing, and Sortformer-style diarization, then combines the outputs into Undertone's transcript schema. The default local stack is:

- ASR: FluidAudio Parakeet TDT
- diarization: FluidAudio Sortformer plus process output
- VAD: FluidAudio / Silero VAD
- speaker embeddings: FluidAudio pyannote-derived embeddings
- fingerprinting: undertone SQLite `speaker_fingerprints`
- acoustic metrics (optional): Parselmouth F0, jitter, shimmer, voiced duration, articulation rate

`fluidaudio-cli` is the simpler FluidAudio process path:

```bash
undertone --db ./undertone.db run-wav ./meeting.wav --engine fluidaudio-cli
```

`fluidaudio-pyannote` uses FluidAudio for word-timestamped ASR and runs `pyannote.audio` in-process for diarization spans and speaker embeddings. Install it only if you need that backend:

```bash
pip install 'undertone-audio[pyannote]'
undertone doctor --check-pyannote
undertone --db ./undertone.db run-wav ./meeting.wav --engine fluidaudio-pyannote
```

Pyannote model/device selection is configurable without local path hooks:

```bash
undertone run-wav ./meeting.wav \
  --engine fluidaudio-pyannote \
  --pyannote-model community-1 \
  --pyannote-device auto
```

Use `community-1`, `3.1`, or a full Hugging Face model ID. If the model is gated, accept the Hugging Face model terms and set `HF_TOKEN` or `HUGGINGFACE_TOKEN`. See [Diarization Backends](https://github.com/zm2231/undertone/blob/main/docs/diarization-backends.md) for backend details.

`fluidaudio-pyannote` runs FluidAudio ASR first and starts pyannote only after ASR succeeds, so a failed ASR run never leaves a diarization model loading in the background. There is no mid-run cancellation: a slow pyannote run completes before the command returns. See [Diarization Backends](https://github.com/zm2231/undertone/blob/main/docs/diarization-backends.md) for details.

To make pyannote the default engine for a shell/session:

```bash
export UNDERTONE_ENGINE=fluidaudio-pyannote
export UNDERTONE_PYANNOTE_MODEL=community-1
export UNDERTONE_PYANNOTE_DEVICE=auto
```

Unset `UNDERTONE_ENGINE` or set it back to `fluidaudio-hybrid` to return to the default FluidAudio hybrid path.

Override model labels per command:

```bash
undertone run-wav ./meeting.wav \
  --asr-model "FluidAudio Parakeet TDT" \
  --diarization-model "FluidAudio Sortformer + process" \
  --vad-model "FluidAudio/Silero VAD" \
  --embedding-model "FluidAudio pyannote-derived speaker embeddings" \
  --pyannote-model "pyannote/speaker-diarization-community-1" \
  --pyannote-device auto \
  --voice-metrics required
```

FluidAudio model flags are passed to the FluidAudio boundary. Pyannote flags configure the optional in-process pyannote backend. Unsupported combinations fail at audio processing time.

External binaries are bounded by `UNDERTONE_PROCESS_TIMEOUT_SECONDS` or `--process-timeout-seconds` on ingest commands. The default is 7200 seconds; set it to `0` only if you intentionally want no subprocess timeout.

## Source Rule

When audio is available, Undertone uses audio. Captions, feed notes, source text, and external speaker labels are provenance or fallback only. Population always goes through local ASR, diarization, embeddings, fingerprints, and enrichment.

This matters most for YouTube and podcasts. Caption pulls and plain podcast scripts produce searchable text but no reliable speaker attribution, so Undertone downloads the media and runs its own local pipeline.

## Sources

### YouTube

```bash
pip install -e '.[connectors,voice]'

UNDERTONE_WEBHOOK_ENABLED=0 undertone --db ./undertone.db youtube-ingest \
  'https://www.youtube.com/watch?v=jNQXAC9IVRw' \
  --download-dir ./downloads/youtube \
  --engine fluidaudio-hybrid \
  --voice-metrics optional \
  --output ./youtube.json
```

Flags: `--yt-dlp-bin` (non-default binary), `--audio-format wav`, `--include-playlist`, `--dry-run` (download the connector asset and print metadata without transcribing). `youtube-ingest` accepts only YouTube hosts; use `web-ingest` for article pages or arbitrary web URLs. Downloads are published only after a completed transfer; interrupted downloads do not leave reusable media files behind.

### Web media

Use web media ingest for content/article pages where yt-dlp must resolve the actual audio source first, such as a newsletter post that embeds an audio player and links to YouTube.

```bash
undertone connector-resolve 'https://example.com/article-with-audio'
undertone connector-resolve 'https://example.com/article-with-audio' --json
undertone --db ./undertone.db web-ingest 'https://example.com/article-with-audio' --list
undertone --db ./undertone.db web-ingest 'https://example.com/article-with-audio' --select <candidate-id>
undertone --db ./undertone.db web-ingest 'https://example.com/article-with-audio' --yes
```

`connector-resolve` and `web-ingest --list` are metadata-only previews and do not download media. They print ranked candidates with stable `candidate_id`, source kind, availability, extractor, title, URL, and duration. Ranking prefers non-voiceover real media, then the longest duration. If more than one downloadable candidate is found, `web-ingest` requires `--select <candidate-id>`; `--yes` only skips confirmation for a single downloadable candidate.

Only candidates with a concrete media URL or extractor URL are marked downloadable. If yt-dlp reports multiple article-page entries that all point back to the same article URL, Undertone lists them as not directly downloadable instead of pretending `--select` can pin a recording.

For arbitrary web URLs, Undertone preflights localhost/private-network user URLs and selected candidate media URLs before download, applies the process timeout, and maps the Undertone `--max-download-size` option (`UNDERTONE_MAX_DOWNLOAD_SIZE` or `2G` by default) to yt-dlp `--max-filesize`. Auth/cookies are explicit only via `--cookies` or `--cookies-from-browser`; Undertone invokes yt-dlp with `--ignore-config` so ambient yt-dlp config is not loaded. yt-dlp performs its own DNS, redirects, extractor logic, and network I/O after Undertone's preflight checks. Treat `web-ingest` as a local trusted-URL tool; do not expose it as a hosted/server-side fetch endpoint without an external egress sandbox or network policy.

### Podcasts

```bash
undertone podcast-list 'https://example.com/feed.xml' --limit 20
undertone --db ./undertone.db podcast-ingest 'https://example.com/feed.xml' --episode 0
undertone --db ./undertone.db podcast-ingest 'https://example.com/feed.xml' --title-contains 'interview'
undertone --db ./undertone.db podcast-ingest 'https://cdn.example.com/episode.mp3'
```

Episodes are selected by zero-based `--episode` index or first `--title-contains` match. Direct media URLs skip RSS parsing. Downloads are published atomically, so a dropped stream does not poison the cache for the next run.

### Quill

```bash
undertone quill-list --limit 20
undertone --db ./undertone.db quill-ingest <quill-meeting-id> --engine fluidaudio-hybrid
undertone --db ./undertone.db quill-ingest --limit 10 --dry-run
```

Precedence: `combined.m4a` when present, otherwise a mix of `mic.m4a` and `system.m4a`. Quill ASR and `SPK-*` labels are ignored when audio exists. Override locations with `--quill-db` and `--meetings-dir`.

### Google Meet

```bash
pip install -e '.[meet]'

gcloud auth application-default login \
  --scopes="https://www.googleapis.com/auth/meetings.space.readonly,https://www.googleapis.com/auth/drive.meet.readonly"

undertone --db ./undertone.db meet-discover --google-account you@example.com
undertone --db ./undertone.db meet-ingest conferenceRecords/... --audio ./recording.mp4
undertone --db ./undertone.db meet-ingest conferenceRecords/... --adc-file ./google-adc.json
```

Prerequisites: install the `.[meet]` extra, install the Google Cloud CLI, enable the Google Meet API for the credential's project, and create ADC with the scopes above. `meetings.space.readonly` is used for conference records, transcript lists, transcript entries, participants, and recording metadata. `drive.meet.readonly` is used only when Undertone downloads a Meet recording file. The authenticated account must have access to the conference/artifact; it does not have to be the organizer for every artifact, but it cannot read arbitrary meetings.

Precedence: explicit `--audio`, then a downloadable Meet Drive recording, then Meet API text. Text fallback is marked `diarization_state=text-fallback` and produces no voice fingerprints. Use `--no-text-fallback` to fail instead of persisting text-only output. For multiple Google accounts, `--adc-file` selects credentials explicitly. Use `--no-probe` on `meet-discover` to skip per-record recording/transcript probes after listing conference records; Meet discovery still requires Google auth.

## Configurable Paths

No command needs a machine-specific absolute path. Defaults are portable:

- database: `UNDERTONE_DB_PATH` or `--db` (default `./undertone.db`)
- connector downloads: `UNDERTONE_DOWNLOAD_DIR`, `XDG_CACHE_HOME/undertone/downloads`, or `~/.cache/undertone/downloads`; `--download-dir` is available on first-party source commands such as `youtube-ingest` and `podcast-ingest`
- FluidAudio binary: `UNDERTONE_FLUIDAUDIO_CLI`, `FLUIDAUDIO_CLI`, or `fluidaudiocli` on `PATH`
- external process timeout: `UNDERTONE_PROCESS_TIMEOUT_SECONDS` or `--process-timeout-seconds` on ingest commands (default `7200`; `0` disables)
- pyannote backend selection: `UNDERTONE_PYANNOTE_MODEL` and `UNDERTONE_PYANNOTE_DEVICE`

## Output Formats

Every ingest and load command can choose a format and a detail level:

```bash
undertone --db ./undertone.db load meeting-1 --output-format md --output-detail minimal --output meeting.md
undertone --db ./undertone.db load meeting-1 --output-format jsonl --output-detail full
undertone --db ./undertone.db run-wav ./meeting.wav --output-format text --output-detail standard
```

Formats:

- `json`: enriched transcript
- `raw-json`: pre-enrichment raw transcript
- `jsonl`: one segment per line
- `csv`: speaker metrics table
- `text`: readable speaker summary and transcript
- `md`: Markdown speaker summary and transcript

Detail levels:

- `minimal`: transcript text, timing, and speaker basics
- `standard`: adds enrichment and non-acoustic speaker metrics
- `full`: adds per-word timings and acoustic metrics

## Python API

Run the pipeline directly:

```python
import asyncio
from undertone_audio import AudioPipeline
from undertone_audio.engines import create_engine
from undertone_audio.storage import TranscriptStore

store = TranscriptStore("undertone.db")
pipeline = AudioPipeline(store=store, engine=create_engine())  # defaults to fluidaudio-hybrid
transcript = asyncio.run(pipeline.run("./meeting.wav", transcript_id="meeting-1"))
```

Save a raw transcript built elsewhere:

```python
from undertone_audio import AudioPipeline, Segment, Speaker
from undertone_audio.engines.base import RawTranscript
from undertone_audio.storage import TranscriptStore

store = TranscriptStore("undertone.db")
pipeline = AudioPipeline(store=store)

pipeline.finalize_raw(
    RawTranscript(
        duration_ms=1000,
        language="en",
        engine="example",
        speakers=[Speaker(speaker_id="S1")],
        segments=[Segment(segment_id="seg1", speaker_id="S1", start_ms=0, end_ms=1000, text="hello")],
    ),
    transcript_id="meeting-1",
)

transcript = store.load("meeting-1")
```

The same raw shape can be saved through the CLI:

```bash
undertone --db ./undertone.db finalize-json raw-transcript.json \
  --transcript-id meeting-1 \
  --diarization-state ok
```

### Plugging In A Diarization Backend

The backend boundary is small. An engine implements the `TranscriptionEngine` protocol from `undertone_audio.engines.base`:

```python
from pathlib import Path
from undertone_audio.engines.base import RawTranscript


class MyEngine:
    name = "my-engine"

    async def healthcheck(self) -> bool:
        return True

    async def transcribe(self, audio_path: Path) -> RawTranscript:
        ...
```

`transcribe()` returns a `RawTranscript`. For diarized output, populate:

- `speakers`: stable source speaker IDs, optional display names, optional embeddings
- `segments`: speaker-attributed text with `start_ms`, `end_ms`, and optional word timings
- `engine`: a backend name that makes the source clear in persisted metadata

If the backend can produce speaker embeddings, set them on `Speaker.embedding`; Undertone assigns and persists cross-recording `fingerprint_id` values. If the backend only produces ASR text, use a single speaker and set a degraded `diarization_state` when finalizing, so downstream consumers do not mistake ASR-only output for speaker-attributed output.

Pass a custom engine to the pipeline directly:

```python
pipeline = AudioPipeline(store=store, engine=MyEngine())
```

To make a backend selectable from `undertone run-wav --engine ...`, add it to `undertone_audio.engines.create_engine()` and add the engine name to the shared pipeline argument choices in `src/undertone_audio/commands/common.py`. `src/undertone_audio/cli.py` only wires command modules.

## Webhook

```bash
export UNDERTONE_WEBHOOK_URL=https://example.com/webhooks/meeting-ready
export UNDERTONE_WEBHOOK_SECRET=shared-secret
export UNDERTONE_WEBHOOK_ENABLED=1
```

When enabled, a ready transcript emits:

```json
{
  "event": "meeting.transcript.ready",
  "transcript_id": "meeting-1",
  "source": "undertone",
  "recorded_at": null,
  "store_ref": "sqlite:/abs/path/undertone.db#meeting-1"
}
```

The signature header is `x-zen-signature-256`, a SHA-256 HMAC over the payload body. Re-emit readiness for a saved transcript with `undertone emit-ready <transcript-id>`.

## Configuration

```bash
UNDERTONE_DB_PATH=./undertone.db
UNDERTONE_ENGINE=fluidaudio-hybrid
UNDERTONE_FLUIDAUDIO_CLI=/path/to/fluidaudiocli
UNDERTONE_PROCESS_TIMEOUT_SECONDS=7200
UNDERTONE_DOWNLOAD_DIR=./downloads
UNDERTONE_VOICE_METRICS=optional
UNDERTONE_OUTPUT_FORMAT=json
UNDERTONE_OUTPUT_DETAIL=full
UNDERTONE_WEBHOOK_ENABLED=0
```

Models and thresholds:

```bash
UNDERTONE_ASR_MODEL="FluidAudio Parakeet TDT"
UNDERTONE_DIARIZATION_MODEL="FluidAudio Sortformer + process"
UNDERTONE_VAD_MODEL="FluidAudio/Silero VAD"
UNDERTONE_EMBEDDING_MODEL="FluidAudio pyannote-derived speaker embeddings"
UNDERTONE_PYANNOTE_MODEL=pyannote/speaker-diarization-community-1
UNDERTONE_PYANNOTE_DEVICE=auto
UNDERTONE_FINGERPRINT_BACKEND=undertone-speaker-fingerprints
UNDERTONE_CLUSTERING_THRESHOLD=0.7045655
UNDERTONE_SPEAKER_MERGE_THRESHOLD=0.82
UNDERTONE_MIN_TALK_SECONDS=1.5
UNDERTONE_FINGERPRINT_SIMILARITY_THRESHOLD=0.78
UNDERTONE_TURN_GAP_MS=800
```

Feature toggles:

```bash
UNDERTONE_ENABLE_TURN_TAKING=1
UNDERTONE_ENABLE_FILLERS=1
UNDERTONE_ENABLE_LINGUISTIC=1
UNDERTONE_ENABLE_MEETING_TYPE=1
```

## Validation

```bash
pip install -e '.[dev]'
pytest -q tests
python -m compileall -q src tests
```

End-to-end smoke test against a real video:

```bash
RUN_DIR="$(mktemp -d)"
UNDERTONE_WEBHOOK_ENABLED=0 undertone --db "$RUN_DIR/undertone.db" youtube-ingest \
  'https://www.youtube.com/watch?v=Aq5WXmQQooo' \
  --download-dir "$RUN_DIR/downloads" \
  --engine fluidaudio-hybrid \
  --voice-metrics optional \
  --output "$RUN_DIR/transcript.json"

undertone --db "$RUN_DIR/undertone.db" load youtube-Aq5WXmQQooo \
  --output-format text --output-detail minimal
```

## Operator Skills

Undertone ships one Claude and Codex skill, `undertone`. It is a router: the skill body dispatches to a focused reference for each surface (ingest, connectors, meetings, exports, fingerprints, ops, and upgrades), and only the reference you need loads. One installed skill, one trigger, full depth on demand. It lives under `skills/undertone/` in the repo and in the wheel.

Install it as a Claude plugin so it updates with the marketplace:

```bash
/plugin marketplace add zm2231/undertone
/plugin install undertone
```

Or copy it from a pip install into your Claude or Codex skill directories:

```bash
pip install undertone-audio
undertone install-skills --target claude-user
undertone install-skills --target codex
undertone install-skills --target claude-project
```

`--target` is repeatable and defaults to `claude-user`. The copy is a snapshot, so re-run `install-skills` after upgrading undertone to refresh it, or use the plugin path to keep it current automatically.

## License

Apache-2.0. See [LICENSE](LICENSE).
