<div align="center">

# 🎙️ Undertone

**On-device audio transcripts that capture the words and how they were said.**

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
- `yt-dlp`, only for the YouTube connector
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

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e '.[voice]'                # Parselmouth acoustic metrics
pip install -e '.[meet]'                 # Google Meet auth helpers
pip install -e '.[connectors]'           # YouTube connector via yt-dlp
pip install -e '.[dev]'                  # pytest
pip install -e '.[dev,voice,meet,connectors]'
```

### 3. Verify

```bash
command -v fluidaudiocli
undertone --help
undertone models
```

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
undertone --db ./undertone.db fingerprints --unnamed --excerpts
undertone --db ./undertone.db fingerprint-label VP-abc123 "Speaker Name"
```

Inspect effective model and backend selections:

```bash
undertone --db ./undertone.db models
undertone --db ./undertone.db doctor
undertone --db ./undertone.db doctor --all
```

Source commands are always visible. There is no per-source enable switch; a source becomes ready when its optional dependency, credentials, or local data exists. `doctor` shows optional source readiness, and source commands print fix-oriented messages when a dependency is missing.

Common maintenance commands:

```bash
undertone --db ./undertone.db reenrich meeting-1 --turn-gap-ms 600
undertone --db ./undertone.db webhook-preview meeting-1
undertone --db ./undertone.db fingerprint-label VP-abc123 "Speaker Name"
undertone --db ./undertone.db stats
undertone --db ./undertone.db delete meeting-1 --yes
```

Ingest commands fail instead of silently overwriting an existing transcript id. Pass `--force` to overwrite or `--skip-existing` to no-op when the target id already exists.

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

Override model labels per command:

```bash
undertone run-wav ./meeting.wav \
  --asr-model "FluidAudio Parakeet TDT" \
  --diarization-model "FluidAudio Sortformer + process" \
  --vad-model "FluidAudio/Silero VAD" \
  --embedding-model "FluidAudio pyannote-derived speaker embeddings" \
  --voice-metrics required
```

Non-default model flags are passed to the FluidAudio boundary. Unsupported combinations fail at audio processing time.

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

Flags: `--yt-dlp-bin` (non-default binary), `--audio-format wav`, `--include-playlist`, `--dry-run` (select media and print metadata without ingesting).

### Podcasts

```bash
undertone podcast-list 'https://example.com/feed.xml' --limit 20
undertone --db ./undertone.db podcast-ingest 'https://example.com/feed.xml' --episode 0
undertone --db ./undertone.db podcast-ingest 'https://example.com/feed.xml' --title-contains 'interview'
undertone --db ./undertone.db podcast-ingest 'https://cdn.example.com/episode.mp3'
```

Episodes are selected by zero-based `--episode` index or first `--title-contains` match. Direct media URLs skip RSS parsing.

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
- connector downloads: `UNDERTONE_DOWNLOAD_DIR`, `--download-dir`, `XDG_CACHE_HOME/undertone/downloads`, or `~/.cache/undertone/downloads`
- FluidAudio binary: `UNDERTONE_FLUIDAUDIO_CLI`, `FLUIDAUDIO_CLI`, or `fluidaudiocli` on `PATH`

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

Agent and operator workflows live under `skills/`:

- `undertone-ingest`: local audio, raw transcript JSON, model flags
- `undertone-meetings-ingest`: Quill and Google Meet recordings, source precedence, text fallback
- `undertone-connectors`: YouTube, podcasts, RSS feeds, direct media URLs, connector paths
- `undertone-exports`: output formats, detail levels, search, load, webhook re-emission
- `undertone-ops`: install, tests, package checks, models, fingerprints

A Claude-compatible dispatcher lives at `.claude/skills/undertone/SKILL.md`.

## License

Apache-2.0. See [LICENSE](LICENSE).
