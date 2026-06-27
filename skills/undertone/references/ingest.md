# Undertone Ingest

Use this skill for producing raw/enriched transcripts from local WAV/MP4/M4A files or raw transcript JSON. For Quill and Google Meet, see `references/meetings.md`. For YouTube, podcasts, RSS, and direct media URLs, see `references/connectors.md`. For speaker fingerprint operations, see `references/fingerprints.md`.

## Core Rules

- `undertone` is the producer for raw audio transcripts. Keep it self-contained; do not import a host application's internal packages.
- When audio exists, always rerun local Undertone ASR, diarization, embeddings, fingerprinting, and audio-derived enrichment.
- Prefer `fluidaudio-hybrid` for quality because it combines FluidAudio ASR/process output with Sortformer overlap-aware spans.
- Use `fluidaudio-pyannote` when Sortformer under-splits speakers and the optional `.[pyannote]` extra is installed.
- Use `fluidaudio-cli` only for faster scans or when Sortformer is unavailable.

## Direct Audio

```bash
undertone --db ./undertone.db run-wav ./meeting.wav \
  --engine fluidaudio-hybrid \
  --voice-metrics optional \
  --progress json \
  --output-format json \
  --output-detail full \
  --output ./meeting.json
```

Use `--voice-metrics required` when acoustic metrics such as F0, jitter, shimmer, voiced duration, and articulation rate must be present. Use `--voice-metrics off` for exports that do not need acoustic metrics.

## Raw Transcript JSON

```bash
undertone --db ./undertone.db finalize-json raw-transcript.json \
  --transcript-id meeting-1 \
  --diarization-state ok
```

## Duplicate IDs

`run-wav` and `finalize-json` fail with a nonzero exit instead of silently overwriting an existing transcript id. Pass `--force` to overwrite the existing transcript, or `--skip-existing` to no-op when the target id already exists. The same guard applies to every ingest path, including connectors and meeting sources.

## Progress Events

Use `--progress json` for queue integrations. Progress events are JSONL on stderr; transcript JSON/stdout stays clean.

## Model Flags

```bash
undertone run-wav ./meeting.wav \
  --asr-model "FluidAudio Parakeet TDT" \
  --diarization-model "FluidAudio Sortformer + process" \
  --vad-model "FluidAudio/Silero VAD" \
  --embedding-model "FluidAudio pyannote-derived speaker embeddings" \
  --pyannote-model community-1 \
  --pyannote-device auto \
  --fingerprint-backend undertone-speaker-fingerprints
```

FluidAudio model selections are passed to the FluidAudio boundary. Pyannote selections configure the optional in-process `fluidaudio-pyannote` backend. Unsupported combinations should fail during audio processing rather than being recorded as metadata-only choices.

Speaker fingerprints are namespaced by the effective embedding model. For `fluidaudio-pyannote`, the fingerprint model is the resolved pyannote model; otherwise it is `UNDERTONE_EMBEDDING_MODEL` / `--embedding-model`. Legacy fingerprints with no model tag are dormant and should trigger `doctor`/ingest warnings. Use `fingerprint-adopt-model --dry-run` only when asserting that old vectors were produced by the active model; it does not convert vectors across embedding spaces.

## External Process Bounds

FluidAudio and ffmpeg/ffprobe subprocesses are bounded. Use `--process-timeout-seconds` or `UNDERTONE_PROCESS_TIMEOUT_SECONDS` to adjust the limit for long media. The default is `7200`; set `0` only when intentionally disabling subprocess timeouts.

## Speaker Fingerprint Gates

Fingerprinting is duration-gated on every engine, not just pyannote. A speaker with less than the enroll threshold of total talk time does not mint a durable cross-recording fingerprint, and a sample below the update threshold is not folded into a stored centroid. This is deliberate: short, noisy speakers are the main source of garbage identities. A brief speaker still appears in the transcript with per-recording diarization; it just does not get a stable durable identity.

## Confidence Fields

- Word ASR confidence is preserved as `words[].confidence`.
- Segment ASR confidence is derived as `segments[].asr_confidence`.
- `segments[].diarization_quality` is nullable.
- `fluidaudio-cli` preserves FluidAudio process `qualityScore`.
- `fluidaudio-hybrid` overlap-maps process `qualityScore` onto Sortformer spans when possible; partial coverage is expected.
- `fluidaudio-pyannote` emits `null` for diarization quality because pyannote's public `DiarizeOutput` does not expose per-span confidence or posteriors.

## Pyannote Backend

The pyannote backend is optional. Do not assume it exists in a base install.

```bash
pip install -e '.[pyannote]'
undertone doctor --check-pyannote
undertone --db ./undertone.db run-wav ./meeting.wav --engine fluidaudio-pyannote
```

`doctor --check-pyannote` checks dependency import only. It does not download or load a gated Hugging Face model; model access is verified when the backend runs.

Use `UNDERTONE_PYANNOTE_MODEL` / `--pyannote-model` and `UNDERTONE_PYANNOTE_DEVICE` / `--pyannote-device` for model and device selection. If the selected Hugging Face model is gated, accept its terms and set `HF_TOKEN` or `HUGGINGFACE_TOKEN`.

This backend is sequential: FluidAudio ASR runs first, and pyannote starts only after ASR succeeds, so a failed ASR run never leaves pyannote loading in the background. There is no mid-run cancellation; a slow pyannote run completes before the command returns.

To flip pyannote on by default for a shell/session:

```bash
export UNDERTONE_ENGINE=fluidaudio-pyannote
export UNDERTONE_PYANNOTE_MODEL=community-1
export UNDERTONE_PYANNOTE_DEVICE=auto
```

To return to the default path:

```bash
unset UNDERTONE_ENGINE
# or
export UNDERTONE_ENGINE=fluidaudio-hybrid
```

## Benchmark Boundary

Raw local benchmark scripts and results are tuning artifacts, not public docs. Do not promote private meeting/audio benchmark outputs into skills, README, or release notes. If publishing benchmark claims, use reproducible public samples and record engine, model, device, FluidAudio build, pyannote version, expected-speaker source, and scoring criteria.
