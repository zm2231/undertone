---
name: undertone-ingest
description: Use when ingesting local audio files or raw transcript JSON into Undertone, running transcription or diarization, selecting FluidAudio backends/models, and producing raw/enriched transcripts without source-specific connector behavior.
---

# Undertone Ingest

Use this skill for producing raw/enriched transcripts from local WAV/MP4/M4A files or raw transcript JSON. For Quill and Google Meet, use `skills/undertone-meetings-ingest/SKILL.md`. For YouTube, podcasts, RSS, and direct media URLs, use `skills/undertone-connectors/SKILL.md`.

## Core Rules

- `undertone` is the producer for raw audio transcripts. Keep it self-contained; do not import a host application's internal packages.
- When audio exists, always rerun local Undertone ASR, diarization, embeddings, fingerprinting, and audio-derived enrichment.
- Prefer `fluidaudio-hybrid` for quality because it combines FluidAudio ASR/process output with Sortformer overlap-aware spans.
- Use `fluidaudio-cli` only for faster scans or when Sortformer is unavailable.

## Direct Audio

```bash
undertone --db ./undertone.db run-wav ./meeting.wav \
  --engine fluidaudio-hybrid \
  --voice-metrics optional \
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

## Model Flags

```bash
undertone run-wav ./meeting.wav \
  --asr-model "FluidAudio Parakeet TDT" \
  --diarization-model "FluidAudio Sortformer + process" \
  --vad-model "FluidAudio/Silero VAD" \
  --embedding-model "FluidAudio pyannote-derived speaker embeddings" \
  --fingerprint-backend undertone-speaker-fingerprints
```

Non-default model selections are passed to the FluidAudio boundary. Unsupported combinations should fail during audio processing rather than being recorded as metadata-only choices.
