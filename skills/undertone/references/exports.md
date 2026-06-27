# Undertone Exports

Use this skill for read paths and artifact generation from persisted Undertone transcripts.

## Load And Search

```bash
undertone --db ./undertone.db load meeting-1
undertone --db ./undertone.db list --limit 20
undertone --db ./undertone.db search "next steps" --limit 20
undertone schema transcript
undertone --db ./undertone.db emit-ready meeting-1
undertone --db ./undertone.db webhook-preview meeting-1
```

Browse/status commands are human-readable by default. For agents and scripts, add `--json` for machine-readable output. Transcript material uses `--output-format` and `--output-detail`.

Use `emit-ready` only after verifying the transcript is present and downstream webhook config is intentional.

## Formats

- `json`: full `EnrichedTranscript`.
- `raw-json`: persisted pre-enrichment `RawTranscript` shape.
- `text`: readable speaker summary and transcript.
- `md`: markdown speaker summary and transcript.
- `jsonl`: one segment per line with enrichment payload.
- `csv`: one speaker-metric row per speaker.

Examples:

```bash
undertone --db ./undertone.db load meeting-1 --output-format md --output-detail minimal --output meeting.md
undertone --db ./undertone.db load meeting-1 --output-format jsonl --output-detail full --output meeting.jsonl
undertone --db ./undertone.db load meeting-1 --output-format csv --output speaker-metrics.csv
undertone --db ./undertone.db run-wav ./meeting.wav --output-format text --output-detail standard
```

Use `reenrich` to rebuild enrichment from the stored raw transcript after changing thresholds or feature toggles:

```bash
undertone --db ./undertone.db reenrich meeting-1 --turn-gap-ms 600 --output-format json
```

## Detail Levels

- `minimal`: transcript text, timing, and speaker basics only.
- `standard`: transcript text/timing, enrichment, and non-acoustic speaker metrics; omits per-word timings and acoustic metrics.
- `full`: complete transcript, per-word timings, enrichment, and acoustic metrics such as F0, jitter, shimmer, voiced duration, and articulation rate.

Use `minimal` for human review or downstream systems that only need transcript text. Use `standard` for most CRM/workflow handoffs. Use `full` for analytics, QA, voice fingerprint inspection, or acoustic research.

## Confidence Fields

`words[].confidence` is FluidAudio ASR confidence. `segments[].asr_confidence` is the segment-level aggregate. `segments[].diarization_quality` is nullable and backend-specific: direct FluidAudio process quality on `fluidaudio-cli`, overlap-mapped FluidAudio process quality on `fluidaudio-hybrid`, and `null` on `fluidaudio-pyannote`.

## Source Metadata Boundary

Exports should remain Undertone audio/source context only. Keep connector/source metadata separate from downstream app enrichment.
