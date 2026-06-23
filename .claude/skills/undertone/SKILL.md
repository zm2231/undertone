---
name: undertone
description: Entry point for undertone, a local audio transcript producer. Use when ingesting audio from local files, YouTube, podcasts, Quill, or Google Meet, producing or re-enriching transcripts, browsing, searching, or exporting them, labeling speaker fingerprints, or running doctor, stats, or webhook checks. Routes to the detailed skills under skills/.
---

# Undertone

Use the repo skills under `skills/` for detailed workflows:

- `skills/undertone-ingest/SKILL.md`: direct local audio, raw transcript JSON, model flags.
- `skills/undertone-meetings-ingest/SKILL.md`: Quill and Google Meet meeting recordings, source precedence, text fallback.
- `skills/undertone-connectors/SKILL.md`: YouTube, podcasts, RSS feeds, direct media URLs, connector install, portable download paths.
- `skills/undertone-exports/SKILL.md`: load/search/export formats and detail levels.
- `skills/undertone-ops/SKILL.md`: install, tests, model inspection, package-boundary checks.

## Non-Negotiables

- `undertone` must stay self-contained; do not import a host application's internal packages.
- When audio exists, rerun Undertone local ASR, diarization, embeddings, fingerprinting, and enrichment.
- Quill and Google Meet text/diarization are fallback or provenance only.

## Common Commands

```bash
undertone --db ./undertone.db run-wav ./meeting.wav --engine fluidaudio-hybrid --voice-metrics optional
undertone quill-list --limit 20
undertone --db ./undertone.db quill-ingest <quill-meeting-id> --engine fluidaudio-hybrid
undertone --db ./undertone.db meet-discover --google-account you@example.com
undertone --db ./undertone.db meet-ingest conferenceRecords/... --audio ./recording.mp4
undertone --db ./undertone.db youtube-ingest 'https://www.youtube.com/watch?v=...' --engine fluidaudio-hybrid
undertone podcast-list 'https://example.com/feed.xml' --limit 20
undertone --db ./undertone.db podcast-ingest 'https://example.com/feed.xml' --episode 0
undertone --db ./undertone.db load meeting-1 --output-format md --output-detail standard
undertone --db ./undertone.db list --limit 20
undertone --db ./undertone.db reenrich meeting-1 --turn-gap-ms 600
undertone --db ./undertone.db webhook-preview meeting-1
undertone --db ./undertone.db stats
undertone --db ./undertone.db doctor
undertone --db ./undertone.db doctor --all
undertone --db ./undertone.db sources
undertone --db ./undertone.db models
undertone --db ./undertone.db fingerprints
undertone --db ./undertone.db fingerprints --unnamed --excerpts
undertone --db ./undertone.db fingerprint-label VP-abc123 "Speaker Name"
undertone --db ./undertone.db delete meeting-1 --yes
```

Browse/status commands are human-readable by default. For agents and scripts, add `--json` for machine-readable output. Source commands are always visible; readiness is dependency/auth/local-data based and visible in `doctor`. Transcript exports use `--output-format md|text|json|jsonl|csv|raw-json` and `--output-detail minimal|standard|full`.

Ingest commands fail on a duplicate transcript id rather than overwriting; pass `--force` to overwrite or `--skip-existing` to no-op.

## Verification

```bash
PYTHONPATH=src pytest -q tests
python -m compileall -q src tests
```
