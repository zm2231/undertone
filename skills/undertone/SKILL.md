---
name: undertone
description: Operate undertone, a local on-device audio transcript producer. Use when ingesting audio from local files, YouTube, podcasts, RSS feeds, direct media URLs, article/web media pages, Quill, or Google Meet; running transcription, diarization, or FluidAudio/pyannote engine selection; producing, re-enriching, browsing, searching, or exporting transcripts; labeling, relabeling, merging, discarding, restoring, destroying, exporting, or adopting speaker voice fingerprints; running doctor, stats, models, or webhook checks; installing, packaging, or upgrading undertone. Routes to detailed references under references/.
---

# Undertone

Undertone is a self-contained, on-device producer of audio transcripts: it downloads or reads audio, runs local ASR, diarization, speaker embeddings, fingerprinting, and enrichment, and persists transcripts you can browse, search, and export. This skill is the router. Read the one reference that matches the task before acting.

## Which reference

| Task | Read |
|------|------|
| Local WAV/MP4/M4A or raw transcript JSON; ASR/diarization; engine and model flags; confidence fields | `references/ingest.md` |
| YouTube, podcasts, RSS feeds, direct media URLs, arbitrary article/web media pages; connector plugins and download paths | `references/connectors.md` |
| Quill or Google Meet meeting recordings; source precedence; multi-account auth; text fallback | `references/meetings.md` |
| Load, search, export formats (json/raw-json/text/md/jsonl/csv) and detail levels | `references/exports.md` |
| Speaker fingerprints: label, relabel, discard/restore/destroy, merge, export/import, embedding-model adoption, over-segmentation | `references/fingerprints.md` |
| Install, doctor, models, stats, health, packaging, self-contained boundary checks | `references/ops.md` |
| Version upgrades, automatic schema migration, fingerprint model namespace, cross-DB library portability | `references/upgrades.md` |

## Non-Negotiables

- `undertone` must stay self-contained; do not import a host application's internal packages.
- When audio exists, rerun Undertone local ASR, diarization, embeddings, fingerprinting, and enrichment. Source-provided transcript text, speaker IDs, and diarization are fallback or provenance only.
- Identity-changing fingerprint operations (import, merge, adopt-model, discard, restore, destroy) require `--dry-run` first, then `--yes` with a timestamped `.bak` when a write occurs.

## Common Commands

```bash
undertone --db ./undertone.db run-wav ./meeting.wav --engine fluidaudio-hybrid --voice-metrics optional
undertone --db ./undertone.db quill-ingest <quill-meeting-id> --engine fluidaudio-hybrid
undertone --db ./undertone.db meet-ingest conferenceRecords/... --audio ./recording.mp4
undertone --db ./undertone.db youtube-ingest 'https://www.youtube.com/watch?v=...' --engine fluidaudio-hybrid
undertone connector-resolve 'https://example.com/article-with-audio'
undertone --db ./undertone.db web-ingest 'https://example.com/article-with-audio' --select <candidate-id>
undertone --db ./undertone.db podcast-ingest 'https://example.com/feed.xml' --episode 0
undertone --db ./undertone.db load meeting-1 --output-format md --output-detail standard
undertone --db ./undertone.db reenrich meeting-1 --turn-gap-ms 600
undertone --db ./undertone.db relabel --all
undertone --db ./undertone.db fingerprints --unnamed --excerpts
undertone --db ./undertone.db stats
undertone --db ./undertone.db doctor --all
```

Browse and status commands are human-readable by default; add `--json` for agents and scripts. Transcript exports use `--output-format md|text|json|jsonl|csv|raw-json` and `--output-detail minimal|standard|full`. Ingest commands fail on a duplicate transcript id rather than overwriting; pass `--force` to overwrite or `--skip-existing` to no-op. Matching audio across different source ids is skipped before fingerprint assignment when Chromaprint `fpcalc` is installed; text simhash is advisory metadata, not a silent skip gate.

## Verification

```bash
PYTHONPATH=src pytest -q tests
python -m compileall -q src tests
```
