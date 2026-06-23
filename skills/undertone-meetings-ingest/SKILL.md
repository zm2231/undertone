---
name: undertone-meetings-ingest
description: Use when ingesting meeting recordings into Undertone from Quill or Google Meet, including source precedence, multi-account Google auth, local recording overrides, and text fallback behavior when no audio exists.
---

# Undertone Meetings Ingest

Use this skill for meeting-source ingest. Quill and Google Meet can provide useful provenance, but audio wins whenever it exists.

## Core Rule

When a meeting has audio, run Undertone local ASR, diarization, embeddings, fingerprinting, and enrichment. Do not trust source-provided transcript text, speaker IDs, or diarization as the population path.

Meeting ingest fails on a duplicate transcript id rather than overwriting. Pass `--force` to overwrite or `--skip-existing` to no-op.

## Quill

```bash
undertone quill-list --limit 20
undertone --db ./undertone.db quill-ingest <quill-meeting-id> --engine fluidaudio-hybrid
undertone --db ./undertone.db quill-ingest --limit 10 --dry-run
```

`quill-list` and Quill dry-run reports are human-readable by default. For agents and scripts, add `--json` for machine-readable output.

Quill precedence:

- Use `combined.m4a` when present.
- Otherwise mix/use local `mic.m4a` and `system.m4a`.
- Ignore Quill ASR, `SPK-*` IDs, and Quill diarization when audio exists.

Quill defaults to `~/Library/Application Support/Quill/quill.db` and `.../meetings`. When Quill lives somewhere else, pass `--quill-db` and `--meetings-dir` on every Quill command (there is no env var for these paths):

```bash
undertone quill-list \
  --quill-db /your/path/quill.db \
  --meetings-dir /your/path/meetings

undertone --db ./undertone.db quill-ingest <quill-meeting-id> \
  --quill-db /your/path/quill.db \
  --meetings-dir /your/path/meetings
```

A missing or wrong path is reported as not-ready (exit 0) with guidance, not a crash; run `undertone doctor` to check readiness.

## Google Meet

```bash
pip install -e '.[meet]'
gcloud auth application-default login \
  --scopes="https://www.googleapis.com/auth/meetings.space.readonly,https://www.googleapis.com/auth/drive.meet.readonly"

undertone --db ./undertone.db meet-discover --google-account you@example.com
undertone --db ./undertone.db meet-ingest conferenceRecords/... --audio ./recording.mp4
undertone --db ./undertone.db meet-ingest conferenceRecords/... --adc-file ./google-adc.json
```

`meet-discover` is human-readable by default. For agents and scripts, add `--json` for machine-readable output. `--no-probe` skips recording/transcript probes after listing conference records, but discovery still requires Google auth.

Google Cloud CLI is required for the ADC setup command. The Meet scope reads conference records, transcript lists, transcript entries, participant display names, and recording metadata. The Drive Meet scope is only needed when downloading recording files. If auth fails or scopes are missing, rerun the scoped `gcloud auth application-default login` command above, or pass an explicit `--adc-file`.

Meet precedence:

1. Explicit local `--audio` recording.
2. Downloadable Meet Drive recording.
3. Meet API text fallback only when no audio exists.

Use `--no-text-fallback` when the operation must fail instead of persisting a text-only transcript. Text fallback is marked `diarization_state=text-fallback` and cannot produce voice fingerprints.

For multiple Google accounts, prefer `--adc-file` because it selects credentials explicitly. `--google-account` is source metadata plus a lookup hint for conventional credential files before active ADC fallback.
