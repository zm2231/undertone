---
name: undertone-ops
description: Use when verifying, debugging, installing, packaging, or inspecting Undertone runtime health, model selections, voice fingerprints, tests, and the standalone (self-contained) package boundary.
---

# Undertone Ops

Use this skill for common development and operator checks.

## Install

```bash
pip install -e .
pip install -e '.[voice]'
pip install -e '.[meet]'
pip install -e '.[connectors]'
pip install -e '.[voice,meet,connectors]'
```

Use `.[voice]` for Parselmouth acoustic metrics. Use `.[meet]` for Google Meet API/Drive credential support.
Use `.[connectors]` for the YouTube connector's `yt-dlp` dependency.
Google Meet auth uses Google Cloud CLI Application Default Credentials. Configure it with:

```bash
gcloud auth application-default login \
  --scopes="https://www.googleapis.com/auth/meetings.space.readonly,https://www.googleapis.com/auth/drive.meet.readonly"
```

FluidAudio is not a Python dependency in this package. Build `fluidaudiocli` from FluidInference/FluidAudio and either put it on `PATH` or set `UNDERTONE_FLUIDAUDIO_CLI`:

```bash
git clone https://github.com/FluidInference/FluidAudio.git
cd FluidAudio
swift build -c release --product fluidaudiocli
export UNDERTONE_FLUIDAUDIO_CLI="$PWD/.build/release/fluidaudiocli"
```

## Runtime Inspection

```bash
undertone --db ./undertone.db models
undertone --db ./undertone.db doctor
undertone --db ./undertone.db doctor --all
undertone --db ./undertone.db sources
undertone --db ./undertone.db stats
undertone --db ./undertone.db fingerprints
undertone --db ./undertone.db fingerprints --unnamed --excerpts
undertone --db ./undertone.db fingerprint-label VP-abc123 "Speaker Name"
undertone --db ./undertone.db search "follow up"
```

`models` should show the effective ASR, diarization, VAD, embedding, fingerprint, and threshold selections.
`doctor` checks DB writability, engine health, and optional source readiness. Optional source commands are always visible; there is no per-source enable switch. Sources become ready when their dependency, credentials, or local data is present.
Runtime inspection commands are human-readable by default. For agents and scripts, add `--json` for machine-readable output.

## Maintenance

```bash
undertone --db ./undertone.db list --limit 20
undertone --db ./undertone.db reenrich meeting-1 --turn-gap-ms 600
undertone --db ./undertone.db delete meeting-1 --yes
```

`reenrich` rebuilds enrichment from the saved `RawTranscript` without retranscribing audio; use it after changing thresholds or feature toggles. `delete` removes a saved transcript and requires `--yes` to proceed.

Ingest commands (`run-wav`, `finalize-json`, connectors, meeting sources) fail on a duplicate transcript id rather than overwriting. Pass `--force` to overwrite or `--skip-existing` to no-op.

## Verification

```bash
PYTHONPATH=src pytest -q tests
python -m compileall -q src tests
python -m build
```

For package-boundary checks, confirm `undertone` stays self-contained and imports only its own `undertone_audio` package plus declared dependencies:

```bash
rg -n "^\s*(from|import)\s+" src/undertone_audio | rg -v "undertone_audio|asyncio|json|sqlite3|logging|hashlib|pathlib|dataclasses|datetime|argparse|os|sys|subprocess|typing|collections|re|hmac|wave|math|urllib|xml|email|__future__|pydantic|requests|numpy|soundfile|parselmouth|google|yt_dlp"
```

The command should print nothing; any line means a new external or host-application import to review.

## Smoke Tests

```bash
undertone --help
undertone --db /tmp/undertone-smoke.db doctor
undertone --db /tmp/undertone-smoke.db models
undertone quill-list --limit 1
undertone meet-discover --limit 1 --no-probe
```

Use `--no-probe` for Meet discovery when credentials are unavailable or when you only need to validate command wiring.

## Debugging Posture

- Check the CLI surface first, then the relevant command module under `src/undertone_audio/commands/`.
- Keep CLI handlers thin; put source-specific behavior in source modules or command modules.
- Preserve the Undertone privacy boundary when adding metadata, exports, migration fields, or webhooks.
- Prefer focused tests for the exact operator path being changed, then run the full `tests` suite before calling the change done.
