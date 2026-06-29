# Undertone Ops

Use this skill for common development and operator checks.

## Install

```bash
pip install -e .
pip install -e '.[voice]'
pip install -e '.[pyannote]'
pip install -e '.[meet]'
pip install -e '.[connectors]'
pip install -e '.[voice,pyannote,meet,connectors]'
```

Use `.[voice]` for Parselmouth acoustic metrics. Use `.[meet]` for Google Meet API/Drive credential support.
Use `.[connectors]` for the YouTube and web media resolver `yt-dlp` dependency.
Use `.[pyannote]` only when the `fluidaudio-pyannote` engine is needed. A base install should not pull torch/pyannote.
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
undertone --db ./undertone.db doctor --check-pyannote
undertone --db ./undertone.db doctor --check-yt-dlp --yt-dlp-bin /path/to/yt-dlp
undertone --db ./undertone.db sources
undertone --db ./undertone.db stats
undertone --db ./undertone.db fingerprints
undertone --db ./undertone.db fingerprints --format json
undertone --db ./undertone.db fingerprints --unnamed --excerpts
undertone --db ./undertone.db fingerprints --status all
undertone schema transcript
undertone schema connector-asset
undertone schema connector-candidate
undertone connector-list
undertone --db ./undertone.db search "follow up"
```

`models` should show the effective ASR, diarization, VAD, embedding, pyannote, fingerprint, threshold selections, fingerprint model compatibility counts, and active/discarded fingerprint status counts.
`doctor` checks DB writability, voice fingerprint model compatibility, fingerprint status counts, engine health, optional source readiness, and pyannote dependency import when `--check-pyannote` or `--all` is passed. The pyannote check does not download or load a gated Hugging Face model; model access is verified when the backend runs. Optional source commands are always visible; there is no per-source enable switch. Sources become ready when their dependency, credentials, or local data is present.
Runtime inspection commands are human-readable by default. For agents and scripts, add `--json` for machine-readable output.

External binaries are bounded by `UNDERTONE_PROCESS_TIMEOUT_SECONDS` or ingest command `--process-timeout-seconds`. The default is `7200`; use `0` only when intentionally disabling subprocess timeouts.

## Maintenance

```bash
undertone --db ./undertone.db list --limit 20
undertone --db ./undertone.db reenrich meeting-1 --turn-gap-ms 600
undertone --db ./undertone.db relabel meeting-1
undertone --db ./undertone.db delete meeting-1 --yes
```

`reenrich` rebuilds enrichment from the saved `RawTranscript` without retranscribing audio; use it after changing thresholds or feature toggles. It re-runs fingerprint matching against today's fingerprint centroids. `relabel` re-stamps saved speaker display names from the fingerprint DB without ASR, enrichment, or fingerprint matching. `delete` removes a saved transcript and requires `--yes` to proceed.

For speaker fingerprint operations (label, relabel, list/excerpts, discard/restore/destroy, export/import, merge, and embedding-model adoption), see `references/fingerprints.md`. For version upgrades and cross-DB fingerprint portability, see `references/upgrades.md`.

Ingest commands (`run-wav`, `finalize-json`, connectors, meeting sources) fail on a duplicate transcript id rather than overwriting. Pass `--force` to overwrite or `--skip-existing` to no-op. Matching audio across different source ids skips before fingerprint assignment when Chromaprint `fpcalc` is installed; pass `--allow-duplicate` only for intentional duplicate audio storage. Text simhash is stored as advisory metadata. `doctor --check-fpcalc` reports the Chromaprint pre-ASR dedupe gate.

## Verification

```bash
PYTHONPATH=src pytest -q tests
python -m compileall -q src tests
python -m build
```

For package-boundary checks, confirm `undertone` stays self-contained and imports only its own `undertone_audio` package plus declared dependencies:

```bash
rg -n "^(from|import)\s+" src/undertone_audio | rg -v "undertone_audio|asyncio|json|sqlite3|logging|hashlib|pathlib|dataclasses|datetime|argparse|os|sys|subprocess|typing|collections|re|hmac|wave|math|urllib|xml|email|__future__|pydantic|requests|numpy|soundfile|parselmouth|google|yt_dlp"
```

The command should print nothing; any line means a new external or host-application import to review.

## Smoke Tests

```bash
undertone --help
undertone --db /tmp/undertone-smoke.db doctor
undertone --db /tmp/undertone-smoke.db doctor --check-pyannote
undertone --db /tmp/undertone-smoke.db models
undertone quill-list --limit 1
undertone meet-discover --limit 1 --no-probe
```

Use `--no-probe` for Meet discovery when credentials are unavailable or when you only need to validate command wiring.
In a base install, `doctor --check-pyannote` should fail with a fix message that names `pip install 'undertone-audio[pyannote]'`. In a pyannote install, it should pass without downloading or loading a Hugging Face model.

## Benchmark Boundary

Keep benchmark scripts, private benchmark outputs, local backfill drivers, and personal databases ignored/local unless explicitly creating a sanitized public benchmark. Public benchmark docs need reproducible public samples plus engine, model, device, FluidAudio build, pyannote version, expected-speaker source, scoring method, and acceptance criteria.

## Debugging Posture

- Check the CLI surface first, then the relevant command module under `src/undertone_audio/commands/`.
- Keep CLI handlers thin; put source-specific behavior in source modules or command modules.
- Preserve the Undertone privacy boundary when adding metadata, exports, migration fields, or webhooks.
- Prefer focused tests for the exact operator path being changed, then run the full `tests` suite before calling the change done.
