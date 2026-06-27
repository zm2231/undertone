# Diarization Backends

Undertone keeps audio processing local. The default backend is `fluidaudio-hybrid`; the optional `fluidaudio-pyannote` backend is available when you install the pyannote extra.

`fluidaudio-pyannote` is not installed by default. This keeps the base package light and avoids pulling the torch/pyannote dependency stack unless a user chooses that backend.

## `fluidaudio-hybrid`

This is the default engine. It runs FluidAudio ASR, FluidAudio process output, and FluidAudio Sortformer diarization, then merges those results into Undertone's transcript schema.

Use it when you want the standard on-device FluidAudio path:

```bash
undertone run-wav ./meeting.wav --engine fluidaudio-hybrid
```

Sortformer spans do not expose a confidence score. Undertone overlap-maps FluidAudio process `qualityScore` onto final Sortformer spans when possible and stores that nullable approximation as `segments[].diarization_quality`.

## `fluidaudio-pyannote`

This engine runs FluidAudio ASR for word timings and runs `pyannote.audio` in-process for diarization spans and speaker embeddings. It is useful when Sortformer under-splits speakers and you need pyannote's speaker separation.

Install it explicitly:

```bash
pip install 'undertone-audio[pyannote]'
```

Then check that the optional Python dependency imports:

```bash
undertone doctor --check-pyannote
```

This is not a full gated-model readiness check. It does not download or load the selected Hugging Face model; model access is verified when `fluidaudio-pyannote` runs.

Run it:

```bash
undertone run-wav ./meeting.wav --engine fluidaudio-pyannote
```

The default pyannote model is `pyannote/speaker-diarization-community-1`. You can use aliases or full Hugging Face model IDs:

```bash
undertone run-wav ./meeting.wav \
  --engine fluidaudio-pyannote \
  --pyannote-model community-1 \
  --pyannote-device auto
```

Supported shorthand aliases:

- `community-1` -> `pyannote/speaker-diarization-community-1`
- `3.1` -> `pyannote/speaker-diarization-3.1`

Environment equivalents:

```bash
export UNDERTONE_ENGINE=fluidaudio-pyannote
export UNDERTONE_PYANNOTE_MODEL=pyannote/speaker-diarization-community-1
export UNDERTONE_PYANNOTE_DEVICE=auto
```

Return to the default engine with:

```bash
unset UNDERTONE_ENGINE
# or
export UNDERTONE_ENGINE=fluidaudio-hybrid
```

If the selected pyannote model is gated, accept the model terms on Hugging Face and set `HF_TOKEN` or `HUGGINGFACE_TOKEN`.

Pyannote's public `DiarizeOutput` exposes diarization annotations and speaker embeddings, but not per-span confidence or posterior fields. Undertone therefore keeps `segments[].diarization_quality` as `null` on this backend.

## Confidence Fields

Undertone always preserves FluidAudio word confidence as `words[].confidence` when FluidAudio emits it, and derives `segments[].asr_confidence` from segment words. Diarization quality is backend-specific: direct FluidAudio process `qualityScore` on `fluidaudio-cli`, overlap-mapped process quality on `fluidaudio-hybrid`, and `null` on `fluidaudio-pyannote`.

## Execution Order

`fluidaudio-pyannote` runs FluidAudio ASR first and starts pyannote only after ASR succeeds. If ASR fails, pyannote never starts, so a failed run does not leave a diarization model loading in the background. This trades a little wall-clock time for predictable behavior; the backend is quality-oriented rather than speed-oriented.

There is no mid-run cancellation. Once pyannote is running, a slow or hung run completes before the command returns, because the CLI waits for the diarization thread to finish. Bounding or killing a running diarization would require a subprocess worker, which is not implemented yet.

## Comparing Backends

Diarization quality is sensitive to recording source, channel separation, microphone placement, overlap, noise, and expected speaker count. If you compare backends, use reproducible samples and record:

- audio source and license
- engine and model names
- FluidAudio build/version
- pyannote version
- device/backend
- expected speaker count source
- scoring method and acceptance criteria
