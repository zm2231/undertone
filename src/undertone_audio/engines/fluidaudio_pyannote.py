"""FluidAudio ASR plus in-process pyannote diarization.

This backend keeps FluidAudio for word-timestamped ASR and uses pyannote.audio
for diarization spans and per-speaker embeddings. pyannote is an optional
dependency, imported lazily only when this backend is selected.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import tempfile
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from undertone_audio.engines.base import RawTranscript
from undertone_audio.engines.fluidaudio_cli import words_to_segments
from undertone_audio.engines.fluidaudio_hybrid import FluidAudioHybridEngine
from undertone_audio.processes import load_json_file
from undertone_audio.schema import Segment, Speaker

log = logging.getLogger(__name__)

DEFAULT_PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"
DEFAULT_PYANNOTE_DEVICE = "auto"
PYANNOTE_INSTALL_FIX = (
    "Install pyannote support with `pip install 'undertone-audio[pyannote]'`."
)
PYANNOTE_MODEL_ALIASES = {
    "community-1": DEFAULT_PYANNOTE_MODEL,
    "3.1": "pyannote/speaker-diarization-3.1",
}


class PyannoteDependencyError(RuntimeError):
    """Raised when the optional pyannote backend is selected without its deps."""


def resolve_pyannote_model(model: str | None) -> str:
    selected = model or DEFAULT_PYANNOTE_MODEL
    return PYANNOTE_MODEL_ALIASES.get(selected, selected)


def pyannote_status(model: str | None = None, device: str | None = None) -> dict[str, Any]:
    """Check import-time readiness without downloading/loading a Hugging Face model."""

    selected_model = resolve_pyannote_model(model)
    selected_device = device or DEFAULT_PYANNOTE_DEVICE
    try:
        _load_pyannote_modules()
    except PyannoteDependencyError as exc:
        return {
            "name": "pyannote",
            "ok": False,
            "model": selected_model,
            "device": selected_device,
            "error": str(exc),
            "detail": "dependency import check only; model access is verified when the backend runs",
            "fix": PYANNOTE_INSTALL_FIX,
        }
    return {
        "name": "pyannote",
        "ok": True,
        "model": selected_model,
        "device": selected_device,
        "detail": "dependency import check only; model access is verified when the backend runs",
        "fix": None,
    }


class FluidAudioPyannoteEngine(FluidAudioHybridEngine):
    """FluidAudio word ASR with pyannote diarization spans and embeddings."""

    name = "fluidaudio-pyannote"

    def __init__(
        self,
        *args,
        pyannote_model: str | None = None,
        pyannote_device: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.pyannote_model = resolve_pyannote_model(pyannote_model)
        self.pyannote_device = pyannote_device or DEFAULT_PYANNOTE_DEVICE

    async def healthcheck(self) -> bool:
        base_ok = await super().healthcheck()
        return bool(base_ok and pyannote_status(self.pyannote_model, self.pyannote_device)["ok"])

    async def transcribe(self, audio_path: Path) -> RawTranscript:
        with tempfile.TemporaryDirectory(prefix="undertone-fa-pyannote-") as tmpdir:
            transcribe_json = Path(tmpdir) / "transcribe.json"
            pyannote_data = await self._run_asr_diar(audio_path, transcribe_json)
            t_data = load_json_file(
                transcribe_json,
                producer="FluidAudio transcribe",
                required_keys=("wordTimings",),
            )
        return merge_pyannote(t_data, pyannote_data, engine_name=self.name)

    async def _run_asr_diar(
        self,
        audio_path: Path,
        transcribe_out: Path,
    ) -> dict[str, Any]:
        await self._run(self._transcribe_cmd(audio_path, transcribe_out), label="transcribe")
        return await asyncio.to_thread(self._run_pyannote, audio_path)

    def _run_pyannote(self, audio_path: Path) -> dict[str, Any]:
        torch, torchaudio, pipeline_cls = _load_pyannote_modules()
        token = _hf_token()
        try:
            try:
                pipeline = pipeline_cls.from_pretrained(self.pyannote_model, token=token)
            except TypeError:
                pipeline = pipeline_cls.from_pretrained(self.pyannote_model, use_auth_token=token)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load pyannote model {self.pyannote_model!r}: {exc}. "
                "If the model is gated, accept its Hugging Face terms and set HF_TOKEN."
            ) from exc
        if pipeline is None:
            raise RuntimeError(
                f"failed to load pyannote model {self.pyannote_model!r}. "
                "If the model is gated, accept its Hugging Face terms and set HF_TOKEN."
            )

        device = _resolve_device(self.pyannote_device, torch)
        if device:
            pipeline.to(torch.device(device))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = torchaudio.info(str(audio_path))
            waveform, sample_rate = torchaudio.load(str(audio_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        output = pipeline({"waveform": waveform, "sample_rate": sample_rate})
        duration_seconds = info.num_frames / info.sample_rate if info.sample_rate else 0.0
        return pyannote_output_to_sortformer_json(
            output,
            audio_path=audio_path,
            model=self.pyannote_model,
            duration_seconds=duration_seconds,
        )


def pyannote_output_to_sortformer_json(
    output: Any,
    *,
    audio_path: Path | None = None,
    model: str | None = None,
    duration_seconds: float = 0.0,
) -> dict[str, Any]:
    """Convert pyannote output to the span/embedding JSON shape Undertone merges."""

    annotation = _speaker_annotation(output)
    labels = sorted(annotation.labels()) if hasattr(annotation, "labels") else []
    label_index = {label: idx for idx, label in enumerate(labels)}

    embeddings_by_speaker = _speaker_embeddings(output, labels, label_index)
    segments = []
    for segment, _track, label in annotation.itertracks(yield_label=True):
        if label not in label_index:
            label_index[label] = len(label_index)
        idx = label_index[label]
        segments.append(
            {
                "speaker": f"Speaker {idx}",
                "speakerIndex": idx,
                "startTimeSeconds": round(float(segment.start), 3),
                "endTimeSeconds": round(float(segment.end), 3),
                "durationSeconds": round(float(segment.end - segment.start), 3),
            }
        )
    segments.sort(key=lambda item: item["startTimeSeconds"])
    return {
        "audioFile": str(audio_path) if audio_path else None,
        "durationSeconds": round(float(duration_seconds), 3),
        "segmentCount": len(segments),
        "speakerCount": len(label_index),
        "model": resolve_pyannote_model(model),
        "segments": segments,
        "speakerEmbeddings": embeddings_by_speaker,
    }


def merge_pyannote(
    t_data: dict[str, Any],
    s_data: dict[str, Any],
    engine_name: str = "fluidaudio-pyannote",
) -> RawTranscript:
    """Build a RawTranscript from FluidAudio ASR words plus pyannote spans."""

    diar_spans = sorted(
        s_data.get("segments", []),
        key=lambda seg: seg["startTimeSeconds"],
    )
    speaker_embeddings: dict[str, list[float]] = s_data.get("speakerEmbeddings", {}) or {}
    diar_segments = [
        {
            "speakerId": seg["speaker"],
            "startTimeSeconds": seg["startTimeSeconds"],
            "endTimeSeconds": seg["endTimeSeconds"],
        }
        for seg in diar_spans
    ]
    segments: list[Segment] = words_to_segments(t_data.get("wordTimings", []), diar_segments)

    speaker_ids = sorted({seg["speaker"] for seg in diar_spans})
    speakers = [
        Speaker(speaker_id=sid, embedding=speaker_embeddings.get(sid)) for sid in speaker_ids
    ]
    duration_ms = int(s_data.get("durationSeconds", 0) * 1000)
    if duration_ms == 0 and segments:
        duration_ms = max(seg.end_ms for seg in segments)

    return RawTranscript(
        duration_ms=duration_ms,
        language="en",
        speakers=speakers,
        segments=segments,
        engine=engine_name,
    )


def _load_pyannote_modules():
    try:
        import torch
        import torchaudio
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise PyannoteDependencyError(
            f"pyannote backend dependencies are not installed. {PYANNOTE_INSTALL_FIX}"
        ) from exc
    return torch, torchaudio, Pipeline


def _speaker_annotation(output: Any) -> Any:
    annotation = getattr(output, "speaker_diarization", None)
    if annotation is not None:
        return annotation
    annotation = getattr(output, "exclusive_speaker_diarization", None)
    if annotation is not None:
        return annotation
    return output


def _speaker_embeddings(
    output: Any,
    labels: list[Any],
    label_index: dict[Any, int],
) -> dict[str, list[float]]:
    raw_embeddings = getattr(output, "speaker_embeddings", None)
    if raw_embeddings is None:
        return {}
    if isinstance(raw_embeddings, Mapping):
        pairs = raw_embeddings.items()
    else:
        try:
            import numpy as np

            arr = np.asarray(raw_embeddings)
        except Exception:
            return {}
        pairs = ((label, arr[idx]) for idx, label in enumerate(labels) if idx < len(arr))

    embeddings: dict[str, list[float]] = {}
    for label, vector in pairs:
        if label not in label_index:
            continue
        cleaned = _clean_vector(vector)
        if cleaned is not None:
            embeddings[f"Speaker {label_index[label]}"] = cleaned
    return embeddings


def _clean_vector(vector: Any) -> list[float] | None:
    try:
        import numpy as np

        arr = np.asarray(vector, dtype=float)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return None
        return [float(item) for item in arr.tolist()]
    except Exception:
        try:
            values = [float(item) for item in vector]
        except Exception:
            return None
        if not values or not all(math.isfinite(item) for item in values):
            return None
        return values


def _resolve_device(device: str | None, torch: Any) -> str | None:
    selected = (device or DEFAULT_PYANNOTE_DEVICE).strip().lower()
    if selected in {"", "none"}:
        return None
    if selected != "auto":
        return selected
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return token
    for path in (
        Path.home() / ".huggingface/token",
        Path.home() / ".cache/huggingface/token",
    ):
        if path.exists():
            return path.read_text().strip()
    return None
