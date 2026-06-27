"""FluidAudio CLI adapter.

Runs FluidAudio transcription and diarization locally through ``fluidaudiocli``.
The adapter keeps ASR text, diarization spans, word timing, and speaker
embeddings inside undertone's raw producer schema.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from undertone_audio.engines.base import RawTranscript
from undertone_audio.processes import load_json_file, run_process_async
from undertone_audio.schema import Segment, Speaker, Word

log = logging.getLogger(__name__)

DEFAULT_ASR_MODEL = "FluidAudio Parakeet TDT"
DEFAULT_DIARIZATION_MODEL = "FluidAudio Sortformer + process"
DEFAULT_VAD_MODEL = "FluidAudio/Silero VAD"
DEFAULT_EMBEDDING_MODEL = "FluidAudio pyannote-derived speaker embeddings"


@dataclass(frozen=True)
class FluidAudioModelSelection:
    asr_model: str = DEFAULT_ASR_MODEL
    diarization_model: str = DEFAULT_DIARIZATION_MODEL
    vad_model: str = DEFAULT_VAD_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL


def _default_cli_path() -> str:
    candidates = [
        os.environ.get("UNDERTONE_FLUIDAUDIO_CLI"),
        os.environ.get("FLUIDAUDIO_CLI"),
        shutil.which("fluidaudiocli"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        "fluidaudiocli binary not found. Set UNDERTONE_FLUIDAUDIO_CLI, set "
        "FLUIDAUDIO_CLI, or put fluidaudiocli on PATH."
    )


class FluidAudioCLIEngine:
    """FluidAudio ASR plus pyannote-derived diarization via CLI."""

    name = "fluidaudio-cli"

    def __init__(
        self,
        cli_path: str | None = None,
        clustering_threshold: float = 0.7045655,
        model_selection: FluidAudioModelSelection | None = None,
        process_timeout_seconds: float = 7200.0,
    ):
        self.cli_path = cli_path or _default_cli_path()
        self.clustering_threshold = clustering_threshold
        self.model_selection = model_selection or FluidAudioModelSelection()
        self.process_timeout_seconds = process_timeout_seconds

    async def healthcheck(self) -> bool:
        return Path(self.cli_path).exists()

    async def transcribe(self, audio_path: Path) -> RawTranscript:
        with tempfile.TemporaryDirectory(prefix="undertone-fa-cli-") as tmpdir:
            transcribe_json = Path(tmpdir) / "transcribe.json"
            diarize_json = Path(tmpdir) / "diarize.json"
            await self._run_parallel(audio_path, transcribe_json, diarize_json)
            t_data = load_json_file(
                transcribe_json,
                producer="FluidAudio transcribe",
                required_keys=("wordTimings",),
            )
            d_data = load_json_file(
                diarize_json,
                producer="FluidAudio process",
                required_keys=("segments",),
            )
        return merge_transcribe_and_diarize(t_data, d_data)

    async def transcribe_single_track(self, audio_path: Path, speaker_id: str) -> RawTranscript:
        with tempfile.TemporaryDirectory(prefix="undertone-fa-cli-") as tmpdir:
            out = Path(tmpdir) / "transcribe.json"
            await self._run(
                self._transcribe_cmd(
                    audio_path,
                    out,
                ),
                label=f"transcribe:{speaker_id}",
            )

            t_data = load_json_file(
                out,
                producer="FluidAudio transcribe",
                required_keys=("wordTimings",),
            )
        return transcribe_only(t_data, speaker_id)

    def _transcribe_cmd(self, audio_path: Path, output_json: Path) -> list[str]:
        cmd = [
            self.cli_path,
            "transcribe",
            str(audio_path),
            "--output-json",
            str(output_json),
            "--word-timestamps",
        ]
        if self.model_selection.asr_model != DEFAULT_ASR_MODEL:
            cmd.extend(["--model", self.model_selection.asr_model])
        return cmd

    def _process_cmd(self, audio_path: Path, output_json: Path) -> list[str]:
        cmd = [
            self.cli_path,
            "process",
            str(audio_path),
            "--mode",
            "offline",
            "--threshold",
            str(self.clustering_threshold),
            "--output",
            str(output_json),
        ]
        if self.model_selection.embedding_model != DEFAULT_EMBEDDING_MODEL:
            cmd.extend(["--embedding-model", self.model_selection.embedding_model])
        if self.model_selection.vad_model != DEFAULT_VAD_MODEL:
            cmd.extend(["--vad-model", self.model_selection.vad_model])
        if self.model_selection.diarization_model != DEFAULT_DIARIZATION_MODEL:
            cmd.extend(["--model", self.model_selection.diarization_model])
        return cmd

    async def transcribe_dual_track(
        self,
        track_a: Path,
        track_b: Path,
        speaker_a: str = "TRACK-A",
        speaker_b: str = "TRACK-B",
    ) -> RawTranscript:
        raw_a = await self.transcribe_single_track(track_a, speaker_a)
        raw_b = await self.transcribe_single_track(track_b, speaker_b)
        return merge_dual_track(raw_a, raw_b)

    async def _run_parallel(
        self,
        audio_path: Path,
        transcribe_out: Path,
        diarize_out: Path,
    ) -> None:
        tasks = [
            asyncio.create_task(
                self._run(
                    self._transcribe_cmd(audio_path, transcribe_out),
                    label="transcribe",
                )
            ),
            asyncio.create_task(
                self._run(
                    self._process_cmd(audio_path, diarize_out),
                    label="diarize",
                )
            ),
        ]
        await _gather_or_cancel(tasks, "fluidaudio parallel run failed")

    async def _run(self, cmd: list[str], label: str) -> None:
        log.info("fluidaudio %s: %s", label, " ".join(cmd))
        await run_process_async(
            cmd,
            label=f"fluidaudio {label}",
            timeout_seconds=self.process_timeout_seconds,
        )


async def _gather_or_cancel(tasks: list[asyncio.Task], message: str) -> None:
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors = [
            result
            for result in results
            if isinstance(result, BaseException)
            and not isinstance(result, asyncio.CancelledError)
        ]
        if errors:
            details = " | ".join(f"{type(error).__name__}: {error}" for error in errors)
            raise RuntimeError(f"{message}: {details}") from errors[0]
        raise


def transcribe_only(t_data: dict[str, Any], speaker_id: str, gap_ms: int = 1500) -> RawTranscript:
    word_timings = t_data.get("wordTimings", [])
    segments: list[Segment] = []
    current_words: list[Word] = []
    seg_start_ms = 0

    for word in word_timings:
        start_ms = int(word["startTime"] * 1000)
        end_ms = int(word["endTime"] * 1000)
        word_obj = Word(
            text=word["word"],
            start_ms=start_ms,
            end_ms=end_ms,
            confidence=word.get("confidence"),
        )
        if current_words and start_ms - current_words[-1].end_ms > gap_ms:
            segments.append(
                Segment(
                    segment_id=f"fa-{len(segments) + 1}",
                    speaker_id=speaker_id,
                    start_ms=seg_start_ms,
                    end_ms=current_words[-1].end_ms,
                    text=" ".join(item.text for item in current_words),
                    words=list(current_words),
                )
            )
            current_words = []
            seg_start_ms = start_ms
        if not current_words:
            seg_start_ms = start_ms
        current_words.append(word_obj)

    if current_words:
        segments.append(
            Segment(
                segment_id=f"fa-{len(segments) + 1}",
                speaker_id=speaker_id,
                start_ms=seg_start_ms,
                end_ms=current_words[-1].end_ms,
                text=" ".join(item.text for item in current_words),
                words=list(current_words),
            )
        )

    return RawTranscript(
        duration_ms=segments[-1].end_ms if segments else 0,
        language="en",
        speakers=[Speaker(speaker_id=speaker_id)],
        segments=segments,
        engine="fluidaudio-cli",
    )


def merge_dual_track(a: RawTranscript, b: RawTranscript) -> RawTranscript:
    all_segments = sorted(a.segments + b.segments, key=lambda segment: segment.start_ms)
    renumbered = [
        segment.model_copy(update={"segment_id": f"fa-{index + 1}"})
        for index, segment in enumerate(all_segments)
    ]
    speaker_ids = {speaker.speaker_id for speaker in a.speakers + b.speakers}
    return RawTranscript(
        duration_ms=max(a.duration_ms, b.duration_ms),
        language=a.language,
        speakers=[Speaker(speaker_id=speaker_id) for speaker_id in sorted(speaker_ids)],
        segments=renumbered,
        engine="fluidaudio-cli",
    )


def merge_transcribe_and_diarize(
    t_data: dict[str, Any],
    d_data: dict[str, Any],
) -> RawTranscript:
    diar_segments = sorted(
        d_data.get("segments", []),
        key=lambda segment: segment["startTimeSeconds"],
    )
    word_timings = t_data.get("wordTimings", [])

    embeddings_by_speaker: dict[str, list[list[float]]] = defaultdict(list)
    for segment in diar_segments:
        if segment.get("embedding"):
            embeddings_by_speaker[segment["speakerId"]].append(segment["embedding"])

    speakers = [
        Speaker(
            speaker_id=speaker_id,
            embedding=[sum(column) / len(column) for column in zip(*embeddings)],
        )
        for speaker_id, embeddings in embeddings_by_speaker.items()
    ]
    if not speakers:
        speakers = [
            Speaker(speaker_id=speaker_id)
            for speaker_id in sorted({segment["speakerId"] for segment in diar_segments})
        ]

    segments = words_to_segments(word_timings, diar_segments)
    duration_ms = int(d_data.get("durationSeconds", 0) * 1000)
    if duration_ms == 0 and segments:
        duration_ms = max(segment.end_ms for segment in segments)

    return RawTranscript(
        duration_ms=duration_ms,
        language="en",
        speakers=speakers,
        segments=segments,
        engine="fluidaudio-cli",
    )


def words_to_segments(word_timings: list[dict], diar_segments: list[dict]) -> list[Segment]:
    if not diar_segments:
        return []

    spans = sorted(diar_segments, key=lambda segment: segment["startTimeSeconds"])
    words_per_span: dict[int, list[Word]] = {index: [] for index in range(len(spans))}

    for word in word_timings or []:
        midpoint_s = (word["startTime"] + word["endTime"]) / 2
        word_obj = Word(
            text=word["word"],
            start_ms=int(word["startTime"] * 1000),
            end_ms=int(word["endTime"] * 1000),
            confidence=word.get("confidence"),
        )
        owners = [
            index
            for index, span in enumerate(spans)
            if span["startTimeSeconds"] <= midpoint_s <= span["endTimeSeconds"]
        ]
        if owners:
            best = max(
                owners,
                key=lambda index: min(spans[index]["endTimeSeconds"], word["endTime"])
                - max(spans[index]["startTimeSeconds"], word["startTime"]),
            )
            words_per_span[best].append(word_obj)
        else:
            nearest = min(
                range(len(spans)),
                key=lambda index: min(
                    abs(spans[index]["startTimeSeconds"] - midpoint_s),
                    abs(spans[index]["endTimeSeconds"] - midpoint_s),
                ),
            )
            words_per_span[nearest].append(word_obj)

    segments: list[Segment] = []
    for index, span in enumerate(spans):
        words = words_per_span[index]
        segments.append(
            Segment(
                segment_id=f"fa-{index + 1}",
                speaker_id=span["speakerId"],
                start_ms=int(span["startTimeSeconds"] * 1000),
                end_ms=int(span["endTimeSeconds"] * 1000),
                text=" ".join(word.text for word in words),
                diarization_quality=span.get("qualityScore"),
                words=words,
            )
        )
    return segments
