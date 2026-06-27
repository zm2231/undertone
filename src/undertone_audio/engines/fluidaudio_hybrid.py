"""FluidAudio hybrid engine.

Combines FluidAudio transcribe + process + sortformer:
- transcribe gives word-level ASR text and timings
- process gives pyannote-derived local speaker embeddings
- sortformer gives overlap-aware diarization spans
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from undertone_audio.engines.base import RawTranscript
from undertone_audio.engines.fluidaudio_cli import (
    DEFAULT_DIARIZATION_MODEL,
    FluidAudioCLIEngine,
    _gather_or_cancel,
    merge_transcribe_and_diarize,
    words_to_segments,
)
from undertone_audio.processes import load_json_file
from undertone_audio.schema import Segment, Speaker

log = logging.getLogger(__name__)


class FluidAudioHybridEngine(FluidAudioCLIEngine):
    """FluidAudio with Sortformer overlap-aware diarization and process embeddings."""

    name = "fluidaudio-hybrid"

    async def transcribe(self, audio_path: Path) -> RawTranscript:
        with tempfile.TemporaryDirectory(prefix="undertone-fa-hybrid-") as tmpdir:
            root = Path(tmpdir)
            transcribe_json = root / "transcribe.json"
            process_json = root / "process.json"
            sortformer_json = root / "sortformer.json"
            await self._run_three(audio_path, transcribe_json, process_json, sortformer_json)
            t_data = load_json_file(
                transcribe_json,
                producer="FluidAudio transcribe",
                required_keys=("wordTimings",),
            )
            p_data = load_json_file(
                process_json,
                producer="FluidAudio process",
                required_keys=("segments",),
            )
            s_data = load_json_file(
                sortformer_json,
                producer="FluidAudio sortformer",
                required_keys=("segments",),
            )
        return merge_hybrid(t_data, p_data, s_data)

    async def _run_three(
        self,
        audio_path: Path,
        transcribe_out: Path,
        process_out: Path,
        sortformer_out: Path,
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
                    self._process_cmd(audio_path, process_out),
                    label="process",
                )
            ),
            asyncio.create_task(
                self._run(
                    self._sortformer_cmd(audio_path, sortformer_out),
                    label="sortformer",
                )
            ),
        ]
        await _gather_or_cancel(tasks, "fluidaudio hybrid run failed")

    def _sortformer_cmd(self, audio_path: Path, output_json: Path) -> list[str]:
        cmd = [
            self.cli_path,
            "sortformer",
            str(audio_path),
            "--output",
            str(output_json),
        ]
        if self.model_selection.diarization_model != DEFAULT_DIARIZATION_MODEL:
            cmd.extend(["--model", self.model_selection.diarization_model])
        return cmd


def merge_hybrid(
    t_data: dict[str, Any],
    p_data: dict[str, Any],
    s_data: dict[str, Any],
) -> RawTranscript:
    process_segments = sorted(
        p_data.get("segments", []),
        key=lambda segment: segment["startTimeSeconds"],
    )
    sort_segments = sorted(
        s_data.get("segments", []),
        key=lambda segment: segment["startTimeSeconds"],
    )
    if not sort_segments:
        return merge_transcribe_and_diarize(t_data, p_data)

    embeddings_by_process: dict[str, list[list[float]]] = defaultdict(list)
    for segment in process_segments:
        if segment.get("embedding"):
            embeddings_by_process[segment["speakerId"]].append(segment["embedding"])

    process_mean_embedding: dict[str, list[float]] = {
        speaker_id: [sum(column) / len(column) for column in zip(*embeddings)]
        for speaker_id, embeddings in embeddings_by_process.items()
    }
    sort_to_process = map_speakers_by_overlap(sort_segments, process_segments)

    speakers: list[Speaker] = []
    for speaker_id in sorted({segment["speaker"] for segment in sort_segments}):
        process_id = sort_to_process.get(speaker_id)
        speakers.append(
            Speaker(
                speaker_id=speaker_id,
                embedding=process_mean_embedding.get(process_id) if process_id else None,
            )
        )

    if not speakers and process_segments:
        speakers = [
            Speaker(speaker_id=speaker_id, embedding=process_mean_embedding.get(speaker_id))
            for speaker_id in sorted({segment["speakerId"] for segment in process_segments})
        ]

    diar_segments = [
        {
            "speakerId": segment["speaker"],
            "startTimeSeconds": segment["startTimeSeconds"],
            "endTimeSeconds": segment["endTimeSeconds"],
            "qualityScore": _overlap_quality(segment, process_segments),
        }
        for segment in sort_segments
    ]
    segments: list[Segment] = words_to_segments(t_data.get("wordTimings", []), diar_segments)

    duration_ms = int(s_data.get("durationSeconds", 0) * 1000)
    if duration_ms == 0 and segments:
        duration_ms = max(segment.end_ms for segment in segments)

    return RawTranscript(
        duration_ms=duration_ms,
        language="en",
        speakers=speakers,
        segments=segments,
        engine="fluidaudio-hybrid",
    )


def map_speakers_by_overlap(
    sort_segments: list[dict],
    process_segments: list[dict],
) -> dict[str, str]:
    overlap: dict[tuple[str, str], float] = defaultdict(float)
    for sort_segment in sort_segments:
        s_start = sort_segment["startTimeSeconds"]
        s_end = sort_segment["endTimeSeconds"]
        for process_segment in process_segments:
            p_start = process_segment["startTimeSeconds"]
            p_end = process_segment["endTimeSeconds"]
            if p_end <= s_start or p_start >= s_end:
                continue
            overlap_s = min(s_end, p_end) - max(s_start, p_start)
            if overlap_s > 0:
                overlap[(sort_segment["speaker"], process_segment["speakerId"])] += overlap_s

    by_sort: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (sort_label, process_id), seconds in overlap.items():
        by_sort[sort_label].append((process_id, seconds))

    mapping: dict[str, str] = {}
    for sort_label, candidates in by_sort.items():
        candidates.sort(key=lambda item: -item[1])
        mapping[sort_label] = candidates[0][0]
    return mapping


def _overlap_quality(sort_segment: dict, process_segments: list[dict]) -> float | None:
    weighted_quality = 0.0
    overlap_total = 0.0
    s_start = sort_segment["startTimeSeconds"]
    s_end = sort_segment["endTimeSeconds"]
    for process_segment in process_segments:
        quality = process_segment.get("qualityScore")
        if quality is None:
            continue
        p_start = process_segment["startTimeSeconds"]
        p_end = process_segment["endTimeSeconds"]
        if p_end <= s_start or p_start >= s_end:
            continue
        overlap_s = min(s_end, p_end) - max(s_start, p_start)
        if overlap_s <= 0:
            continue
        weighted_quality += overlap_s * float(quality)
        overlap_total += overlap_s
    if overlap_total == 0:
        return None
    return weighted_quality / overlap_total
