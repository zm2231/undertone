from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from undertone_audio.engines.base import RawTranscript
from undertone_audio.schema import EnrichedTranscript

OUTPUT_FORMATS = {"json", "raw-json", "text", "md", "jsonl", "csv"}
OUTPUT_DETAIL_LEVELS = {"minimal", "standard", "full"}

ACOUSTIC_METRIC_FIELDS = {
    "articulation_rate",
    "f0_mean_hz",
    "f0_stdev_hz",
    "jitter_local",
    "shimmer_local",
    "voiced_duration_s",
}


def render_transcript(
    transcript: EnrichedTranscript,
    output_format: str = "json",
    *,
    raw: RawTranscript | None = None,
    detail: str = "full",
) -> str:
    fmt = output_format.lower()
    detail_level = detail.lower()
    if fmt not in OUTPUT_FORMATS:
        raise ValueError(f"unknown output format {output_format!r}; expected {sorted(OUTPUT_FORMATS)}")
    if detail_level not in OUTPUT_DETAIL_LEVELS:
        raise ValueError(f"unknown output detail {detail!r}; expected {sorted(OUTPUT_DETAIL_LEVELS)}")
    if fmt == "json":
        return json.dumps(_transcript_payload(transcript, detail_level), separators=(",", ":"), default=str)
    if fmt == "raw-json":
        if raw is None:
            raw = RawTranscript(
                duration_ms=transcript.metadata.duration_ms,
                language=transcript.metadata.language,
                speakers=transcript.speakers,
                segments=transcript.segments,
                engine=transcript.metadata.engine,
            )
        return json.dumps(_raw_payload(raw, detail_level), separators=(",", ":"), default=str)
    if fmt == "jsonl":
        return "\n".join(
            json.dumps(
                _segment_payload(transcript.transcript_id, segment, detail_level),
                default=str,
                separators=(",", ":"),
            )
            for segment in transcript.segments
        )
    if fmt == "csv":
        return _render_csv(transcript, detail_level)
    if fmt == "md":
        return _render_markdown(transcript, detail_level)
    return _render_text(transcript, detail_level)


def write_or_print(body: str, output: Path | None = None) -> None:
    if output is None:
        print(body)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body + ("" if body.endswith("\n") else "\n"))


def _timestamp(ms: int) -> str:
    hours, remainder = divmod(ms // 1000, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def _speaker_name(transcript: EnrichedTranscript, speaker_id: str) -> str:
    for speaker in transcript.speakers:
        if speaker.speaker_id == speaker_id:
            return speaker.display_name or speaker.fingerprint_id or speaker.speaker_id
    return speaker_id


def _transcript_payload(transcript: EnrichedTranscript, detail: str) -> dict[str, Any]:
    payload = transcript.model_dump(mode="json")
    if detail == "full":
        return payload
    for segment in payload["segments"]:
        segment.pop("words", None)
        if detail == "minimal":
            segment.pop("enrichment", None)
    if detail == "minimal":
        payload["speaker_metrics"] = []
        payload["metadata"] = {
            key: payload["metadata"].get(key)
            for key in (
                "source_path",
                "source_url",
                "duration_ms",
                "language",
                "meeting_type",
                "recorded_at",
                "engine",
                "diarization_state",
            )
            if payload["metadata"].get(key) is not None
        }
    else:
        payload["speaker_metrics"] = [
            _drop_acoustic_metrics(metric) for metric in payload["speaker_metrics"]
        ]
    return payload


def _raw_payload(raw: RawTranscript, detail: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "duration_ms": raw.duration_ms,
        "language": raw.language,
        "engine": raw.engine,
        "speakers": [
            {
                key: value
                for key, value in speaker.model_dump(mode="json").items()
                if value is not None and (detail == "full" or key not in {"embedding"})
            }
            for speaker in raw.speakers
        ],
        "segments": [],
    }
    for segment in raw.segments:
        row = {
            "segment_id": segment.segment_id,
            "speaker_id": segment.speaker_id,
            "start_ms": segment.start_ms,
            "end_ms": segment.end_ms,
            "text": segment.text,
        }
        if detail == "full":
            row["words"] = [word.model_dump(mode="json") for word in segment.words]
        payload["segments"].append(row)
    return payload


def _segment_payload(transcript_id: str, segment, detail: str) -> dict[str, Any]:
    payload = {
        "transcript_id": transcript_id,
        "segment_id": segment.segment_id,
        "speaker_id": segment.speaker_id,
        "start_ms": segment.start_ms,
        "end_ms": segment.end_ms,
        "text": segment.text,
    }
    if detail == "full":
        payload["words"] = [word.model_dump(mode="json") for word in segment.words]
        payload["enrichment"] = segment.enrichment.model_dump(mode="json")
    elif detail == "standard":
        payload["enrichment"] = segment.enrichment.model_dump(mode="json")
    return payload


def _drop_acoustic_metrics(metric: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metric.items() if key not in ACOUSTIC_METRIC_FIELDS}


def _render_text(transcript: EnrichedTranscript, detail: str) -> str:
    metadata = transcript.metadata
    lines = [
        "=" * 70,
        transcript.transcript_id,
        f"{metadata.duration_ms / 1000 / 60:.1f} min | {metadata.engine}",
        "=" * 70,
        "",
        "SPEAKERS",
        "-" * 40,
    ]
    for metric in sorted(transcript.speaker_metrics, key=lambda item: -item.talk_ratio):
        name = _speaker_name(transcript, metric.speaker_id)
        line = (
            f"  {name}: {metric.talk_ratio * 100:.1f}% | "
            f"{metric.word_count} words | {metric.wpm:.0f} wpm"
        )
        if detail == "full":
            f0 = f"{metric.f0_mean_hz:.0f}Hz" if metric.f0_mean_hz else "-"
            ar = f"{metric.articulation_rate:.1f}" if metric.articulation_rate else "-"
            jitter = f"{metric.jitter_local:.4f}" if metric.jitter_local is not None else "-"
            line += f" | f0={f0} | ar={ar} | jitter={jitter}"
        if detail != "minimal":
            line += (
                f" | fillers={metric.filler_count} | "
            f"interruptions={metric.interruptions_made}/{metric.interruptions_received}"
            )
        lines.append(line)
    lines += ["", "TRANSCRIPT", "-" * 40]
    previous = None
    for segment in transcript.segments:
        if not segment.text.strip():
            continue
        if segment.speaker_id != previous:
            lines.append(f"\n[{_timestamp(segment.start_ms)}] {_speaker_name(transcript, segment.speaker_id)}")
            previous = segment.speaker_id
        prefix = "  [INT] " if segment.enrichment.is_interruption else "  "
        lines.append(prefix + segment.text.strip())
    return "\n".join(lines)


def _render_markdown(transcript: EnrichedTranscript, detail: str) -> str:
    metadata = transcript.metadata
    lines = [
        f"# Transcript {transcript.transcript_id}",
        "",
        f"- Engine: `{metadata.engine}`",
        f"- Duration: `{metadata.duration_ms / 1000:.1f}s`",
        f"- Meeting type: `{metadata.meeting_type.value}`",
        "",
        "## Speakers",
        "",
    ]
    for metric in sorted(transcript.speaker_metrics, key=lambda item: -item.talk_ratio):
        name = _speaker_name(transcript, metric.speaker_id)
        line = (
            f"- **{name}**: {metric.talk_ratio * 100:.1f}% talk, "
            f"{metric.word_count} words, {metric.wpm:.0f} wpm"
        )
        if detail == "full":
            f0 = f"{metric.f0_mean_hz:.0f}Hz" if metric.f0_mean_hz else "-"
            jitter = f"{metric.jitter_local:.4f}" if metric.jitter_local is not None else "-"
            line += f", f0 {f0}, jitter {jitter}"
        elif detail == "standard":
            line += f", fillers {metric.filler_count}"
        lines.append(line)
    lines += ["", "## Transcript", ""]
    previous = None
    for segment in transcript.segments:
        if not segment.text.strip():
            continue
        if segment.speaker_id != previous:
            lines.append(f"### {_timestamp(segment.start_ms)} {_speaker_name(transcript, segment.speaker_id)}")
            previous = segment.speaker_id
        marker = "**[INT]** " if segment.enrichment.is_interruption else ""
        lines.append(f"{marker}{segment.text.strip()}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _render_csv(transcript: EnrichedTranscript, detail: str) -> str:
    import csv
    import io

    buffer = io.StringIO()
    fields = [
        "transcript_id",
        "speaker_id",
        "speaker_name",
        "talk_time_ms",
        "talk_ratio",
        "word_count",
        "wpm",
        "pause_count",
        "avg_pause_ms",
        "filler_count",
        "filler_rate",
        "interruptions_made",
        "interruptions_received",
    ]
    if detail == "full":
        fields.extend(
            [
                "articulation_rate",
                "f0_mean_hz",
                "f0_stdev_hz",
                "jitter_local",
                "shimmer_local",
                "voiced_duration_s",
            ]
        )
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for metric in transcript.speaker_metrics:
        row = {
            "transcript_id": transcript.transcript_id,
            "speaker_id": metric.speaker_id,
            "speaker_name": _speaker_name(transcript, metric.speaker_id),
            "talk_time_ms": metric.talk_time_ms,
            "talk_ratio": metric.talk_ratio,
            "word_count": metric.word_count,
            "wpm": metric.wpm,
            "pause_count": metric.pause_count,
            "avg_pause_ms": metric.avg_pause_ms,
            "filler_count": metric.filler_count,
            "filler_rate": metric.filler_rate,
            "interruptions_made": metric.interruptions_made,
            "interruptions_received": metric.interruptions_received,
        }
        if detail == "full":
            row.update(
                {
                    "articulation_rate": metric.articulation_rate,
                    "f0_mean_hz": metric.f0_mean_hz,
                    "f0_stdev_hz": metric.f0_stdev_hz,
                    "jitter_local": metric.jitter_local,
                    "shimmer_local": metric.shimmer_local,
                    "voiced_duration_s": metric.voiced_duration_s,
                }
            )
        writer.writerow(row)
    return buffer.getvalue().rstrip()
