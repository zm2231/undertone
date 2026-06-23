from __future__ import annotations

from collections import defaultdict

from undertone_audio.schema import Segment


def talk_time_per_speaker(segments: list[Segment]) -> dict[str, int]:
    spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for segment in segments:
        spans[segment.speaker_id].append((segment.start_ms, segment.end_ms))

    totals: dict[str, int] = {}
    for speaker, speaker_spans in spans.items():
        speaker_spans.sort()
        current_start, current_end = speaker_spans[0]
        total_ms = 0
        for start, end in speaker_spans[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                total_ms += current_end - current_start
                current_start, current_end = start, end
        total_ms += current_end - current_start
        totals[speaker] = total_ms
    return totals


def talk_ratio_per_speaker(segments: list[Segment], total_duration_ms: int) -> dict[str, float]:
    if total_duration_ms <= 0:
        return {}
    return {
        speaker: duration_ms / total_duration_ms
        for speaker, duration_ms in talk_time_per_speaker(segments).items()
    }


def word_count_per_speaker(segments: list[Segment]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for segment in segments:
        counts[segment.speaker_id] += len(segment.text.split())
    return dict(counts)


def wpm_per_speaker(segments: list[Segment]) -> dict[str, float]:
    talk = talk_time_per_speaker(segments)
    words = word_count_per_speaker(segments)
    return {
        speaker: (words[speaker] / (talk[speaker] / 60_000)) if talk[speaker] > 0 else 0.0
        for speaker in talk
    }
