from __future__ import annotations

from collections import defaultdict

from undertone_audio.schema import Segment


def annotate_turn_taking(
    segments: list[Segment],
    overlap_threshold_ms: int = 200,
    rapid_switch_threshold_ms: int = 150,
    prev_short_turn_max_ms: int = 2_000,
    min_response_ms: int = 600,
) -> None:
    for previous, current in zip(segments, segments[1:]):
        delta = current.start_ms - previous.end_ms
        if delta < 0:
            current.enrichment.overlap_with_prev_ms = -delta
            if (
                current.speaker_id != previous.speaker_id
                and -delta >= overlap_threshold_ms
            ):
                current.enrichment.is_interruption = True
        else:
            current.enrichment.gap_before_ms = delta
            if (
                current.speaker_id != previous.speaker_id
                and delta <= rapid_switch_threshold_ms
                and previous.duration_ms <= prev_short_turn_max_ms
                and current.duration_ms >= min_response_ms
            ):
                current.enrichment.is_interruption = True


def interruption_counts(segments: list[Segment]) -> dict[str, tuple[int, int]]:
    made: dict[str, int] = defaultdict(int)
    received: dict[str, int] = defaultdict(int)
    for previous, current in zip(segments, segments[1:]):
        if current.enrichment.is_interruption:
            made[current.speaker_id] += 1
            received[previous.speaker_id] += 1
    speakers = set(made) | set(received)
    return {speaker: (made[speaker], received[speaker]) for speaker in speakers}
