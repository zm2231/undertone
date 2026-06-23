from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from undertone_audio.schema import Segment


@dataclass
class Gap:
    after_segment_id: str
    before_segment_id: str
    speaker_before: str
    speaker_after: str
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def find_gaps(segments: list[Segment], min_gap_ms: int = 250) -> list[Gap]:
    gaps: list[Gap] = []
    for previous, current in zip(segments, segments[1:]):
        gap_ms = current.start_ms - previous.end_ms
        if gap_ms >= min_gap_ms:
            gaps.append(
                Gap(
                    after_segment_id=previous.segment_id,
                    before_segment_id=current.segment_id,
                    speaker_before=previous.speaker_id,
                    speaker_after=current.speaker_id,
                    start_ms=previous.end_ms,
                    end_ms=current.start_ms,
                )
            )
    return gaps


def pause_profile_per_speaker(
    segments: list[Segment],
    min_gap_ms: int = 250,
) -> dict[str, tuple[int, float]]:
    counts: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)

    for segment in segments:
        for previous, current in zip(segment.words, segment.words[1:]):
            gap = current.start_ms - previous.end_ms
            if gap >= min_gap_ms:
                counts[segment.speaker_id] += 1
                totals[segment.speaker_id] += gap

    return {
        speaker: (counts[speaker], totals[speaker] / counts[speaker] if counts[speaker] else 0.0)
        for speaker in counts
    }
