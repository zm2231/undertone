from __future__ import annotations

from undertone_audio.schema import Segment


def merge_adjacent_turns(
    segments: list[Segment],
    gap_threshold_ms: int = 800,
) -> list[Segment]:
    if not segments:
        return []

    merged: list[Segment] = [segments[0].model_copy(deep=True)]
    for segment in segments[1:]:
        previous = merged[-1]
        gap = segment.start_ms - previous.end_ms
        if segment.speaker_id == previous.speaker_id and gap <= gap_threshold_ms:
            previous.end_ms = max(previous.end_ms, segment.end_ms)
            previous.text = f"{previous.text} {segment.text}".strip()
            previous.words.extend(segment.words)
        else:
            merged.append(segment.model_copy(deep=True))
    return merged
