from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

from undertone_audio.schema import Segment, Speaker

log = logging.getLogger(__name__)

DEFAULT_MERGE_THRESHOLD = 0.82
DEFAULT_MIN_TALK_SECONDS = 1.5


@dataclass
class MergeReport:
    original_speaker_count: int
    final_speaker_count: int
    merges: list[tuple[str, str, float]]
    dropped_low_talk: list[tuple[str, float]]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def collapse_overdetected_speakers(
    speakers: list[Speaker],
    segments: list[Segment],
    merge_threshold: float = DEFAULT_MERGE_THRESHOLD,
    min_talk_seconds: float = DEFAULT_MIN_TALK_SECONDS,
    target_speaker_count: int | None = None,
) -> tuple[list[Speaker], list[Segment], MergeReport]:
    original_count = len(speakers)
    talk_seconds = _talk_seconds(segments)

    remap: dict[str, str] = {}
    merges: list[tuple[str, str, float]] = []
    candidates = sorted(speakers, key=lambda speaker: -talk_seconds.get(speaker.speaker_id, 0))

    kept: list[Speaker] = []
    for speaker in candidates:
        if speaker.speaker_id in remap:
            continue
        match = None
        match_similarity = 0.0
        for kept_speaker in kept:
            if not (speaker.embedding and kept_speaker.embedding):
                continue
            similarity = _cosine(speaker.embedding, kept_speaker.embedding)
            if similarity >= merge_threshold and similarity > match_similarity:
                match = kept_speaker
                match_similarity = similarity
        if match is not None:
            remap[speaker.speaker_id] = match.speaker_id
            merges.append((match.speaker_id, speaker.speaker_id, match_similarity))
            log.info(
                "merging speaker %s -> %s (sim=%.3f)",
                speaker.speaker_id,
                match.speaker_id,
                match_similarity,
            )
        else:
            kept.append(speaker)

    remapped_segments = [
        segment.model_copy(update={"speaker_id": remap.get(segment.speaker_id, segment.speaker_id)})
        for segment in segments
    ]

    dropped_low: list[tuple[str, float]] = []
    if min_talk_seconds > 0:
        post_merge_talk = _talk_seconds(remapped_segments)
        ranked = sorted(kept, key=lambda speaker: -post_merge_talk.get(speaker.speaker_id, 0))
        protected = {ranked[0].speaker_id} if ranked else set()
        for speaker in list(kept):
            if speaker.speaker_id in protected:
                continue
            talk = post_merge_talk.get(speaker.speaker_id, 0)
            if talk < min_talk_seconds:
                kept.remove(speaker)
                dropped_low.append((speaker.speaker_id, talk))
                log.info("dropping low-talk speaker %s (%.2fs)", speaker.speaker_id, talk)

        kept_ids = {speaker.speaker_id for speaker in kept}
        if dropped_low and kept_ids:
            speaker_lookup = {speaker.speaker_id: speaker for speaker in speakers}
            dominant_id = max(post_merge_talk, key=post_merge_talk.get) if post_merge_talk else None

            def reassign(speaker_id: str) -> str:
                if speaker_id in kept_ids:
                    return speaker_id
                source = speaker_lookup.get(speaker_id)
                if source and source.embedding:
                    best = None
                    best_similarity = -1.0
                    for speaker in kept:
                        if not speaker.embedding:
                            continue
                        similarity = _cosine(source.embedding, speaker.embedding)
                        if similarity > best_similarity:
                            best = speaker.speaker_id
                            best_similarity = similarity
                    if best:
                        return best
                return dominant_id or next(iter(kept_ids))

            remapped_segments = [
                segment.model_copy(update={"speaker_id": reassign(segment.speaker_id)})
                for segment in remapped_segments
            ]

    if target_speaker_count is not None and len(kept) > target_speaker_count:
        post_talk = _talk_seconds(remapped_segments)
        kept_sorted = sorted(kept, key=lambda speaker: -post_talk.get(speaker.speaker_id, 0))
        survivors = kept_sorted[:target_speaker_count]
        excess = kept_sorted[target_speaker_count:]
        survivor_ids = {speaker.speaker_id for speaker in survivors}

        cap_remap: dict[str, str] = {}
        for extra in excess:
            best_match: str | None = None
            if extra.embedding:
                best_similarity = -1.0
                for survivor in survivors:
                    if not survivor.embedding:
                        continue
                    similarity = _cosine(extra.embedding, survivor.embedding)
                    if similarity > best_similarity:
                        best_match = survivor.speaker_id
                        best_similarity = similarity
            if not best_match:
                best_match = max(survivor_ids, key=lambda sid: post_talk.get(sid, 0))
            cap_remap[extra.speaker_id] = best_match
            merges.append((best_match, extra.speaker_id, 0.0))

        remapped_segments = [
            segment.model_copy(update={"speaker_id": cap_remap.get(segment.speaker_id, segment.speaker_id)})
            for segment in remapped_segments
        ]
        kept = survivors

    kept_ids = {speaker.speaker_id for speaker in kept}
    orphan_ids = {segment.speaker_id for segment in remapped_segments} - kept_ids
    if orphan_ids and kept_ids:
        post_talk = _talk_seconds(remapped_segments)
        fallback = max(kept_ids, key=lambda speaker_id: post_talk.get(speaker_id, 0))
        remapped_segments = [
            segment.model_copy(update={"speaker_id": fallback})
            if segment.speaker_id in orphan_ids
            else segment
            for segment in remapped_segments
        ]

    return (
        kept,
        remapped_segments,
        MergeReport(
            original_speaker_count=original_count,
            final_speaker_count=len(kept),
            merges=merges,
            dropped_low_talk=dropped_low,
        ),
    )


def _talk_seconds(segments: list[Segment]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for segment in segments:
        totals[segment.speaker_id] += segment.duration_ms / 1000.0
    return dict(totals)
