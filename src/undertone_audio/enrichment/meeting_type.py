from __future__ import annotations

import re

from undertone_audio.schema import MeetingType, Segment

TITLE_PATTERNS: list[tuple[re.Pattern, MeetingType, float]] = [
    (re.compile(r"\bbuild\s*lab\b", re.I), MeetingType.COMMUNITY, 0.95),
    (re.compile(r"\bhappy\s*hour\b|\bHH\b"), MeetingType.COMMUNITY, 0.95),
    (re.compile(r"\bcohort\b|\bcommunity\b", re.I), MeetingType.COMMUNITY, 0.90),
    (re.compile(r"\bhuddle\b", re.I), MeetingType.HUDDLE, 0.95),
    (re.compile(r"\bstand[\s-]?up\b", re.I), MeetingType.HUDDLE, 0.90),
    (re.compile(r"\bworkshop\b", re.I), MeetingType.WORKSHOP, 0.90),
    (re.compile(r"\btraining\b|\bonboarding\b", re.I), MeetingType.TRAINING, 0.90),
    (re.compile(r"\bintro\s*(call)?\b", re.I), MeetingType.DISCOVERY, 0.85),
    (re.compile(r"\bdiscovery\b", re.I), MeetingType.DISCOVERY, 0.90),
    (re.compile(r"\bdemo\b", re.I), MeetingType.DEMO, 0.90),
    (re.compile(r"\bretro\b|\bretrospective\b", re.I), MeetingType.RETRO, 0.90),
    (re.compile(r"\bbrainstorm\b", re.I), MeetingType.BRAINSTORM, 0.90),
    (re.compile(r"\binterview\b", re.I), MeetingType.INTERVIEW, 0.85),
    (re.compile(r"\bpitch\b|\bproposal\b|\bclosing\b", re.I), MeetingType.SALES, 0.85),
    (re.compile(r"\bfireside\b|\bpanel\b|\bbriefing\b", re.I), MeetingType.DEMO, 0.80),
]

KEYWORD_HINTS: dict[MeetingType, list[str]] = {
    MeetingType.DISCOVERY: [
        "tell me about",
        "walk me through",
        "what does",
        "how does",
        "could you explain",
        "what are you looking for",
        "what's the challenge",
    ],
    MeetingType.DEMO: [
        "let me show you",
        "as you can see",
        "this feature",
        "the demo",
        "share my screen",
    ],
    MeetingType.STATUS: [
        "update on",
        "since last",
        "blocked on",
        "this week",
        "next steps",
        "action items",
    ],
    MeetingType.RETRO: [
        "went well",
        "didn't go well",
        "what we learned",
        "next time",
        "post-mortem",
    ],
    MeetingType.BRAINSTORM: ["what if", "could we", "another idea", "wild idea", "throw out"],
    MeetingType.INTERVIEW: ["tell me about a time", "your experience", "your background", "why are you"],
    MeetingType.HUDDLE: [
        "quick sync",
        "let's go around",
        "what's everyone working on",
        "any blockers",
        "before we wrap",
        "good to go",
    ],
    MeetingType.WORKSHOP: [
        "let's build",
        "let's work through",
        "open your",
        "follow along",
        "hands on",
        "breakout",
        "exercise",
    ],
    MeetingType.SALES: [
        "pricing",
        "proposal",
        "contract",
        "next steps to move forward",
        "decision maker",
        "budget",
        "timeline to sign",
    ],
    MeetingType.TRAINING: [
        "today we'll cover",
        "learning objective",
        "module",
        "quiz",
        "take notes",
    ],
}


def classify_from_title(title: str | None) -> tuple[MeetingType, float] | None:
    if not title:
        return None
    for pattern, meeting_type, confidence in TITLE_PATTERNS:
        if pattern.search(title):
            return meeting_type, confidence
    return None


def classify_meeting_type(
    segments: list[Segment],
    *,
    title: str | None = None,
) -> tuple[MeetingType, float]:
    title_result = classify_from_title(title)
    if title_result is not None:
        return title_result

    if not segments:
        return MeetingType.UNKNOWN, 0.0

    speakers = {segment.speaker_id for segment in segments}
    full_text = " ".join(segment.text for segment in segments).lower()
    scores: dict[MeetingType, int] = {
        meeting_type: sum(len(re.findall(re.escape(phrase), full_text)) for phrase in phrases)
        for meeting_type, phrases in KEYWORD_HINTS.items()
    }

    if len(speakers) == 2 and scores.get(MeetingType.INTERVIEW, 0) >= 2:
        return MeetingType.INTERVIEW, 0.7
    if len(speakers) == 2 and max(scores.values(), default=0) < 3:
        return MeetingType.ONE_ON_ONE, 0.5
    if len(speakers) >= 4 and max(scores.values(), default=0) < 3:
        return MeetingType.HUDDLE, 0.4

    best = max(scores, key=scores.get)
    if scores[best] >= 5:
        return best, min(0.4 + 0.05 * scores[best], 0.85)
    return MeetingType.UNKNOWN, 0.0
