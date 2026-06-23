from __future__ import annotations

import re
from collections import defaultdict

from undertone_audio.schema import Segment

FILLER_PATTERNS = [
    r"\bum+\b",
    r"\buh+\b",
    r"\berm+\b",
    r"\blike\b",
    r"\byou know\b",
    r"\bi mean\b",
    r"\bsort of\b",
    r"\bkind of\b",
    r"\bbasically\b",
    r"\bactually\b",
    r"\bliterally\b",
    r"\bright\?",
]
_FILLER_RE = re.compile("|".join(FILLER_PATTERNS), re.IGNORECASE)


def annotate_fillers(segments: list[Segment]) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for segment in segments:
        matches = [match.group(0).lower() for match in _FILLER_RE.finditer(segment.text)]
        segment.enrichment.fillers = matches
        totals[segment.speaker_id] += len(matches)
    return dict(totals)
