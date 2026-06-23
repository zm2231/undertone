from __future__ import annotations

import re

from undertone_audio.schema import LinguisticFeatures, Segment

LEXICONS: dict[str, set[str]] = {
    "cognitive_process": {
        "think",
        "thought",
        "thinking",
        "know",
        "knew",
        "knowing",
        "knowledge",
        "understand",
        "understood",
        "understanding",
        "realize",
        "realized",
        "consider",
        "considered",
        "remember",
        "recall",
        "decide",
        "decided",
        "reason",
        "reasoning",
        "analyze",
        "analyzed",
    },
    "tentative": {
        "maybe",
        "perhaps",
        "possibly",
        "might",
        "could",
        "may",
        "seems",
        "seem",
        "seemed",
        "appears",
        "appear",
        "guess",
        "suppose",
        "probably",
        "somewhat",
        "approximately",
        "around",
        "roughly",
    },
    "certainty": {
        "always",
        "never",
        "definitely",
        "certainly",
        "must",
        "absolutely",
        "clearly",
        "obviously",
        "undoubtedly",
        "completely",
        "totally",
        "entirely",
        "fully",
        "exactly",
        "precisely",
    },
    "inclusive": {
        "and",
        "with",
        "also",
        "together",
        "both",
        "include",
        "including",
        "plus",
        "additionally",
        "furthermore",
        "moreover",
    },
    "exclusive": {
        "but",
        "except",
        "without",
        "however",
        "though",
        "although",
        "unless",
        "rather",
        "instead",
        "neither",
        "nor",
        "exclude",
        "excluding",
    },
    "insight": {
        "realize",
        "realized",
        "understand",
        "understood",
        "see",
        "notice",
        "noticed",
        "discover",
        "discovered",
        "recognize",
        "recognized",
        "aware",
        "awareness",
    },
    "causation": {
        "because",
        "cause",
        "caused",
        "causes",
        "why",
        "since",
        "therefore",
        "thus",
        "hence",
        "consequently",
        "result",
        "resulted",
        "leads",
        "led",
        "due",
        "reason",
        "effect",
    },
}

_TOKEN_RE = re.compile(r"\b[\w']+\b")


def extract_linguistic_features(text: str) -> LinguisticFeatures:
    tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
    counts = {
        category: sum(1 for token in tokens if token in lexicon)
        for category, lexicon in LEXICONS.items()
    }
    return LinguisticFeatures(word_count=len(tokens), **counts)


def annotate_linguistic(segments: list[Segment]) -> None:
    for segment in segments:
        segment.enrichment.linguistic = extract_linguistic_features(segment.text)
