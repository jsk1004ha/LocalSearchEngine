from __future__ import annotations

import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"[a-zA-Z0-9가-힣]+")
_NEGATION_HINTS = (
    "아니다",
    "없",
    "불가",
    "불가능",
    "제한",
    "예외",
    "금지",
    "못",
    "중단",
)


@dataclass(frozen=True)
class PairInferenceScore:
    entail: float
    contradict: float
    neutral: float


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def infer_claim_evidence_relation(claim: str, evidence: str) -> PairInferenceScore:
    claim_tokens = set(_tokenize(claim))
    evidence_tokens = set(_tokenize(evidence))

    if not claim_tokens or not evidence_tokens:
        return PairInferenceScore(entail=0.0, contradict=0.0, neutral=1.0)

    overlap = len(claim_tokens & evidence_tokens) / max(len(claim_tokens), 1)
    evidence_has_negation = any(hint in evidence for hint in _NEGATION_HINTS)
    claim_has_negation = any(hint in claim for hint in _NEGATION_HINTS)

    entail = min(1.0, overlap * (1.1 if not evidence_has_negation else 0.6))
    contradict = min(
        1.0,
        overlap * (1.1 if evidence_has_negation != claim_has_negation else 0.3),
    )

    neutral = max(0.0, 1.0 - max(entail, contradict))
    total = entail + contradict + neutral
    if total <= 0:
        return PairInferenceScore(entail=0.0, contradict=0.0, neutral=1.0)

    return PairInferenceScore(
        entail=entail / total,
        contradict=contradict / total,
        neutral=neutral / total,
    )
