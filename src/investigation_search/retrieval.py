from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .schema import EvidenceUnit, ScoredEvidence, Verdict


_WORD_RE = re.compile(r"[a-zA-Z0-9가-힣]+")


@dataclass(frozen=True)
class QueryPass:
    name: str
    query: str


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def build_passes(query: str) -> Sequence[QueryPass]:
    return (
        QueryPass("pass_a_support", query),
        QueryPass("pass_b_contradict", f"{query} 하지만 예외 제한 반대 근거"),
        QueryPass("pass_c_boundary", f"{query} 조건 시점 대상 변경"),
    )


def lexical_score(query: str, text: str) -> float:
    q = set(tokenize(query))
    t = set(tokenize(text))
    if not q or not t:
        return 0.0
    inter = len(q & t)
    return inter / max(len(q), 1)


def classify_verdict(pass_name: str, evidence_text: str) -> Verdict:
    contradiction_hints = ["아니다", "없다", "제한", "예외", "반면", "불가"]
    support_hints = ["가능", "효과", "충족", "성공", "개선"]

    if pass_name == "pass_b_contradict" or any(k in evidence_text for k in contradiction_hints):
        return Verdict.CONTRADICTS
    if any(k in evidence_text for k in support_hints):
        return Verdict.SUPPORTS
    return Verdict.UNCERTAIN


def retrieve(
    query_pass: QueryPass,
    evidence_units: Iterable[EvidenceUnit],
    top_k: int,
) -> List[ScoredEvidence]:
    scored: List[ScoredEvidence] = []
    for unit in evidence_units:
        score = lexical_score(query_pass.query, unit.content)
        if score <= 0:
            continue
        verdict = classify_verdict(query_pass.name, unit.content)
        scored.append(
            ScoredEvidence(
                evidence=unit,
                score=score,
                verdict=verdict,
                why_it_matches=f"{query_pass.name}: 키워드 중첩 {score:.2f}",
            )
        )
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:top_k]
