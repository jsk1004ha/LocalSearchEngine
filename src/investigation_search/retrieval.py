from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

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
        QueryPass("pass_b_contradict", f"{query} 하지만 예외 제한 반대 근거 실패 불가"),
        QueryPass("pass_c_boundary", f"{query} 조건 시점 대상 변경 장기 단기"),
    )


def _idf_map(evidence_units: Iterable[EvidenceUnit]) -> Dict[str, float]:
    docs = [set(tokenize(unit.content)) for unit in evidence_units]
    n = len(docs)
    df: Counter[str] = Counter()
    for d in docs:
        df.update(d)
    return {tok: math.log((n + 1) / (freq + 1)) + 1.0 for tok, freq in df.items()}


def lexical_score(query: str, text: str) -> float:
    q = set(tokenize(query))
    t = set(tokenize(text))
    if not q or not t:
        return 0.0
    inter = len(q & t)
    return inter / max(len(q), 1)


def dense_like_score(query: str, text: str, idf: Dict[str, float]) -> float:
    q_tokens = tokenize(query)
    t_tokens = tokenize(text)
    if not q_tokens or not t_tokens:
        return 0.0

    q_tf = Counter(q_tokens)
    t_tf = Counter(t_tokens)

    q_norm = math.sqrt(sum((q_tf[tok] * idf.get(tok, 1.0)) ** 2 for tok in q_tf))
    t_norm = math.sqrt(sum((t_tf[tok] * idf.get(tok, 1.0)) ** 2 for tok in t_tf))
    if q_norm == 0 or t_norm == 0:
        return 0.0

    dot = 0.0
    for tok, qv in q_tf.items():
        dot += (qv * idf.get(tok, 1.0)) * (t_tf.get(tok, 0) * idf.get(tok, 1.0))
    return dot / (q_norm * t_norm)


def rerank_score(query: str, evidence: EvidenceUnit, base_score: float) -> float:
    # Evidence confidence and shorter focused spans are preferred in reranking.
    span_len = max(evidence.char_end - evidence.char_start, 1)
    span_bonus = 1.0 / (1.0 + math.log1p(span_len))
    confidence_bonus = min(max(evidence.confidence, 0.0), 1.0)
    recency_bonus = 0.05 if evidence.timestamp >= "2025" else 0.0
    return (base_score * 0.65) + (confidence_bonus * 0.25) + (span_bonus * 0.08) + recency_bonus


def classify_verdict(pass_name: str, evidence_text: str) -> Verdict:
    contradiction_hints = ["아니다", "없다", "제한", "예외", "반면", "불가", "실패", "위험"]
    support_hints = ["가능", "효과", "충족", "성공", "개선", "증가", "향상"]

    if pass_name == "pass_b_contradict" or any(k in evidence_text for k in contradiction_hints):
        return Verdict.CONTRADICTS
    if any(k in evidence_text for k in support_hints):
        return Verdict.SUPPORTS
    return Verdict.UNCERTAIN


def retrieve(
    query_pass: QueryPass,
    evidence_units: Iterable[EvidenceUnit],
    top_k: int,
    lexical_weight: float,
    dense_weight: float,
    rerank_weight: float,
) -> List[ScoredEvidence]:
    units = list(evidence_units)
    idf = _idf_map(units)
    scored: List[ScoredEvidence] = []
    for unit in units:
        l_score = lexical_score(query_pass.query, unit.content)
        d_score = dense_like_score(query_pass.query, unit.content, idf)
        if l_score <= 0 and d_score <= 0:
            continue

        base = (l_score * lexical_weight) + (d_score * dense_weight)
        r_score = rerank_score(query_pass.query, unit, base)
        final = base + (r_score * rerank_weight)
        verdict = classify_verdict(query_pass.name, unit.content)

        scored.append(
            ScoredEvidence(
                evidence=unit,
                score=final,
                lexical_score=l_score,
                dense_score=d_score,
                rerank_score=r_score,
                verdict=verdict,
                why_it_matches=(
                    f"{query_pass.name}: lexical={l_score:.3f}, dense={d_score:.3f}, rerank={r_score:.3f}"
                ),
            )
        )

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:top_k]
