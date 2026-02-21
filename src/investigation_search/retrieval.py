from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .embedding import DEFAULT_EMBEDDING_MODEL, encode_texts
from .index_ann import ANNIndex, search_index
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


def _span_key(unit: EvidenceUnit) -> tuple[str, int, int]:
    return (unit.doc_id, unit.char_start, unit.char_end)


def retrieve(
    query_pass: QueryPass,
    evidence_units: Iterable[EvidenceUnit],
    top_k: int,
    *,
    ann_index: ANNIndex | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> List[ScoredEvidence]:
    units = list(evidence_units)
    lexical_hits: List[ScoredEvidence] = []
    for unit in units:
        score = lexical_score(query_pass.query, unit.content)
        if score <= 0:
            continue
        verdict = classify_verdict(query_pass.name, unit.content)
        lexical_hits.append(
            ScoredEvidence(
                evidence=unit,
                score=score,
                verdict=verdict,
                why_it_matches=f"{query_pass.name}: lexical 키워드 중첩 {score:.2f}",
            )
        )

    lexical_hits.sort(key=lambda s: s.score, reverse=True)
    candidate_pool = lexical_hits[: top_k * 3]

    dense_hits: List[ScoredEvidence] = []
    if ann_index is not None:
        query_vec = encode_texts([query_pass.query], model_name=embedding_model, text_type="query")
        for doc_ix, dense_score in search_index(ann_index, query_vec, top_k=top_k * 3)[0]:
            if doc_ix < 0 or doc_ix >= len(units):
                continue
            unit = units[doc_ix]
            dense_hits.append(
                ScoredEvidence(
                    evidence=unit,
                    score=dense_score,
                    verdict=classify_verdict(query_pass.name, unit.content),
                    why_it_matches=f"{query_pass.name}: dense 유사도 {dense_score:.2f}",
                )
            )

    merged: dict[tuple[str, int, int], ScoredEvidence] = {}
    for hit in [*candidate_pool, *dense_hits]:
        key = _span_key(hit.evidence)
        prev = merged.get(key)
        if prev is None:
            merged[key] = hit
            continue
        if hit.score > prev.score:
            merged[key] = ScoredEvidence(
                evidence=hit.evidence,
                score=hit.score,
                verdict=hit.verdict,
                why_it_matches=f"{prev.why_it_matches} + {hit.why_it_matches}",
            )
        else:
            merged[key] = ScoredEvidence(
                evidence=prev.evidence,
                score=prev.score,
                verdict=prev.verdict,
                why_it_matches=f"{prev.why_it_matches} + {hit.why_it_matches}",
            )

    scored = sorted(merged.values(), key=lambda s: s.score, reverse=True)
    return scored[:top_k]
