from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

from .bm25 import BM25Index, build_bm25_index, search_bm25
from .embedding import DEFAULT_EMBEDDING_MODEL, encode_texts
from .entity import expansion_factor, generate_alias_candidates
from .index_ann import ANNIndex, search_index
from .schema import EvidenceUnit, ScoredEvidence, Verdict, build_source_citation


_WORD_RE = re.compile(r"[a-zA-Z0-9가-힣]+")


@dataclass(frozen=True)
class QueryPass:
    name: str
    query: str


@dataclass(frozen=True)
class ExpandedQuery:
    query: str
    penalty: float = 0.0
    reason: str = "original"


@dataclass(frozen=True)
class RetrievalWeights:
    bm25: float = 0.42
    dense: float = 0.33
    lexical: float = 0.15
    rrf: float = 0.10
    rrf_k: int = 60


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def build_passes(query: str) -> Sequence[QueryPass]:
    return (
        QueryPass("pass_a_support", query),
        QueryPass("pass_b_contradict", f"{query} 하지만 예외 제한 반대 근거"),
        QueryPass("pass_c_boundary", f"{query} 조건 시점 대상 변경"),
    )


def expand_query_aliases(query: str, max_aliases: int = 5) -> List[ExpandedQuery]:
    expansions: List[ExpandedQuery] = [ExpandedQuery(query=query)]
    for candidate in generate_alias_candidates(query)[:max_aliases]:
        alias = candidate.alias
        if alias == query:
            continue
        factor = expansion_factor(query, alias)
        adjusted_penalty = min(candidate.penalty + (1 - factor) * 0.15, 0.65)
        expansions.append(
            ExpandedQuery(
                query=alias,
                penalty=adjusted_penalty,
                reason=f"alias:{candidate.reason}",
            )
        )
    return expansions


def lexical_score(
    query: str,
    text: str,
    *,
    token_boosts: Mapping[str, float] | None = None,
) -> float:
    q = set(tokenize(query))
    t = set(tokenize(text))
    if not q or not t:
        return 0.0
    inter_tokens = q & t
    if not inter_tokens:
        return 0.0
    if token_boosts is None:
        return len(inter_tokens) / max(len(q), 1)

    def _term_weight(term: str) -> float:
        return 1.0 + max(0.0, min(float(token_boosts.get(term, 0.0)), 2.0))

    inter_weight = sum(_term_weight(term) for term in inter_tokens)
    query_weight = sum(_term_weight(term) for term in q)
    return inter_weight / max(query_weight, 1e-9)


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


def build_bm25_for_units(evidence_units: Iterable[EvidenceUnit]) -> BM25Index:
    units = list(evidence_units)
    return build_bm25_index([u.content for u in units])


def _normalize_scores(raw_scores: Dict[tuple[str, int, int], float]) -> Dict[tuple[str, int, int], float]:
    if not raw_scores:
        return {}
    vals = list(raw_scores.values())
    lo = min(vals)
    hi = max(vals)
    if hi == lo:
        base = 1.0 if hi > 0 else 0.0
        return {key: base for key in raw_scores}
    return {key: (score - lo) / (hi - lo) for key, score in raw_scores.items()}


def _rank_map(raw_scores: Dict[tuple[str, int, int], float]) -> Dict[tuple[str, int, int], int]:
    ranked = sorted(raw_scores.items(), key=lambda item: item[1], reverse=True)
    return {key: idx + 1 for idx, (key, _) in enumerate(ranked)}


def _top_keys(
    raw_scores: Dict[tuple[str, int, int], float],
    *,
    limit: int,
) -> set[tuple[str, int, int]]:
    if limit <= 0:
        return set()
    ranked = sorted(raw_scores.items(), key=lambda item: item[1], reverse=True)
    return {key for key, _ in ranked[:limit]}


def retrieve(
    query_pass: QueryPass,
    evidence_units: Iterable[EvidenceUnit],
    top_k: int,
    *,
    bm25_index: BM25Index | None = None,
    ann_index: ANNIndex | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    weights: RetrievalWeights | None = None,
    token_boosts: Mapping[str, float] | None = None,
) -> List[ScoredEvidence]:
    units = list(evidence_units)
    if not units or top_k <= 0:
        return []

    unit_by_key = {_span_key(unit): unit for unit in units}
    weights = weights or RetrievalWeights()
    expansions = expand_query_aliases(query_pass.query)

    lexical_scores: dict[tuple[str, int, int], float] = {}
    lexical_penalty: dict[tuple[str, int, int], float] = {}
    for expanded in expansions:
        for unit in units:
            base_score = lexical_score(expanded.query, unit.content, token_boosts=token_boosts)
            if base_score <= 0:
                continue
            final_score = base_score * (1 - expanded.penalty)
            key = _span_key(unit)
            if final_score > lexical_scores.get(key, -1.0):
                lexical_scores[key] = final_score
                lexical_penalty[key] = expanded.penalty

    if bm25_index is None:
        bm25_index = build_bm25_index([u.content for u in units])

    bm25_scores: dict[tuple[str, int, int], float] = {}
    bm25_penalty: dict[tuple[str, int, int], float] = {}
    bm25_limit = max(top_k * 6, 12)
    for expanded in expansions:
        bm25_rows = search_bm25(
            bm25_index,
            expanded.query,
            top_k=bm25_limit,
            token_boosts=token_boosts,
        )
        for doc_ix, raw_score in bm25_rows:
            if doc_ix < 0 or doc_ix >= len(units):
                continue
            adjusted = raw_score * (1 - expanded.penalty)
            key = _span_key(units[doc_ix])
            if adjusted > bm25_scores.get(key, -1.0):
                bm25_scores[key] = adjusted
                bm25_penalty[key] = expanded.penalty

    dense_scores: dict[tuple[str, int, int], float] = {}
    if ann_index is not None:
        query_vec = encode_texts([query_pass.query], model_name=embedding_model, text_type="query")
        for doc_ix, dense_score in search_index(ann_index, query_vec, top_k=top_k * 6)[0]:
            if doc_ix < 0 or doc_ix >= len(units):
                continue
            unit = units[doc_ix]
            dense_scores[_span_key(unit)] = float(dense_score)

    candidate_limit = max(top_k * 8, 20)
    candidate_keys = (
        _top_keys(bm25_scores, limit=candidate_limit)
        | _top_keys(dense_scores, limit=candidate_limit)
        | _top_keys(lexical_scores, limit=candidate_limit)
    )
    if not candidate_keys:
        return []

    bm25_norm = _normalize_scores(bm25_scores)
    dense_norm = _normalize_scores(dense_scores)
    lexical_norm = _normalize_scores(lexical_scores)
    bm25_rank = _rank_map(bm25_scores)
    dense_rank = _rank_map(dense_scores)
    lexical_rank = _rank_map(lexical_scores)

    scored: List[ScoredEvidence] = []
    for key in candidate_keys:
        unit = unit_by_key[key]
        rrf_score = 0.0
        if key in bm25_rank:
            rrf_score += 1.0 / (weights.rrf_k + bm25_rank[key])
        if key in dense_rank:
            rrf_score += 1.0 / (weights.rrf_k + dense_rank[key])
        if key in lexical_rank:
            rrf_score += 1.0 / (weights.rrf_k + lexical_rank[key])

        final_score = (
            weights.bm25 * bm25_norm.get(key, 0.0)
            + weights.dense * dense_norm.get(key, 0.0)
            + weights.lexical * lexical_norm.get(key, 0.0)
            + weights.rrf * rrf_score
        )

        why_parts = [f"{query_pass.name}: hybrid"]
        if key in bm25_scores:
            why_parts.append(f"bm25={bm25_scores[key]:.3f}")
        if key in dense_scores:
            why_parts.append(f"dense={dense_scores[key]:.3f}")
        if key in lexical_scores:
            why_parts.append(f"lexical={lexical_scores[key]:.3f}")

        stage_scores = {
            "bm25": bm25_scores.get(key, 0.0),
            "dense": dense_scores.get(key, 0.0),
            "lexical": lexical_scores.get(key, 0.0),
            "rrf": rrf_score,
            "hybrid": final_score,
        }
        if key in bm25_penalty:
            stage_scores["alias_penalty"] = bm25_penalty[key]
        elif key in lexical_penalty:
            stage_scores["alias_penalty"] = lexical_penalty[key]

        scored.append(
            ScoredEvidence(
                evidence=unit,
                score=final_score,
                verdict=classify_verdict(query_pass.name, unit.content),
                why_it_matches=", ".join(why_parts),
                stage_scores=stage_scores,
                source=build_source_citation(unit),
            )
        )

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:top_k]
