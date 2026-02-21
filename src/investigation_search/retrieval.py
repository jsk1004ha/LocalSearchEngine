from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Mapping, Sequence

from .analyzer import detect_language, unique_tokens
from .bm25 import BM25Index, build_bm25_index, search_bm25
from .embedding import DEFAULT_EMBEDDING_MODEL, encode_texts
from .entity import expansion_factor, generate_alias_candidates
from .index_ann import ANNIndex, search_index
from .schema import EvidenceUnit, ScoredEvidence, SourceType, Verdict, build_source_citation


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


@dataclass(frozen=True)
class RetrievalOptions:
    use_morphology: bool = False
    include_char_ngrams: bool = True
    min_ocr_confidence: float = 0.35
    short_table_cell_len: int = 4
    short_table_penalty: float = 0.88
    enable_recency_boost: bool = False
    recency_half_life_days: float = 30.0
    max_recency_boost: float = 0.12
    now_utc: datetime | None = None


def tokenize(text: str, *, use_morphology: bool = False, include_char_ngrams: bool = True) -> List[str]:
    return sorted(
        unique_tokens(
            text,
            mode="search",
            use_morphology=use_morphology,
            include_char_ngrams=include_char_ngrams,
        )
    )


def build_passes(query: str) -> Sequence[QueryPass]:
    lang = detect_language(query)
    if lang == "en":
        contradict_hint = "however exception limitation counter evidence unless not"
        boundary_hint = "condition timeline scope boundary edge case only if"
    elif lang == "mixed":
        contradict_hint = "하지만 예외 제한 반대 근거 however exception limitation"
        boundary_hint = "조건 시점 대상 변경 condition timeline scope"
    else:
        contradict_hint = "하지만 예외 제한 반대 근거 단 only if unless limitation"
        boundary_hint = "조건 시점 대상 변경 경계 사례 edge case boundary"

    return (
        QueryPass("pass_a_support", query),
        QueryPass("pass_b_contradict", f"{query} {contradict_hint}"),
        QueryPass("pass_c_boundary", f"{query} {boundary_hint}"),
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
    options: RetrievalOptions | None = None,
) -> float:
    options = options or RetrievalOptions()
    q = set(
        tokenize(
            query,
            use_morphology=options.use_morphology,
            include_char_ngrams=options.include_char_ngrams,
        )
    )
    t = set(
        tokenize(
            text,
            use_morphology=options.use_morphology,
            include_char_ngrams=options.include_char_ngrams,
        )
    )
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
    options: RetrievalOptions | None = None,
    dense_top_k_factor: int = 6,
    candidate_limit_factor: int = 8,
) -> List[ScoredEvidence]:
    units = list(evidence_units)
    if not units or top_k <= 0:
        return []

    unit_by_key = {_span_key(unit): unit for unit in units}
    weights = weights or RetrievalWeights()
    options = options or RetrievalOptions()
    expansions = expand_query_aliases(query_pass.query)

    lexical_scores: dict[tuple[str, int, int], float] = {}
    lexical_penalty: dict[tuple[str, int, int], float] = {}
    for expanded in expansions:
        for unit in units:
            base_score = lexical_score(
                expanded.query,
                unit.content,
                token_boosts=token_boosts,
                options=options,
            )
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
    bm25_limit = max(top_k * max(dense_top_k_factor, 1), 12)
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
        for doc_ix, dense_score in search_index(ann_index, query_vec, top_k=top_k * max(dense_top_k_factor, 1))[0]:
            if doc_ix < 0 or doc_ix >= len(units):
                continue
            unit = units[doc_ix]
            dense_scores[_span_key(unit)] = float(dense_score)

    candidate_limit = max(top_k * max(candidate_limit_factor, 1), 20)
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
        quality_prior = _quality_prior(unit, options=options)
        final_score *= quality_prior
        if final_score <= 0:
            continue

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
            "quality_prior": quality_prior,
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


def _quality_prior(unit: EvidenceUnit, *, options: RetrievalOptions) -> float:
    confidence = max(0.0, min(float(unit.confidence), 1.0))
    prior = confidence**0.7

    if unit.source_type == SourceType.OCR_TEXT:
        if confidence < options.min_ocr_confidence:
            return 0.0
        prior *= 0.78
    elif unit.source_type == SourceType.TABLE_CELL:
        prior *= 0.94
        if len(unit.content.strip()) < options.short_table_cell_len:
            prior *= options.short_table_penalty
    elif unit.source_type == SourceType.CAPTION:
        prior *= 0.97
    else:
        prior *= 1.0

    if options.enable_recency_boost:
        age_days = _age_days(unit.timestamp, now_utc=options.now_utc)
        if age_days is not None and age_days >= 0:
            half_life = max(options.recency_half_life_days, 1.0)
            decay = 0.5 ** (age_days / half_life)
            prior *= 1.0 + min(options.max_recency_boost, max(0.0, decay * options.max_recency_boost))

    return max(0.0, min(prior, 1.5))


def _age_days(timestamp: str, *, now_utc: datetime | None = None) -> float | None:
    try:
        normalized = timestamp.replace("Z", "+00:00")
        ts = datetime.fromisoformat(normalized)
    except Exception:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    now = now_utc or datetime.now(timezone.utc)
    delta = now - ts.astimezone(timezone.utc)
    return delta.total_seconds() / 86400.0
