from __future__ import annotations

from dataclasses import dataclass, field
from math import log2
from typing import Dict, Iterable, List, Sequence

from .schema import SearchResult


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    relevant_citation_ids: Sequence[str] = field(default_factory=tuple)
    relevant_doc_ids: Sequence[str] = field(default_factory=tuple)
    expect_contradiction: bool | None = None
    top_k: int = 5
    time_budget_sec: int = 120


@dataclass(frozen=True)
class EvaluationReport:
    metrics: Dict[str, float]
    per_case: List[Dict[str, object]]


def evaluate_engine(engine, cases: Sequence[EvaluationCase] | Iterable[EvaluationCase]) -> EvaluationReport:
    rows: list[dict[str, object]] = []
    recall_values: list[float] = []
    mrr_values: list[float] = []
    ndcg_values: list[float] = []
    contradiction_values: list[float] = []

    for case in cases:
        result = engine.search(case.query, top_k_per_pass=case.top_k, time_budget_sec=case.time_budget_sec)
        row = _evaluate_case(result, case)
        rows.append(row)
        recall_values.append(float(row["recall_at_k"]))
        mrr_values.append(float(row["mrr"]))
        ndcg_values.append(float(row["ndcg_at_k"]))
        if case.expect_contradiction is not None:
            contradiction_values.append(float(row["contradiction_hit"]))

    metrics = {
        "query_count": float(len(rows)),
        "recall_at_k": _mean(recall_values),
        "mrr": _mean(mrr_values),
        "ndcg_at_k": _mean(ndcg_values),
        "contradiction_hit_rate": _mean(contradiction_values),
    }
    return EvaluationReport(metrics=metrics, per_case=rows)


def compare_reports(a: EvaluationReport, b: EvaluationReport) -> Dict[str, float]:
    keys = {"recall_at_k", "mrr", "ndcg_at_k", "contradiction_hit_rate"}
    out: dict[str, float] = {}
    for key in sorted(keys):
        out[key] = float(b.metrics.get(key, 0.0)) - float(a.metrics.get(key, 0.0))
    return out


def _evaluate_case(result: SearchResult, case: EvaluationCase) -> Dict[str, object]:
    evidence = list(result.evidence[: max(case.top_k, 1)])
    relevance = [1 if _is_relevant(item, case) else 0 for item in evidence]

    recall = 1.0 if any(relevance) else 0.0
    mrr = 0.0
    for idx, rel in enumerate(relevance, start=1):
        if rel > 0:
            mrr = 1.0 / idx
            break

    dcg = 0.0
    for idx, rel in enumerate(relevance, start=1):
        dcg += rel / log2(idx + 1)

    max_rel = sum(relevance)
    idcg = 0.0
    for idx in range(1, max_rel + 1):
        idcg += 1.0 / log2(idx + 1)
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    contradiction_hit = 0.0
    if case.expect_contradiction is True:
        contradiction_hit = 1.0 if result.contradictions else 0.0
    elif case.expect_contradiction is False:
        contradiction_hit = 1.0 if not result.contradictions else 0.0

    return {
        "query": case.query,
        "top_k": case.top_k,
        "recall_at_k": recall,
        "mrr": mrr,
        "ndcg_at_k": ndcg,
        "contradiction_hit": contradiction_hit,
        "result_sources": [src.citation_id for src in result.sources],
        "answer_sources": [src.citation_id for src in result.answer_sources],
    }


def _is_relevant(item, case: EvaluationCase) -> bool:
    source = item.source
    if source is None:
        return False
    if case.relevant_citation_ids and source.citation_id in case.relevant_citation_ids:
        return True
    if case.relevant_doc_ids and source.doc_id in case.relevant_doc_ids:
        return True
    return False


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))
