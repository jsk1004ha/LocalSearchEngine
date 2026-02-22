from __future__ import annotations

from typing import Any

from .schema import ScoredEvidence, SearchResult, SourceCitation


def result_to_payload(
    result: SearchResult,
    *,
    query: str,
    mode: str,
    top_k_per_pass: int,
    time_budget_sec: int,
    max_items: int = 8,
    include_diagnostics: bool = True,
) -> dict[str, Any]:
    limit = max(0, int(max_items))
    payload: dict[str, Any] = {
        "query": query,
        "mode": mode,
        "top_k_per_pass": int(top_k_per_pass),
        "time_budget_sec": int(time_budget_sec),
        "answer": result.answer,
        "answer_sources": [_serialize_source(src) for src in result.answer_sources],
        "source_count": len(result.sources),
        "sources": [_serialize_source(src) for src in result.sources],
        "evidence_total": len(result.evidence),
        "evidence": [_serialize_scored(item) for item in result.evidence[:limit]],
        "contradictions_total": len(result.contradictions),
        "contradictions": [_serialize_scored(item) for item in result.contradictions[:limit]],
        "reranker_model": result.reranker_model,
        "reranker_version": result.reranker_version,
        "build_id": result.build_id,
    }
    if include_diagnostics:
        payload["diagnostics"] = dict(result.diagnostics)
    return payload


def _serialize_source(src: SourceCitation) -> dict[str, Any]:
    return {
        "citation_id": src.citation_id,
        "doc_id": src.doc_id,
        "section_path": src.section_path,
        "char_start": src.char_start,
        "char_end": src.char_end,
        "source_type": src.source_type.value,
        "timestamp": src.timestamp,
    }


def _serialize_scored(item: ScoredEvidence) -> dict[str, Any]:
    source = item.source
    source_payload = _serialize_source(source) if source is not None else None
    evidence = item.evidence
    return {
        "score": float(item.score),
        "verdict": item.verdict.value,
        "why_it_matches": item.why_it_matches,
        "stage_scores": dict(item.stage_scores),
        "source": source_payload,
        "evidence": {
            "doc_id": evidence.doc_id,
            "source_type": evidence.source_type.value,
            "content": evidence.content,
            "section_path": evidence.section_path,
            "char_start": evidence.char_start,
            "char_end": evidence.char_end,
            "timestamp": evidence.timestamp,
            "confidence": evidence.confidence,
            "metadata": dict(evidence.metadata),
        },
    }
