from __future__ import annotations

from unittest.mock import patch

import sys
import types

if "numpy" not in sys.modules:
    sys.modules["numpy"] = types.SimpleNamespace(
        ndarray=object,
        float32="float32",
        zeros=lambda *args, **kwargs: [],
        asarray=lambda arr, dtype=None: arr,
    )

from investigation_search.engine import InvestigationEngine
from investigation_search.contradiction import ContradictionPrediction
from investigation_search.retrieval import QueryPass, retrieve
from investigation_search.schema import EvidenceUnit, SourceType, Verdict


def _sample_units() -> list[EvidenceUnit]:
    return [
        EvidenceUnit(
            doc_id="doc-1",
            source_type=SourceType.TEXT_SENTENCE,
            content="정책 변경으로 처리 속도 개선 효과가 확인되었다.",
            section_path="A/1",
            char_start=0,
            char_end=25,
            timestamp="2024-01-01T00:00:00Z",
            confidence=0.9,
            metadata={"canonical_entity_id": "entity-alpha"},
        ),
        EvidenceUnit(
            doc_id="doc-2",
            source_type=SourceType.TEXT_SENTENCE,
            content="단, 예외 구간에서는 적용이 불가하고 제한이 있다.",
            section_path="A/2",
            char_start=26,
            char_end=53,
            timestamp="2024-01-01T00:00:00Z",
            confidence=0.8,
            metadata={"canonical_entity_id": None},
        ),
        EvidenceUnit(
            doc_id="doc-3",
            source_type=SourceType.TEXT_SENTENCE,
            content="조건 변경 시 대상 범위가 확대되어 성능이 향상될 수 있다.",
            section_path="A/3",
            char_start=54,
            char_end=86,
            timestamp="2024-01-01T00:00:00Z",
            confidence=0.7,
            metadata={"canonical_entity_id": "entity-alpha"},
        ),
    ]


def test_search_rerank_enabled_adds_diagnostics_and_stage_scores() -> None:
    engine = InvestigationEngine(_sample_units())

    result = engine.search("처리 속도 개선", top_k_per_pass=2, time_budget_sec=120)

    assert result.reranker_model is not None
    assert result.reranker_version is not None
    assert result.diagnostics["rerank"]["reason"] == "applied"
    assert result.diagnostics["rerank"]["skipped"] is False
    assert all("first_pass" in item.stage_scores for item in result.evidence)
    assert all("rerank" in item.stage_scores for item in result.evidence)
    assert all("final" in item.stage_scores for item in result.evidence)

    scores = [item.score for item in result.evidence]
    assert scores == sorted(scores, reverse=True)


def test_search_rerank_skipped_when_time_budget_exceeded_keeps_sorted_scores() -> None:
    engine = InvestigationEngine(_sample_units())

    fake_times = [0.0, 0.0, 0.0, 0.0, 119.995, 120.0]
    with patch("investigation_search.engine.time.time", side_effect=fake_times):
        result = engine.search("처리 속도 개선", top_k_per_pass=2, time_budget_sec=120)

    assert result.diagnostics["rerank"]["reason"] == "time_budget_exceeded"
    assert result.diagnostics["rerank"]["skipped"] is True
    assert all("first_pass" in item.stage_scores for item in result.evidence)
    assert all("rerank" not in item.stage_scores for item in result.evidence)

    scores = [item.score for item in result.evidence]
    assert scores == sorted(scores, reverse=True)


def test_query_expansion_penalty_and_entity_grouping() -> None:
    units = _sample_units()
    hits = retrieve(QueryPass(name="pass_a_support", query="처리 속도개선"), units, top_k=3)

    assert hits
    assert any("alias_penalty" in item.stage_scores for item in hits)

    engine = InvestigationEngine(units)
    result = engine.search("처리 속도개선", top_k_per_pass=5, time_budget_sec=120)
    groups = result.diagnostics["entity_groups"]

    assert "entity-alpha" in groups
    assert "null" in groups
    assert groups["entity-alpha"]["count"] >= 1


def test_search_cache_hit_diagnostics() -> None:
    engine = InvestigationEngine(_sample_units(), cache_ttl_sec=100.0, cache_size=16)
    first = engine.search("처리 속도 개선", top_k_per_pass=2, time_budget_sec=120)
    second = engine.search("처리 속도 개선", top_k_per_pass=2, time_budget_sec=120)

    assert first.diagnostics["cache"]["status"] == "miss"
    assert second.diagnostics["cache"]["status"] == "hit"


def test_contradiction_detector_can_override_verdict() -> None:
    class StubDetector:
        model_name = "stub-contradiction"
        model_version = "test"

        def predict(self, query, candidates):
            return [
                ContradictionPrediction(
                    verdict=Verdict.CONTRADICTS,
                    score=0.99,
                    rationale="force_contradiction",
                )
                for _ in candidates
            ]

    engine = InvestigationEngine(_sample_units(), contradiction_detector=StubDetector())
    result = engine.search("처리 속도 개선", top_k_per_pass=2, time_budget_sec=120)

    assert result.evidence
    assert all(item.verdict == Verdict.CONTRADICTS for item in result.evidence)
    assert result.diagnostics["contradiction_detector"]["overrides"] >= 1


def test_search_results_always_include_sources() -> None:
    engine = InvestigationEngine(_sample_units())
    result = engine.search("처리 속도 개선", top_k_per_pass=2, time_budget_sec=120)

    assert result.sources
    assert result.answer_sources
    assert all(item.source is not None for item in result.evidence)
    assert all(src.citation_id for src in result.sources)


def test_online_learning_progresses_with_searches() -> None:
    engine = InvestigationEngine(_sample_units(), enable_cache=False, online_learning=True)
    first = engine.search("처리 속도 개선", top_k_per_pass=2, time_budget_sec=120)
    second = engine.search("조건 변경 성능 향상", top_k_per_pass=2, time_budget_sec=120)

    assert first.diagnostics["online_learning"]["version"] >= 1
    assert second.diagnostics["online_learning"]["version"] >= first.diagnostics["online_learning"]["version"]
    assert second.diagnostics["online_learning"]["updates"] >= first.diagnostics["online_learning"]["updates"]
