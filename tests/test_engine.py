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
from investigation_search.schema import EvidenceUnit, SourceType


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
