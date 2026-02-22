from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from unittest.mock import patch

from investigation_search.cli import main
from investigation_search.schema import EvidenceUnit, ScoredEvidence, SearchResult, SourceType, Verdict, build_source_citation


def _sample_result() -> SearchResult:
    unit = EvidenceUnit(
        doc_id="https://example.com/a",
        source_type=SourceType.WEB_SNIPPET,
        content="테스트 근거 문장",
        section_path="example",
        char_start=0,
        char_end=8,
        timestamp="2026-02-22T00:00:00Z",
        confidence=0.8,
        metadata={"url": "https://example.com/a"},
    )
    source = build_source_citation(unit)
    evidence = ScoredEvidence(
        evidence=unit,
        score=0.91,
        verdict=Verdict.SUPPORTS,
        why_it_matches="keyword overlap",
        stage_scores={"hybrid": 0.91},
        source=source,
    )
    return SearchResult(
        answer="테스트 답변",
        evidence=[evidence],
        contradictions=[],
        diagnostics={"cache": {"status": "miss"}, "source_count": 1},
        answer_sources=[source],
        sources=[source],
        reranker_model="stub",
        reranker_version="test",
        build_id="build-test",
    )


def test_cli_search_json_output() -> None:
    class StubEngine:
        def search(self, query, top_k_per_pass, time_budget_sec, mode=None):
            assert query == "테스트 쿼리"
            assert top_k_per_pass == 5
            assert time_budget_sec == 120
            assert mode == "investigation"
            return _sample_result()

    out = io.StringIO()
    with patch("investigation_search.cli._load_engine_for_runtime", return_value=StubEngine()):
        with redirect_stdout(out):
            rc = main(["search", "테스트 쿼리", "--json"])

    assert rc == 0
    payload = json.loads(out.getvalue())
    assert payload["query"] == "테스트 쿼리"
    assert payload["answer"] == "테스트 답변"
    assert payload["evidence_total"] == 1
    assert "diagnostics" in payload


def test_cli_search_json_without_diagnostics() -> None:
    class StubEngine:
        def search(self, query, top_k_per_pass, time_budget_sec, mode=None):
            return _sample_result()

    out = io.StringIO()
    with patch("investigation_search.cli._load_engine_for_runtime", return_value=StubEngine()):
        with redirect_stdout(out):
            rc = main(["search", "테스트 쿼리", "--json", "--no-diagnostics"])

    assert rc == 0
    payload = json.loads(out.getvalue())
    assert "diagnostics" not in payload


def test_cli_web_command_calls_webapp_runner() -> None:
    stub_engine = object()
    with patch("investigation_search.cli._load_engine_for_runtime", return_value=stub_engine):
        with patch("investigation_search.webapp.run_web_ui") as run_web_ui:
            rc = main(
                [
                    "web",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "9100",
                    "--title",
                    "UI Test",
                    "--diagnostics-default",
                ]
            )

    assert rc == 0
    args, kwargs = run_web_ui.call_args
    assert args[0] is stub_engine
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 9100
    assert kwargs["config"].title == "UI Test"
    assert kwargs["config"].default_show_diagnostics is True
