from __future__ import annotations

from unittest.mock import patch
import tempfile
from pathlib import Path

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
from investigation_search.dsl import parse_search_query
from investigation_search.evaluation import EvaluationCase, evaluate_engine
from investigation_search.retrieval import QueryPass, RetrievalOptions, build_passes, retrieve
from investigation_search.schema import EvidenceUnit, SourceType, Verdict
from investigation_search.viewer import render_result_text
from investigation_search.websearch import WebSearchResult

from investigation_search.webfetch import _html_to_text, fetch_urls, WebFetchedPage


def test_html_to_text_extracts_same_host_links() -> None:
    html = """
    <html><head><title>Sample</title></head>
    <body>
      <a href="/docs/a.pdf">A</a>
      <a href="https://example.com/b">B</a>
      <a href="https://other.com/c">C</a>
      <p>Hello world</p>
    </body></html>
    """
    text, title, links = _html_to_text(html, base_url="https://example.com/start")
    assert title == "Sample"
    assert "Hello" in text
    assert "https://example.com/docs/a.pdf" in links
    assert "https://example.com/b" in links
    assert all("other.com" not in u for u in links)


def test_fetch_urls_preserves_input_order_with_parallel_fetch() -> None:
    urls = ["https://a.example.com", "https://b.example.com", "https://c.example.com"]

    def fake_fetch_one(raw_url: str, **kwargs):
        return WebFetchedPage(
            url=raw_url,
            final_url=raw_url,
            status=200,
            content_type="text/plain",
            title=raw_url,
            text="ok",
            bytes_read=2,
            truncated=False,
        )

    with patch("investigation_search.webfetch._fetch_one", side_effect=fake_fetch_one):
        pages = fetch_urls(
            urls,
            timeout_sec=1.0,
            user_agent="ua",
            max_bytes=1024,
            max_text_chars=1000,
            max_redirects=1,
            max_pages=3,
            allow_pdf=False,
            max_workers=3,
        )

    assert [p.url for p in pages] == urls


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


def test_build_passes_adapts_to_english_query() -> None:
    passes = build_passes("network latency optimization")
    assert len(passes) == 3
    assert "however" in passes[1].query
    assert "boundary" in passes[2].query


def test_low_confidence_ocr_is_filtered_by_quality_prior() -> None:
    units = [
        EvidenceUnit(
            doc_id="doc-text",
            source_type=SourceType.TEXT_SENTENCE,
            content="처리 속도 개선 효과가 있다.",
            section_path="A/1",
            char_start=0,
            char_end=16,
            timestamp="2024-01-01T00:00:00Z",
            confidence=0.9,
            metadata={},
        ),
        EvidenceUnit(
            doc_id="doc-ocr",
            source_type=SourceType.OCR_TEXT,
            content="처리 속도 개선 효과가 있다.",
            section_path="A/2",
            char_start=0,
            char_end=16,
            timestamp="2024-01-01T00:00:00Z",
            confidence=0.1,
            metadata={},
        ),
    ]
    hits = retrieve(
        QueryPass(name="pass_a_support", query="처리 속도 개선"),
        units,
        top_k=3,
        options=RetrievalOptions(min_ocr_confidence=0.3),
    )
    assert hits
    assert all(item.evidence.source_type != SourceType.OCR_TEXT for item in hits)


def test_delete_user_search_data_clears_cache_and_learning() -> None:
    engine = InvestigationEngine(_sample_units(), online_learning=True, cache_size=16, cache_ttl_sec=120.0)
    engine.search("처리 속도 개선", top_k_per_pass=2, time_budget_sec=120)
    before = engine.learning_snapshot()
    assert before["searches"] >= 1

    deleted = engine.delete_user_search_data(delete_learning_state_file=False)
    after = engine.learning_snapshot()

    assert deleted["learning_state_reset"] is True
    assert deleted["cache_entries_deleted"] >= 1
    assert after["searches"] == 0
    assert after["token_boost_count"] == 0


def test_parse_search_query_extracts_filters() -> None:
    parsed = parse_search_query('doc:doc-1 source:text -source:ocr after:2024-01-01 section:A/ "처리 속도"')
    assert parsed.clean_query == "처리 속도"
    assert "doc-1" in parsed.filters.doc_ids
    assert SourceType.TEXT_SENTENCE in parsed.filters.include_source_types
    assert SourceType.OCR_TEXT in parsed.filters.exclude_source_types
    assert "A/" in parsed.filters.section_prefixes
    assert parsed.filters.after == "2024-01-01"


def test_engine_search_dsl_filters_results() -> None:
    engine = InvestigationEngine(_sample_units(), enable_cache=False, online_learning=False)
    result = engine.search('doc:doc-1 source:text "처리 속도"', top_k_per_pass=3)
    assert result.evidence
    assert all(item.evidence.doc_id == "doc-1" for item in result.evidence)
    assert result.diagnostics["query_dsl"]["clean_query"] == "처리 속도"


def test_engine_explain_and_viewer_output() -> None:
    engine = InvestigationEngine(_sample_units())
    result = engine.search("처리 속도 개선", top_k_per_pass=2, time_budget_sec=120)
    explanation = engine.explain(result, max_items=2)
    rendered = render_result_text(result, query="처리 속도 개선", max_items=2)

    assert explanation["answer"]
    assert "stage_score_summary" in explanation
    assert "Answer:" in rendered
    assert "Evidence:" in rendered


def test_evaluation_harness_returns_metrics() -> None:
    engine = InvestigationEngine(_sample_units(), enable_cache=False, online_learning=False)
    case = EvaluationCase(query="처리 속도 개선", relevant_doc_ids=("doc-1",), expect_contradiction=True, top_k=3)
    report = evaluate_engine(engine, [case])
    assert report.metrics["query_count"] == 1.0
    assert 0.0 <= report.metrics["mrr"] <= 1.0
    assert len(report.per_case) == 1


def test_web_fallback_uses_provider_when_local_empty() -> None:
    class StubWebProvider:
        provider_name = "duckduckgo-stub"

        def search(self, query, *, max_results=5):
            return [
                WebSearchResult(
                    title="외부 검색 제목",
                    url="https://example.com/a",
                    snippet="외부 검색 스니펫",
                    rank=1,
                    provider=self.provider_name,
                )
            ]

    engine = InvestigationEngine(
        _sample_units(),
        enable_cache=False,
        online_learning=False,
        web_search_provider=StubWebProvider(),
        enable_web_fallback=True,
        web_fallback_min_local_hits=1,
    )
    result = engine.search("zzzxxyyqq", top_k_per_pass=2, time_budget_sec=120)

    assert result.evidence
    assert result.evidence[0].evidence.source_type == SourceType.WEB_SNIPPET
    assert result.diagnostics["web_fallback"]["used"] is True
    assert result.diagnostics["web_fallback"]["provider"] == "duckduckgo-stub"


def test_web_fallback_respects_dsl_source_exclusion() -> None:
    class StubWebProvider:
        provider_name = "duckduckgo-stub"

        def search(self, query, *, max_results=5):
            return [
                WebSearchResult(
                    title="외부 검색 제목",
                    url="https://example.com/a",
                    snippet="외부 검색 스니펫",
                    rank=1,
                    provider=self.provider_name,
                )
            ]

    engine = InvestigationEngine(
        _sample_units(),
        enable_cache=False,
        online_learning=False,
        web_search_provider=StubWebProvider(),
        enable_web_fallback=True,
    )
    result = engine.search('-source:web_snippet "zzzxxyyqq"', top_k_per_pass=2, time_budget_sec=120)

    assert result.evidence == []
    assert result.diagnostics["web_fallback"]["used"] is False
    assert result.diagnostics["web_fallback"]["reason"] == "filtered_out_by_query_dsl"


def test_sniper_mode_returns_single_evidence() -> None:
    engine = InvestigationEngine(_sample_units(), enable_cache=False, online_learning=False)
    result = engine.search("처리 속도 개선", top_k_per_pass=5, time_budget_sec=120, mode="sniper")
    assert len(result.evidence) == 1


def test_knowledge_library_persists_sessions() -> None:
    with tempfile.TemporaryDirectory() as td:
        class StubWebProvider:
            provider_name = "duckduckgo-stub"

            def search(self, query, *, max_results=5):
                return [
                    WebSearchResult(
                        title="외부 검색 제목",
                        url="https://example.com/a",
                        snippet="외부 검색 스니펫",
                        rank=1,
                        provider=self.provider_name,
                    )
                ]

        lib_dir = Path(td) / "library"
        engine = InvestigationEngine(
            _sample_units(),
            enable_cache=False,
            online_learning=False,
            enable_knowledge_library=True,
            knowledge_library_dir=lib_dir,
            web_search_provider=StubWebProvider(),
        )
        result = engine.search("처리 속도 개선", top_k_per_pass=2, time_budget_sec=120, mode="reporter")
        session_id = result.diagnostics.get("knowledge_library_session_id")
        assert session_id
        assert (lib_dir / "sessions" / f"{session_id}.json").exists()
        assert (lib_dir / "evidence" / "web_snippets.jsonl").exists()


def test_fbi_mode_emits_osint_graph_and_timeline() -> None:
    class StubWebProvider:
        provider_name = "duckduckgo-stub"

        def search(self, query, *, max_results=5):
            return [
                WebSearchResult(
                    title="Incident report 2024-01-02",
                    url="https://example.com/a",
                    snippet="Update on 2024-01-03 with details",
                    rank=1,
                    provider=self.provider_name,
                ),
                WebSearchResult(
                    title="Follow-up",
                    url="https://sub.example.com/b",
                    snippet="No explicit date here",
                    rank=2,
                    provider=self.provider_name,
                ),
            ]

    engine = InvestigationEngine(
        _sample_units(),
        enable_cache=False,
        online_learning=False,
        web_search_provider=StubWebProvider(),
        enable_web_fallback=True,
    )
    result = engine.search("zzzxxyyqq", mode="fbi")
    osint = result.diagnostics.get("osint")
    assert isinstance(osint, dict)
    assert "graph" in osint
    assert "timeline" in osint
    assert osint["graph"]["node_count"] >= 3
    assert osint["timeline"][0]["date"] == "2024-01-02"


def test_library_export_generates_html_pages() -> None:
    class StubWebProvider:
        provider_name = "duckduckgo-stub"

        def search(self, query, *, max_results=5):
            return [
                WebSearchResult(
                    title="Incident report 2024-01-02",
                    url="https://example.com/a",
                    snippet="Update on 2024-01-03 with details",
                    rank=1,
                    provider=self.provider_name,
                )
            ]

    with tempfile.TemporaryDirectory() as td:
        lib_dir = Path(td) / "library"
        out_dir = Path(td) / "export"
        engine = InvestigationEngine(
            _sample_units(),
            enable_cache=False,
            online_learning=False,
            web_search_provider=StubWebProvider(),
            enable_web_fallback=True,
            enable_knowledge_library=True,
            knowledge_library_dir=lib_dir,
        )
        result = engine.search("zzzxxyyqq", mode="fbi")
        session_id = result.diagnostics.get("knowledge_library_session_id")
        assert session_id

        from investigation_search.library_viewer import export_knowledge_library

        report = export_knowledge_library(lib_dir, out_dir, format="html", include_raw=True)
        assert report["exported_sessions"] >= 1
        assert (out_dir / "index.html").exists()
        assert (out_dir / "sessions" / f"{session_id}.html").exists()
        assert (out_dir / "osint" / f"{session_id}.dot").exists()


def test_library_publish_zip_creates_archive() -> None:
    with tempfile.TemporaryDirectory() as td:
        lib_dir = Path(td) / "library"
        out_zip = Path(td) / "bundle.zip"
        engine = InvestigationEngine(
            _sample_units(),
            enable_cache=False,
            online_learning=False,
            enable_knowledge_library=True,
            knowledge_library_dir=lib_dir,
        )
        result = engine.search("처리 속도 개선", mode="investigation")
        session_id = result.diagnostics.get("knowledge_library_session_id")
        assert session_id

        from investigation_search.publisher import publish_knowledge_library_zip

        out = publish_knowledge_library_zip(lib_dir, out_zip, export_format="md", session_ids=[session_id])
        assert out.exists()
        assert out.stat().st_size > 0




def test_modes_are_web_enabled_for_online_search() -> None:
    from investigation_search.modes import SearchMode, profile_for_mode

    for mode in SearchMode:
        profile = profile_for_mode(mode)
        assert profile.always_web_search is True
        assert profile.enable_web_fallback is True
        assert profile.web_max_results > 0

def test_subprocess_sandbox_web_provider_parses_json() -> None:
    from investigation_search.websearch import SubprocessSandboxWebSearchProvider

    class DummyProc:
        stdout = (
            '{'
            '"ok": true,'
            '"results": ['
            '{"title":"t","url":"https://example.com/a","snippet":"s","rank":1,"provider":"duckduckgo"}'
            "]"
            "}"
        )

    with patch("investigation_search.websearch.subprocess.run", return_value=DummyProc()):
        provider = SubprocessSandboxWebSearchProvider(timeout_sec=1.0)
        rows = provider.search("hello", max_results=2)
        assert rows
        assert rows[0].url == "https://example.com/a"
