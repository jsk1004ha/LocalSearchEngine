from __future__ import annotations

from unittest.mock import patch

from investigation_search.contradiction import ContradictionPrediction, HeuristicContradictionDetector
from investigation_search.engine import InvestigationEngine
from investigation_search.retrieval import QueryPass, build_passes, classify_verdict, retrieve
from investigation_search.schema import EvidenceUnit, ScoredEvidence, SourceType, Verdict
from investigation_search.verification import infer_claim_evidence_relation
from investigation_search.websearch import WebSearchResult


CLAIM = "처리 속도 개선"


def unit(text: str, doc_id: str = "fixture") -> EvidenceUnit:
    return EvidenceUnit(doc_id, SourceType.TEXT_SENTENCE, text, "fixture", 0, len(text), "2026-11-23", 1.0)


def test_counterevidence_pass_name_cannot_fabricate_a_contradiction() -> None:
    assert classify_verdict("pass_b_contradict", "처리 속도 개선 가능", claim=CLAIM) == Verdict.SUPPORTS
    assert classify_verdict("pass_d_disconfirming", "전혀 관계없는 작업은 불가", claim=CLAIM) == Verdict.UNCERTAIN
    assert classify_verdict("pass_a_support", "처리 속도 개선 효과는 없다", claim=CLAIM) == Verdict.CONTRADICTS


def test_retrieval_uses_original_claim_and_preserves_two_argument_query_passes() -> None:
    assert QueryPass("legacy", CLAIM).claim is None
    assert all(qp.claim == CLAIM for qp in build_passes(CLAIM))
    hits = retrieve(
        QueryPass("pass_d_disconfirming", f"{CLAIM} 부정 예외 반증 제한 실패", claim=CLAIM),
        [unit("처리 속도 개선 가능")], top_k=1,
    )
    assert hits[0].verdict == Verdict.SUPPORTS
    assert hits[0].stage_scores["pair_entail"] > hits[0].stage_scores["pair_contradict"]
    assert hits[0].source is not None


def test_default_detector_does_not_restore_keyword_only_contradictions() -> None:
    candidate = ScoredEvidence(unit("전혀 관계없는 작업은 불가"), 0.8, Verdict.UNCERTAIN, "fixture")
    result = HeuristicContradictionDetector().predict(CLAIM, [candidate])[0]
    assert result.verdict == Verdict.UNCERTAIN
    english = infer_claim_evidence_relation("processing speed improves", "processing speed improves is not supported")
    assert english.contradict > english.entail


def test_disconfirming_pass_runs_once_and_keeps_sources_and_confidence() -> None:
    engine = InvestigationEngine([unit("처리 속도 개선 가능")], enable_web_fallback=False, online_learning=False)
    with patch.object(engine, "_retrieve_from_shards", wraps=engine._retrieve_from_shards) as retrieve_spy:
        result = engine.search(CLAIM)
    assert sum(call.args[1] == "pass_d_disconfirming" for call in retrieve_spy.call_args_list) == 1
    assert all(call.kwargs["claim"] == CLAIM for call in retrieve_spy.call_args_list)
    assert result.diagnostics["evidence_confidence"]["method"] == "heuristic_pair_score"
    assert result.diagnostics["evidence_confidence"]["supports"] > 0
    assert result.diagnostics["evidence_confidence"]["contradicts"] == 0
    assert result.sources and result.answer_sources


def test_custom_detector_contradiction_skips_extra_pass() -> None:
    class Detector:
        model_name = "fixture"
        model_version = "1"

        def predict(self, query, candidates):
            return [ContradictionPrediction(Verdict.CONTRADICTS, 0.99, "fixture") for _ in candidates]

    engine = InvestigationEngine(
        [unit("처리 속도 개선 가능")], enable_web_fallback=False, online_learning=False,
        contradiction_detector=Detector(),
    )
    result = engine.search(CLAIM)
    assert result.contradictions
    assert "pass_d_disconfirming" not in result.diagnostics


def test_exhausted_budget_does_not_launch_an_extra_pass() -> None:
    engine = InvestigationEngine([unit("처리 속도 개선 가능")], enable_web_fallback=False, online_learning=False)
    clock = [0.0]
    rerank = engine._rerank

    def finish_at_deadline(*args, **kwargs):
        result = rerank(*args, **kwargs)
        clock[0] = 120.0
        return result

    with patch("investigation_search.engine.time.time", side_effect=lambda: clock[0]), \
            patch.object(engine, "_rerank", side_effect=finish_at_deadline), \
            patch.object(engine, "_retrieve_from_shards", wraps=engine._retrieve_from_shards) as retrieve_spy:
        result = engine.search(CLAIM, time_budget_sec=120)
    assert result.diagnostics["pass_d_disconfirming"] == "skipped_time_budget"
    assert all(call.args[1] != "pass_d_disconfirming" for call in retrieve_spy.call_args_list)


def test_extra_web_pass_finds_counterevidence_without_losing_citations() -> None:
    class Provider:
        provider_name = "fixture"

        def __init__(self):
            self.queries = []

        def search(self, query, max_results=3):
            self.queries.append(query)
            extra = "반증 제한 실패" in query
            return [WebSearchResult(
                title=CLAIM,
                url="https://example.test/contra" if extra else "https://example.test/support",
                snippet="처리 속도 개선 효과는 없다" if extra else "처리 속도 개선 가능",
                rank=1, provider=self.provider_name,
            )]

    provider = Provider()
    engine = InvestigationEngine([], web_search_provider=provider, online_learning=False)
    with patch("investigation_search.engine.encode_texts", side_effect=RuntimeError("offline embedding fixture")):
        result = engine.search(CLAIM)
    assert sum("반증 제한 실패" in query for query in provider.queries) == 1
    assert any(item.evidence.doc_id.endswith("/contra") for item in result.contradictions)
    assert all(item.source is not None for item in result.evidence)
    assert result.diagnostics["evidence_confidence"]["contradicts"] > 0


def test_counter_query_suffixes_do_not_replace_results_with_unrelated_local_text() -> None:
    class Provider:
        provider_name = "fixture"

        def search(self, query, max_results=3):
            return [WebSearchResult(
                title="external result", url="https://example.test/external",
                snippet="external search snippet", rank=1, provider=self.provider_name,
            )]

    engine = InvestigationEngine(
        [unit("다른 작업은 예외 구간에서 불가하고 제한이 있다")],
        web_search_provider=Provider(), online_learning=False,
    )
    with patch("investigation_search.engine.encode_texts", side_effect=RuntimeError("offline embedding fixture")):
        result = engine.search("zzzxxyyqq")
    assert result.evidence
    assert all(item.evidence.source_type == SourceType.WEB_SNIPPET for item in result.evidence)
    assert result.diagnostics["pass_d_disconfirming"]["hits"] == 0


def test_negated_claim_tie_cannot_become_support_in_the_default_detector() -> None:
    claim = "the system does not allow users to open a protected file without permission"
    evidence = "the system does allow users to open a protected file without permission"
    verdict = classify_verdict("pass_a_support", evidence, claim=claim)
    assert verdict == Verdict.CONTRADICTS
    candidate = ScoredEvidence(unit(evidence), 0.9, verdict, "fixture")
    prediction = HeuristicContradictionDetector().predict(claim, [candidate])[0]
    assert prediction.verdict != Verdict.SUPPORTS
    engine = InvestigationEngine([], enable_web_fallback=False, online_learning=False)
    updated, meta = engine._apply_contradiction_detector(claim, [candidate], {})
    assert updated[0].verdict == Verdict.CONTRADICTS
    assert meta["overrides"] == 0
