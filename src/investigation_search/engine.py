from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional

from .embedding import DEFAULT_EMBEDDING_MODEL
from .index_ann import ANNIndex
from .reranker import LocalCrossEncoderReranker, Reranker
from .retrieval import build_passes, retrieve
from .schema import EvidenceUnit, ScoredEvidence, SearchResult, Verdict


class InvestigationEngine:
    def __init__(
        self,
        evidence_units: Iterable[EvidenceUnit],
        build_id: Optional[str] = None,
        *,
        ann_index: ANNIndex | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        reranker: Reranker | None = None,
    ):
        self.evidence_units = list(evidence_units)
        self.build_id = build_id
        self.ann_index = ann_index
        self.embedding_model = embedding_model
        self.reranker = reranker or LocalCrossEncoderReranker()

    def search(
        self,
        query: str,
        top_k_per_pass: int = 5,
        time_budget_sec: int = 120,
    ) -> SearchResult:
        start = time.time()
        selected: List[ScoredEvidence] = []
        pass_stats: Dict[str, object] = {}

        for qp in build_passes(query):
            elapsed = time.time() - start
            remaining = time_budget_sec - elapsed
            if remaining <= 0:
                pass_stats[qp.name] = "skipped_time_budget"
                continue

            hits = retrieve(
                qp,
                self.evidence_units,
                top_k=top_k_per_pass,
                ann_index=self.ann_index,
                embedding_model=self.embedding_model,
            )
            selected.extend(hits)
            pass_stats[qp.name] = f"{len(hits)} hits"

        rerank_budget_sec = 0.015 * max(len(selected), 1)
        rerank_meta: Dict[str, object] = {
            "enabled": bool(selected),
            "skipped": False,
            "reason": None,
            "candidate_count": len(selected),
            "budget_needed_sec": round(rerank_budget_sec, 4),
        }

        if not selected:
            rerank_meta["enabled"] = False
            rerank_meta["skipped"] = True
            rerank_meta["reason"] = "empty_candidates"
            reranked = []
        else:
            elapsed = time.time() - start
            remaining = time_budget_sec - elapsed
            # rerank must keep minimum slack for answer/diagnostics serialization.
            if remaining <= rerank_budget_sec + 0.01:
                rerank_meta["skipped"] = True
                rerank_meta["reason"] = "time_budget_exceeded"
                reranked = self._annotate_initial_scores(selected)
            else:
                reranked = self._rerank(query, selected)
                rerank_meta["reason"] = "applied"

        reranked.sort(key=lambda s: s.score, reverse=True)
        best = reranked[: max(top_k_per_pass, 3)]

        contradictions = [s for s in best if s.verdict == Verdict.CONTRADICTS]
        supports = [s for s in best if s.verdict == Verdict.SUPPORTS]

        if supports:
            answer = supports[0].evidence.content
        elif best:
            answer = best[0].evidence.content
        else:
            answer = "근거를 찾지 못했습니다."

        if not contradictions and best:
            pass_stats["disconfirming_check"] = "insufficient_contradicting_evidence"

        pass_stats["rerank"] = rerank_meta
        pass_stats["elapsed_sec"] = f"{time.time() - start:.3f}"
        pass_stats["time_budget_sec"] = str(time_budget_sec)

        return SearchResult(
            answer=answer,
            evidence=best,
            contradictions=contradictions,
            diagnostics=pass_stats,
            reranker_model=self.reranker.model_name,
            reranker_version=self.reranker.model_version,
            build_id=self.build_id,
        )

    def _annotate_initial_scores(self, candidates: List[ScoredEvidence]) -> List[ScoredEvidence]:
        return [
            ScoredEvidence(
                evidence=candidate.evidence,
                score=candidate.score,
                verdict=candidate.verdict,
                why_it_matches=candidate.why_it_matches,
                stage_scores={**candidate.stage_scores, "first_pass": candidate.score},
            )
            for candidate in candidates
        ]

    def _rerank(self, query: str, candidates: List[ScoredEvidence]) -> List[ScoredEvidence]:
        rerank_scores = self.reranker.score(query, candidates)
        reranked: List[ScoredEvidence] = []
        for candidate, rerank_score in zip(candidates, rerank_scores):
            final_score = candidate.score * 0.35 + rerank_score * 0.65
            reranked.append(
                ScoredEvidence(
                    evidence=candidate.evidence,
                    score=final_score,
                    verdict=candidate.verdict,
                    why_it_matches=f"{candidate.why_it_matches} + rerank {rerank_score:.3f}",
                    stage_scores={
                        **candidate.stage_scores,
                        "first_pass": candidate.score,
                        "rerank": rerank_score,
                        "final": final_score,
                    },
                )
            )
        return reranked
