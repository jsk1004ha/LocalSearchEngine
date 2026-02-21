from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional

from .retrieval import build_passes, retrieve
from .schema import EvidenceUnit, ScoredEvidence, SearchResult, Verdict


class InvestigationEngine:
    def __init__(self, evidence_units: Iterable[EvidenceUnit], build_id: Optional[str] = None):
        self.evidence_units = list(evidence_units)
        self.build_id = build_id

    def search(
        self,
        query: str,
        top_k_per_pass: int = 5,
        time_budget_sec: int = 120,
    ) -> SearchResult:
        start = time.time()
        selected: List[ScoredEvidence] = []
        pass_stats: Dict[str, str] = {}

        for qp in build_passes(query):
            elapsed = time.time() - start
            remaining = time_budget_sec - elapsed
            if remaining <= 0:
                pass_stats[qp.name] = "skipped_time_budget"
                continue

            hits = retrieve(qp, self.evidence_units, top_k=top_k_per_pass)
            selected.extend(hits)
            pass_stats[qp.name] = f"{len(hits)} hits"

        selected.sort(key=lambda s: s.score, reverse=True)
        best = selected[: max(top_k_per_pass, 3)]

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

        pass_stats["elapsed_sec"] = f"{time.time() - start:.3f}"
        pass_stats["time_budget_sec"] = str(time_budget_sec)

        return SearchResult(
            answer=answer,
            evidence=best,
            contradictions=contradictions,
            diagnostics=pass_stats,
            build_id=self.build_id,
        )
