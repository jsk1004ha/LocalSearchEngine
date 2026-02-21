from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional

from .llm import LLMConfig, synthesize_answer
from .retrieval import build_passes, retrieve
from .schema import EvidenceUnit, ScoredEvidence, SearchConfig, SearchResult, Verdict


class InvestigationEngine:
    def __init__(self, evidence_units: Iterable[EvidenceUnit], build_id: Optional[str] = None):
        self.evidence_units = list(evidence_units)
        self.build_id = build_id

    def search(
        self,
        query: str,
        config: Optional[SearchConfig] = None,
        llm_config: Optional[LLMConfig] = None,
    ) -> SearchResult:
        cfg = config or SearchConfig()
        start = time.time()
        selected: List[ScoredEvidence] = []
        pass_stats: Dict[str, str] = {}

        for qp in build_passes(query):
            elapsed = time.time() - start
            remaining = cfg.time_budget_sec - elapsed
            if remaining <= 0:
                pass_stats[qp.name] = "skipped_time_budget"
                continue

            hits = retrieve(
                qp,
                self.evidence_units,
                top_k=cfg.top_k_per_pass,
                lexical_weight=cfg.lexical_weight,
                dense_weight=cfg.dense_weight,
                rerank_weight=cfg.rerank_weight,
            )
            selected.extend(hits)
            pass_stats[qp.name] = f"{len(hits)} hits"

        selected.sort(key=lambda s: s.score, reverse=True)
        best = selected[: max(cfg.final_top_k, 3)]

        contradictions = [s for s in best if s.verdict == Verdict.CONTRADICTS]
        supports = [s for s in best if s.verdict == Verdict.SUPPORTS]

        synthesized = synthesize_answer(query, supports or best, llm_config=llm_config)
        answer = synthesized["answer"]
        pass_stats["answer_mode"] = synthesized["mode"]

        if not contradictions and best:
            pass_stats["disconfirming_check"] = "insufficient_contradicting_evidence"

        pass_stats["elapsed_sec"] = f"{time.time() - start:.3f}"
        pass_stats["time_budget_sec"] = str(cfg.time_budget_sec)

        return SearchResult(
            answer=answer,
            evidence=best,
            contradictions=contradictions,
            diagnostics=pass_stats,
            build_id=self.build_id,
        )
