from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, Iterable, List, Optional, Sequence

from .bm25 import BM25Index, build_bm25_index
from .contradiction import ContradictionDetector, HeuristicContradictionDetector
from .dsl import ParsedSearchQuery, SearchFilters, apply_search_filters, filter_to_dict, parse_search_query
from .embedding import DEFAULT_EMBEDDING_MODEL, encode_texts
from .index_ann import ANNIndex, build_index
from .library import KnowledgeLibrary
from .learning import LearningConfig, OnlineLearningManager
from .modes import SearchMode, build_passes_for_mode, parse_mode, profile_for_mode
from .osint import build_osint_graph, extract_timeline
from .reranker import LocalCrossEncoderReranker, Reranker
from .retrieval import (
    QueryPass,
    RetrievalOptions,
    RetrievalWeights,
    build_bm25_for_units,
    build_passes,
    retrieve,
)
from .schema import EvidenceUnit, ScoredEvidence, SearchResult, SourceCitation, SourceType, Verdict, build_source_citation
from .viewer import summarize_stage_scores
from .webfetch import (
    StdlibWebFetchProvider,
    SubprocessSandboxWebFetchProvider,
    WebFetchProvider,
    WebFetchedPage,
    chunk_text,
)
from .websearch import DuckDuckGoSearchProvider, SubprocessSandboxWebSearchProvider, WebSearchProvider, WebSearchResult


@dataclass(frozen=True)
class _ShardRuntime:
    shard_id: int
    units: List[EvidenceUnit]
    ann_index: ANNIndex | None
    bm25_index: BM25Index


@dataclass
class _CacheEntry:
    created_at: float
    result: SearchResult


class InvestigationEngine:
    def __init__(
        self,
        evidence_units: Iterable[EvidenceUnit],
        build_id: Optional[str] = None,
        *,
        ann_index: ANNIndex | None = None,
        ann_shards: Sequence[ANNIndex] | None = None,
        bm25_index: BM25Index | None = None,
        bm25_shards: Sequence[BM25Index] | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        reranker: Reranker | None = None,
        contradiction_detector: ContradictionDetector | None = None,
        retrieval_weights: RetrievalWeights | None = None,
        shard_count: int = 1,
        enable_cache: bool = True,
        cache_size: int = 128,
        cache_ttl_sec: float = 90.0,
        online_learning: bool = True,
        learning_state_path: str | None = None,
        learning_rate: float = 0.08,
        learning_autosave_every: int = 20,
        retrieval_options: RetrievalOptions | None = None,
        max_rerank_candidates: int = 32,
        enable_search_dsl: bool = True,
        enable_web_fallback: bool = True,
        web_fallback_min_local_hits: int = 1,
        web_max_results: int = 3,
        web_search_provider: WebSearchProvider | None = None,
        enable_web_sandbox: bool = True,
        web_sandbox_timeout_sec: float = 14.0,
        web_sandbox_max_bytes: int = 1_500_000,
        enable_web_fetch: bool = False,
        web_fetch_max_pages: int = 4,
        web_fetch_provider: WebFetchProvider | None = None,
        web_fetch_timeout_sec: float = 12.0,
        web_fetch_max_bytes: int = 2_500_000,
        web_fetch_max_text_chars: int = 180_000,
        web_fetch_allow_pdf: bool = True,
        web_fetch_crawl_depth: int = 0,
        web_fetch_max_workers: int = 4,
        enable_knowledge_library: bool = False,
        knowledge_library_dir: str | Path = Path("artifacts") / "knowledge_library",
        trusted_domains: Sequence[str] | None = None,
    ):
        self.evidence_units = list(evidence_units)
        self.build_id = build_id
        self.shard_count = max(1, int(shard_count))
        self.embedding_model = embedding_model
        self.reranker = reranker or LocalCrossEncoderReranker()
        self.contradiction_detector = contradiction_detector or HeuristicContradictionDetector()
        self.retrieval_weights = retrieval_weights or RetrievalWeights()
        self.retrieval_options = retrieval_options or RetrievalOptions()
        self.enable_cache = enable_cache
        self.cache_size = max(0, cache_size)
        self.cache_ttl_sec = max(0.0, cache_ttl_sec)
        self.max_rerank_candidates = max(1, int(max_rerank_candidates))
        self.enable_search_dsl = bool(enable_search_dsl)
        self.enable_web_fallback = bool(enable_web_fallback)
        self.web_fallback_min_local_hits = max(0, int(web_fallback_min_local_hits))
        self.web_max_results = max(1, int(web_max_results))
        if web_search_provider is not None:
            self.web_search_provider = web_search_provider
        else:
            if enable_web_sandbox:
                self.web_search_provider = SubprocessSandboxWebSearchProvider(
                    timeout_sec=max(1.0, float(web_sandbox_timeout_sec)),
                    max_bytes=max(0, int(web_sandbox_max_bytes)),
                )
            else:
                self.web_search_provider = DuckDuckGoSearchProvider(
                    timeout_sec=max(1.0, float(web_sandbox_timeout_sec)),
                    max_bytes=max(0, int(web_sandbox_max_bytes)),
                )
        self.enable_web_fetch = bool(enable_web_fetch)
        self.web_fetch_max_pages = max(0, int(web_fetch_max_pages))
        self.web_fetch_crawl_depth = max(0, min(int(web_fetch_crawl_depth), 2))
        if web_fetch_provider is not None:
            self.web_fetch_provider = web_fetch_provider
        else:
            if enable_web_sandbox:
                self.web_fetch_provider = SubprocessSandboxWebFetchProvider(
                    timeout_sec=max(1.0, float(web_fetch_timeout_sec)),
                    max_bytes=max(0, int(web_fetch_max_bytes)),
                    max_text_chars=max(0, int(web_fetch_max_text_chars)),
                    allow_pdf=bool(web_fetch_allow_pdf),
                    max_workers=max(1, int(web_fetch_max_workers)),
                )
            else:
                self.web_fetch_provider = StdlibWebFetchProvider(
                    timeout_sec=max(1.0, float(web_fetch_timeout_sec)),
                    max_bytes=max(0, int(web_fetch_max_bytes)),
                    max_text_chars=max(0, int(web_fetch_max_text_chars)),
                    allow_pdf=bool(web_fetch_allow_pdf),
                    max_workers=max(1, int(web_fetch_max_workers)),
                )
        self.trusted_domains = {d.strip().lower() for d in (trusted_domains or []) if str(d).strip()}
        self._library = KnowledgeLibrary(knowledge_library_dir) if enable_knowledge_library else None
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._learner = OnlineLearningManager(
            self.retrieval_weights,
            LearningConfig(
                enabled=online_learning,
                learning_rate=learning_rate,
                autosave_every=max(1, learning_autosave_every),
                state_path=learning_state_path,
            ),
        )

        shard_units = self._partition_units(self.evidence_units, self.shard_count)
        ann_by_shard = self._prepare_ann_shards(ann_index=ann_index, ann_shards=ann_shards, shard_count=self.shard_count)
        bm25_by_shard = self._prepare_bm25_shards(
            shard_units=shard_units,
            bm25_index=bm25_index,
            bm25_shards=bm25_shards,
        )

        self._shards: list[_ShardRuntime] = []
        for shard_id in range(self.shard_count):
            units = shard_units[shard_id]
            self._shards.append(
                _ShardRuntime(
                    shard_id=shard_id,
                    units=units,
                    ann_index=ann_by_shard[shard_id],
                    bm25_index=bm25_by_shard[shard_id],
                )
            )

    def search(
        self,
        query: str,
        top_k_per_pass: int = 5,
        time_budget_sec: int = 120,
        *,
        mode: str | SearchMode | None = None,
    ) -> SearchResult:
        mode_enum = parse_mode(mode)
        profile = profile_for_mode(mode_enum)
        effective_top_k = int(top_k_per_pass)
        if mode_enum != SearchMode.INVESTIGATION:
            effective_top_k = max(effective_top_k, profile.top_k_per_pass_min)
        if mode_enum == SearchMode.SNIPER:
            effective_top_k = 1
        effective_retrieval_options = replace(self.retrieval_options, **dict(profile.retrieval_options_overrides))

        cache_key = self._cache_key(
            query=query,
            top_k_per_pass=effective_top_k,
            time_budget_sec=time_budget_sec,
            mode=mode_enum.value,
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            return self._with_cache_status(cached, status="hit", learning_meta=self._learner.snapshot())

        start = time.time()
        selected: List[ScoredEvidence] = []
        pass_stats: Dict[str, object] = {}
        pass_stats["cache"] = {"status": "miss"}
        pass_stats["sharding"] = {"shard_count": self.shard_count}
        pass_stats["online_learning"] = self._learner.snapshot()
        active_weights = self._learner.current_weights()
        active_token_boosts = self._learner.current_token_boosts()

        parsed_query = self._parse_query(query)
        query_text = parsed_query.clean_query
        pass_stats["query_dsl"] = {
            "enabled": self.enable_search_dsl,
            "raw_query": parsed_query.raw_query,
            "clean_query": parsed_query.clean_query,
            "filters": filter_to_dict(parsed_query.filters),
            "phrase_terms": list(parsed_query.phrase_terms),
        }
        pass_stats["mode"] = {
            "name": mode_enum.value,
            "description": profile.description,
            "effective_top_k_per_pass": effective_top_k,
        }
        pass_stats["retrieval_weights"] = {
            "bm25": active_weights.bm25,
            "dense": active_weights.dense,
            "lexical": active_weights.lexical,
            "rrf": active_weights.rrf,
        }

        passes = tuple(build_passes_for_mode(query_text, mode_enum))
        for qp in passes:
            elapsed = time.time() - start
            remaining = time_budget_sec - elapsed
            if remaining <= 0:
                pass_stats[qp.name] = "skipped_time_budget"
                continue

            hits, shard_hits = self._retrieve_from_shards(
                qp.query,
                qp.name,
                top_k=effective_top_k,
                weights=active_weights,
                token_boosts=active_token_boosts,
                time_budget_sec=time_budget_sec,
                elapsed_sec=elapsed,
                filters=parsed_query.filters,
                retrieval_options=effective_retrieval_options,
            )
            selected.extend(hits)
            pass_stats[qp.name] = {"hits": len(hits), "shards": shard_hits}

        selected = self._dedupe_candidates(selected)

        enable_web = self.enable_web_fallback and profile.enable_web_fallback
        web_meta: dict[str, object] = {
            "enabled": enable_web,
            "provider": getattr(self.web_search_provider, "provider_name", None),
            "used": False,
            "reason": None,
            "result_count": 0,
        }
        web_rows: list[WebSearchResult] = []

        should_web_search = profile.always_web_search or (len(selected) < self.web_fallback_min_local_hits)
        if enable_web and should_web_search:
            if self._web_allowed_by_filters(parsed_query.filters):
                web_hits, web_meta, web_rows = self._web_fallback(
                    query_text,
                    passes=passes,
                    filters=parsed_query.filters,
                    web_meta=web_meta,
                    score_base=profile.web_score_base,
                    max_results=profile.web_max_results,
                    top_k=effective_top_k,
                    weights=active_weights,
                    token_boosts=active_token_boosts,
                    retrieval_options=effective_retrieval_options,
                    enable_web_fetch=bool(self.enable_web_fetch and profile.enable_web_fetch),
                    web_fetch_max_pages=min(self.web_fetch_max_pages, int(profile.web_fetch_max_pages)),
                    web_fetch_crawl_depth=int(self.web_fetch_crawl_depth),
                )
                if web_hits:
                    selected.extend(web_hits)
                    selected = self._dedupe_candidates(selected)
                if mode_enum in {SearchMode.FBI, SearchMode.COLLECTION} and web_rows:
                    pass_stats.setdefault("osint", {})
                    pass_stats["osint"]["web_results"] = [
                        {"rank": row.rank, "title": row.title, "url": row.url, "snippet": row.snippet}
                        for row in web_rows
                    ]
                    if mode_enum == SearchMode.FBI:
                        pass_stats["osint"]["graph"] = build_osint_graph(query_text, web_rows)
                        pass_stats["osint"]["timeline"] = extract_timeline(web_rows, limit=30)
            else:
                web_meta["reason"] = "filtered_out_by_query_dsl"
        elif enable_web:
            web_meta["reason"] = "enough_local_hits"
        else:
            web_meta["reason"] = "disabled"

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
                reranked = self._rerank(query_text, selected, max_candidates=profile.max_rerank_candidates)
                rerank_meta["reason"] = "applied"
                rerank_meta["applied_count"] = int(
                    sum(1 for item in reranked if item.stage_scores.get("rerank_applied", 0.0) >= 0.5)
                )
                rerank_meta["max_rerank_candidates"] = profile.max_rerank_candidates

        detector_meta = {
            "model": self.contradiction_detector.model_name,
            "version": self.contradiction_detector.model_version,
            "overrides": 0,
        }
        if reranked:
            reranked, detector_meta = self._apply_contradiction_detector(query_text, reranked, detector_meta)

        reranked.sort(key=lambda s: s.score, reverse=True)
        best = reranked[: max(effective_top_k, 3)]

        contradictions = [s for s in best if s.verdict == Verdict.CONTRADICTS]
        supports = [s for s in best if s.verdict == Verdict.SUPPORTS]

        if supports:
            answer = supports[0].evidence.content
            answer_source = supports[0].source or build_source_citation(supports[0].evidence)
        elif best:
            answer = best[0].evidence.content
            answer_source = best[0].source or build_source_citation(best[0].evidence)
        else:
            answer = "근거를 찾지 못했습니다."
            answer_source = None

        if not contradictions and best:
            pass_stats["disconfirming_check"] = "insufficient_contradicting_evidence"

        if profile.include_only_trusted_sources:
            best = self._filter_trusted(best)
            contradictions = [s for s in best if s.verdict == Verdict.CONTRADICTS]
            supports = [s for s in best if s.verdict == Verdict.SUPPORTS]
            if supports:
                answer = supports[0].evidence.content
                answer_source = supports[0].source or build_source_citation(supports[0].evidence)
            elif best:
                answer = best[0].evidence.content
                answer_source = best[0].source or build_source_citation(best[0].evidence)
            else:
                answer = "근거를 찾지 못했습니다."
                answer_source = None

        sources = self._collect_sources(best)
        if answer_source is not None:
            pass_stats["answer_source"] = answer_source.citation_id
        pass_stats["source_count"] = len(sources)

        end_ts = time.time()
        pass_stats["rerank"] = rerank_meta
        pass_stats["contradiction_detector"] = detector_meta
        pass_stats["web_fallback"] = web_meta
        pass_stats["entity_groups"] = self._group_by_entity(best)
        pass_stats["elapsed_sec"] = f"{end_ts - start:.3f}"
        pass_stats["time_budget_sec"] = str(time_budget_sec)

        result = SearchResult(
            answer=self._format_answer_for_mode(mode_enum, answer=answer, best=best, supports=supports, contradictions=contradictions),
            evidence=best[:1] if mode_enum == SearchMode.SNIPER else best,
            contradictions=contradictions,
            diagnostics=pass_stats,
            reranker_model=self.reranker.model_name,
            reranker_version=self.reranker.model_version,
            build_id=self.build_id,
            answer_sources=[answer_source] if answer_source is not None else [],
            sources=sources,
        )
        pass_stats["online_learning"] = self._learner.learn(query_text, best)
        session_id = self._persist_to_library(
            mode=mode_enum.value,
            parsed_query=parsed_query,
            result=result,
            extra={"web_fallback": web_meta, "mode": pass_stats.get("mode"), "query_dsl": pass_stats.get("query_dsl")},
        )
        if session_id is not None:
            pass_stats["knowledge_library_session_id"] = session_id
        self._cache_put(cache_key, result, created_at=end_ts)
        return result

    def _annotate_initial_scores(self, candidates: List[ScoredEvidence]) -> List[ScoredEvidence]:
        return [
            ScoredEvidence(
                evidence=candidate.evidence,
                score=candidate.score,
                verdict=candidate.verdict,
                why_it_matches=candidate.why_it_matches,
                stage_scores={**candidate.stage_scores, "first_pass": candidate.score, "rerank_applied": 0.0},
                source=candidate.source or build_source_citation(candidate.evidence),
            )
            for candidate in candidates
        ]

    def _rerank(self, query: str, candidates: List[ScoredEvidence], *, max_candidates: int) -> List[ScoredEvidence]:
        max_candidates = max(1, int(max_candidates))
        if len(candidates) <= max_candidates:
            rerank_indices = list(range(len(candidates)))
        else:
            ranked_by_score = sorted(range(len(candidates)), key=lambda idx: candidates[idx].score, reverse=True)
            rerank_indices = ranked_by_score[:max_candidates]

        rerank_subset = [candidates[idx] for idx in rerank_indices]
        rerank_scores = self.reranker.score(query, rerank_subset)
        rerank_map = {idx: score for idx, score in zip(rerank_indices, rerank_scores)}

        reranked: List[ScoredEvidence] = []
        for idx, candidate in enumerate(candidates):
            rerank_score = rerank_map.get(idx, candidate.score)
            rerank_applied = idx in rerank_map
            final_score = candidate.score * 0.35 + rerank_score * 0.65
            reranked.append(
                ScoredEvidence(
                    evidence=candidate.evidence,
                    score=final_score,
                    verdict=candidate.verdict,
                    why_it_matches=(
                        f"{candidate.why_it_matches} + rerank {rerank_score:.3f}"
                        if rerank_applied
                        else f"{candidate.why_it_matches} + rerank_skipped"
                    ),
                    stage_scores={
                        **candidate.stage_scores,
                        "first_pass": candidate.score,
                        "rerank": rerank_score if rerank_applied else 0.0,
                        "rerank_applied": 1.0 if rerank_applied else 0.0,
                        "final": final_score,
                    },
                    source=candidate.source or build_source_citation(candidate.evidence),
                )
            )
        return reranked

    def _group_by_entity(self, evidence_items: List[ScoredEvidence]) -> Dict[str, object]:
        grouped: dict[str, list[ScoredEvidence]] = defaultdict(list)
        for item in evidence_items:
            canonical_id = item.evidence.metadata.get("canonical_entity_id")
            if canonical_id is None:
                grouped["null"].append(item)
            else:
                grouped[str(canonical_id)].append(item)

        return {
            key: {
                "count": len(items),
                "top_score": max(e.score for e in items),
                "doc_ids": sorted({e.evidence.doc_id for e in items}),
            }
            for key, items in grouped.items()
        }

    def _retrieve_from_shards(
        self,
        query: str,
        pass_name: str,
        *,
        top_k: int,
        weights: RetrievalWeights,
        token_boosts: Dict[str, float],
        time_budget_sec: int,
        elapsed_sec: float,
        filters: SearchFilters,
        retrieval_options: RetrievalOptions,
    ) -> tuple[List[ScoredEvidence], Dict[str, int]]:
        shard_hits: dict[str, int] = {}
        all_hits: list[ScoredEvidence] = []
        budget_ratio = 1.0
        if time_budget_sec > 0:
            budget_ratio = max(0.05, min(1.0, (time_budget_sec - elapsed_sec) / time_budget_sec))
        dense_factor = 6 if budget_ratio >= 0.45 else 4
        candidate_factor = 8 if budget_ratio >= 0.45 else 5

        for shard in self._shards:
            filtered_units = apply_search_filters(shard.units, filters)
            if not filtered_units:
                shard_hits[str(shard.shard_id)] = 0
                continue
            can_reuse_shard_indexes = len(filtered_units) == len(shard.units)
            hits = retrieve(
                query_pass=QueryPass(name=pass_name, query=query),
                evidence_units=filtered_units,
                top_k=max(top_k, 2),
                bm25_index=shard.bm25_index if can_reuse_shard_indexes else None,
                ann_index=shard.ann_index if can_reuse_shard_indexes else None,
                embedding_model=self.embedding_model,
                weights=weights,
                token_boosts=token_boosts,
                options=retrieval_options,
                dense_top_k_factor=dense_factor,
                candidate_limit_factor=candidate_factor,
            )
            all_hits.extend(hits)
            shard_hits[str(shard.shard_id)] = len(hits)

        merged = self._dedupe_candidates(all_hits)
        merged.sort(key=lambda item: item.score, reverse=True)
        return merged[:top_k], shard_hits

    def _dedupe_candidates(self, candidates: List[ScoredEvidence]) -> List[ScoredEvidence]:
        merged: dict[tuple[str, int, int], ScoredEvidence] = {}
        for candidate in candidates:
            key = (candidate.evidence.doc_id, candidate.evidence.char_start, candidate.evidence.char_end)
            prev = merged.get(key)
            if prev is None:
                merged[key] = candidate
                continue
            if candidate.score >= prev.score:
                merged[key] = ScoredEvidence(
                    evidence=candidate.evidence,
                    score=candidate.score,
                    verdict=candidate.verdict,
                    why_it_matches=f"{prev.why_it_matches} + {candidate.why_it_matches}",
                    stage_scores={**prev.stage_scores, **candidate.stage_scores},
                    source=candidate.source or prev.source or build_source_citation(candidate.evidence),
                )
            else:
                merged[key] = ScoredEvidence(
                    evidence=prev.evidence,
                    score=prev.score,
                    verdict=prev.verdict,
                    why_it_matches=f"{prev.why_it_matches} + {candidate.why_it_matches}",
                    stage_scores={**prev.stage_scores, **candidate.stage_scores},
                    source=prev.source or candidate.source or build_source_citation(prev.evidence),
                )
        return list(merged.values())

    def _apply_contradiction_detector(
        self,
        query: str,
        candidates: List[ScoredEvidence],
        detector_meta: Dict[str, object],
    ) -> tuple[List[ScoredEvidence], Dict[str, object]]:
        predictions = self.contradiction_detector.predict(query, candidates)
        updated: list[ScoredEvidence] = []
        overrides = 0

        for candidate, prediction in zip(candidates, predictions):
            verdict = candidate.verdict
            if prediction.score >= 0.55:
                if verdict != prediction.verdict:
                    overrides += 1
                verdict = prediction.verdict
            updated.append(
                ScoredEvidence(
                    evidence=candidate.evidence,
                    score=candidate.score,
                    verdict=verdict,
                    why_it_matches=f"{candidate.why_it_matches}, contradiction={prediction.rationale}:{prediction.score:.3f}",
                    stage_scores={**candidate.stage_scores, "contradiction_score": prediction.score},
                    source=candidate.source or build_source_citation(candidate.evidence),
                )
            )

        detector_meta = dict(detector_meta)
        detector_meta["overrides"] = overrides
        detector_meta["candidate_count"] = len(candidates)
        return updated, detector_meta

    def _partition_units(self, units: List[EvidenceUnit], shard_count: int) -> List[List[EvidenceUnit]]:
        partitions = [[] for _ in range(shard_count)]
        for unit in units:
            shard_id = self._shard_for_doc(unit.doc_id, shard_count)
            partitions[shard_id].append(unit)
        return partitions

    def _prepare_ann_shards(
        self,
        *,
        ann_index: ANNIndex | None,
        ann_shards: Sequence[ANNIndex] | None,
        shard_count: int,
    ) -> List[ANNIndex | None]:
        if shard_count == 1:
            if ann_shards is None:
                return [ann_index]
            if len(ann_shards) != 1:
                raise ValueError("shard_count=1일 때 ann_shards 길이는 1이어야 합니다.")
            return [ann_shards[0]]
        if ann_shards is None:
            return [None for _ in range(shard_count)]
        if len(ann_shards) != shard_count:
            raise ValueError("ann_shards 길이는 shard_count와 같아야 합니다.")
        return list(ann_shards)

    def _prepare_bm25_shards(
        self,
        *,
        shard_units: List[List[EvidenceUnit]],
        bm25_index: BM25Index | None,
        bm25_shards: Sequence[BM25Index] | None,
    ) -> List[BM25Index]:
        if self.shard_count == 1:
            if bm25_shards:
                if len(bm25_shards) != 1:
                    raise ValueError("shard_count=1일 때 bm25_shards 길이는 1이어야 합니다.")
                return [bm25_shards[0]]
            if bm25_index is not None:
                return [bm25_index]
            return [build_bm25_for_units(shard_units[0])]

        if bm25_shards is not None:
            if len(bm25_shards) != self.shard_count:
                raise ValueError("bm25_shards 길이는 shard_count와 같아야 합니다.")
            return list(bm25_shards)

        return [build_bm25_for_units(units) for units in shard_units]

    def _cache_key(self, *, query: str, top_k_per_pass: int, time_budget_sec: int, mode: str) -> str:
        learning_bucket = self._learner.version // 10
        payload = (
            f"{self.build_id}|{self.embedding_model}|{self.shard_count}|"
            f"{top_k_per_pass}|{time_budget_sec}|{len(self.evidence_units)}|"
            f"{learning_bucket}|{int(self.enable_web_fallback)}|{int(self.enable_web_fetch)}|{self.web_fetch_max_pages}|"
            f"{mode}|{query.strip().lower()}"
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> SearchResult | None:
        if not self.enable_cache or self.cache_size <= 0:
            return None
        entry = self._cache.get(key)
        if entry is None:
            return None
        now = time.time()
        if (now - entry.created_at) > self.cache_ttl_sec:
            self._cache.pop(key, None)
            return None
        self._cache.move_to_end(key)
        return entry.result

    def _cache_put(self, key: str, result: SearchResult, *, created_at: float | None = None) -> None:
        if not self.enable_cache or self.cache_size <= 0:
            return
        if created_at is None:
            created_at = time.time()
        self._cache[key] = _CacheEntry(created_at=created_at, result=result)
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _with_cache_status(self, result: SearchResult, *, status: str, learning_meta: Dict[str, object]) -> SearchResult:
        diagnostics = dict(result.diagnostics)
        prev_cache = diagnostics.get("cache")
        if isinstance(prev_cache, dict):
            cache_meta = dict(prev_cache)
        else:
            cache_meta = {}
        cache_meta["status"] = status
        diagnostics["cache"] = cache_meta
        diagnostics["online_learning"] = learning_meta
        return SearchResult(
            answer=result.answer,
            evidence=result.evidence,
            contradictions=result.contradictions,
            diagnostics=diagnostics,
            reranker_model=result.reranker_model,
            reranker_version=result.reranker_version,
            build_id=result.build_id,
            answer_sources=result.answer_sources,
            sources=result.sources,
        )

    def _collect_sources(self, items: Sequence[ScoredEvidence]) -> List[SourceCitation]:
        out: list[SourceCitation] = []
        seen: set[str] = set()
        for item in items:
            source = item.source or build_source_citation(item.evidence)
            if source.citation_id in seen:
                continue
            seen.add(source.citation_id)
            out.append(source)
        return out

    def learning_snapshot(self) -> Dict[str, object]:
        return self._learner.snapshot()

    def save_learning_state(self, path: str | None = None) -> None:
        self._learner.save(path)

    def clear_cache(self) -> None:
        self._cache.clear()

    def delete_user_search_data(
        self,
        *,
        delete_learning_state_file: bool = True,
        delete_knowledge_library: bool = False,
    ) -> Dict[str, object]:
        cache_size = len(self._cache)
        self._cache.clear()
        self._learner.clear(delete_state_file=delete_learning_state_file)
        library_deleted = False
        if delete_knowledge_library and self._library is not None:
            self._library.delete_all()
            self._library = None
            library_deleted = True
        return {
            "cache_entries_deleted": cache_size,
            "learning_state_reset": True,
            "learning_state_file_deleted": bool(delete_learning_state_file),
            "knowledge_library_deleted": library_deleted,
        }

    def explain(self, result: SearchResult, *, max_items: int = 5) -> Dict[str, object]:
        items = list(result.evidence[: max(max_items, 0)])
        evidence = []
        for item in items:
            source = item.source or build_source_citation(item.evidence)
            evidence.append(
                {
                    "citation_id": source.citation_id,
                    "doc_id": source.doc_id,
                    "section_path": source.section_path,
                    "source_type": source.source_type.value,
                    "verdict": item.verdict.value,
                    "score": item.score,
                    "why": item.why_it_matches,
                    "stage_scores": dict(item.stage_scores),
                }
            )

        return {
            "answer": result.answer,
            "answer_sources": [s.citation_id for s in result.answer_sources],
            "source_count": len(result.sources),
            "evidence": evidence,
            "stage_score_summary": summarize_stage_scores(items),
            "diagnostics": dict(result.diagnostics),
        }

    def _parse_query(self, query: str) -> ParsedSearchQuery:
        if not self.enable_search_dsl:
            return ParsedSearchQuery(
                raw_query=query,
                clean_query=query.strip(),
                filters=SearchFilters(),
                phrase_terms=[],
                free_terms=[query.strip()] if query.strip() else [],
            )
        return parse_search_query(query)

    def _web_allowed_by_filters(self, filters: SearchFilters) -> bool:
        web_types = {SourceType.WEB_SNIPPET, SourceType.WEB_PAGE_TEXT}
        if filters.include_source_types and not (filters.include_source_types & web_types):
            return False
        if web_types.issubset(filters.exclude_source_types):
            return False
        # If user is scoping search tightly to local sections, skip web to avoid noisy results.
        if filters.section_prefixes:
            return False
        return True

    def _web_fallback(
        self,
        query: str,
        *,
        passes: Sequence[QueryPass],
        filters: SearchFilters,
        web_meta: Dict[str, object],
        score_base: float,
        max_results: int,
        top_k: int,
        weights: RetrievalWeights,
        token_boosts: Dict[str, float],
        retrieval_options: RetrievalOptions,
        enable_web_fetch: bool,
        web_fetch_max_pages: int,
        web_fetch_crawl_depth: int,
    ):
        provider = self.web_search_provider
        pass_summaries: list[dict[str, object]] = []

        max_results = max(1, int(max_results))
        # Always do multi-pass web search to preserve "contradiction/boundary" discovery semantics.
        seen_by_url: dict[str, WebSearchResult] = {}
        url_passes: dict[str, set[str]] = defaultdict(set)
        had_error = False

        for qp in passes:
            try:
                rows = provider.search(qp.query, max_results=max_results)
            except Exception as exc:
                had_error = True
                pass_summaries.append({"pass": qp.name, "query": qp.query, "error": f"{type(exc).__name__}"})
                continue
            pass_summaries.append({"pass": qp.name, "query": qp.query, "result_count": len(rows)})
            for row in rows:
                url = str(row.url or "").strip()
                if not url:
                    continue
                url_passes[url].add(qp.name)
                prev = seen_by_url.get(url)
                if prev is None or int(row.rank) < int(prev.rank):
                    seen_by_url[url] = row

        web_rows = sorted(seen_by_url.values(), key=lambda r: (int(r.rank), str(r.url)))
        max_unique = max_results * max(len(passes), 1)
        web_rows = web_rows[:max_unique]

        now_iso = datetime.now(timezone.utc).isoformat()
        web_units: list[EvidenceUnit] = []
        for row in web_rows:
            content = (row.snippet or "").strip() or (row.title or "").strip()
            if not content:
                continue
            url = str(row.url or "").strip()
            unit = EvidenceUnit(
                doc_id=url,
                source_type=SourceType.WEB_SNIPPET,
                content=content,
                section_path=(row.title or "").strip()[:120] or "web",
                char_start=0,
                char_end=len(content),
                timestamp=now_iso,
                confidence=0.62,
                metadata={
                    "provider": row.provider,
                    "url": url,
                    "title": row.title,
                    "web_rank": int(row.rank),
                    "web_passes": sorted(url_passes.get(url, set())),
                },
            )
            web_units.append(unit)

        fetch_meta: dict[str, object] = {"enabled": bool(enable_web_fetch), "used": False, "page_count": 0, "unit_count": 0}
        want_page_text = True
        if SourceType.WEB_PAGE_TEXT in filters.exclude_source_types:
            want_page_text = False
        if filters.include_source_types and SourceType.WEB_PAGE_TEXT not in filters.include_source_types:
            want_page_text = False

        page_units: list[EvidenceUnit] = []
        if enable_web_fetch and want_page_text and web_rows and web_fetch_max_pages > 0:
            max_chunks_per_page = 10
            base_urls = [str(row.url or "").strip() for row in web_rows]
            base_urls = [u for u in base_urls if u]
            base_urls = base_urls[: max(0, int(web_fetch_max_pages))]
            visited: set[str] = set()
            to_visit = list(base_urls)
            for depth in range(max(0, int(web_fetch_crawl_depth)) + 1):
                if not to_visit:
                    break
                batch = [u for u in to_visit if u and u not in visited]
                to_visit = []
                if not batch:
                    continue
                for u in batch:
                    visited.add(u)
                try:
                    pages = self.web_fetch_provider.fetch(batch, max_pages=len(batch))
                    fetch_meta["used"] = True
                    fetch_meta["page_count"] = int(fetch_meta.get("page_count", 0)) + len(pages)
                    next_level: list[str] = []
                    for page in pages:
                        if not page.ok:
                            continue
                        base_url = (page.final_url or page.url or "").strip()
                        if not base_url:
                            continue
                        base_row = seen_by_url.get(page.url) or seen_by_url.get(base_url)
                        rank = int(getattr(base_row, "rank", 0) or 0) if base_row is not None else 0
                        passes_for_url = sorted(url_passes.get(page.url, set()) | url_passes.get(base_url, set()))

                        chunks = chunk_text(page.text, max_chars=900, overlap=120, min_chars=160)
                        if not chunks:
                            continue
                        title = (page.title or (base_row.title if base_row is not None else "") or "").strip()
                        section_base = title[:90] if title else "web_page"

                        for idx, (chunk, start_off, end_off) in enumerate(chunks[:max_chunks_per_page], start=1):
                            page_units.append(
                                EvidenceUnit(
                                    doc_id=base_url,
                                    source_type=SourceType.WEB_PAGE_TEXT,
                                    content=chunk,
                                    section_path=f"{section_base}#chunk{idx}",
                                    char_start=int(start_off),
                                    char_end=int(end_off),
                                    timestamp=now_iso,
                                    confidence=0.6,
                                    metadata={
                                        "provider": getattr(self.web_search_provider, "provider_name", None),
                                        "fetch_provider": getattr(self.web_fetch_provider, "provider_name", None),
                                        "url": base_url,
                                        "title": title,
                                        "status": page.status,
                                        "content_type": page.content_type,
                                        "web_rank": rank,
                                        "web_passes": passes_for_url,
                                        "crawl_depth": depth,
                                    },
                                )
                            )
                        if depth < max(0, int(web_fetch_crawl_depth)):
                            next_level.extend([u for u in page.discovered_links if u not in visited])
                    if depth < max(0, int(web_fetch_crawl_depth)):
                        unique_next: list[str] = []
                        seen_next: set[str] = set()
                        for u in next_level:
                            if u in seen_next:
                                continue
                            seen_next.add(u)
                            unique_next.append(u)
                        remain_budget = max(0, int(web_fetch_max_pages) - len(base_urls))
                        if remain_budget > 0:
                            to_visit = unique_next[:remain_budget]
                except Exception as exc:
                    fetch_meta["used"] = False
                    fetch_meta["error"] = f"{type(exc).__name__}"
                    break
                finally:
                    fetch_meta["unit_count"] = len(page_units)
        if page_units:
            web_units.extend(page_units)

        web_units = apply_search_filters(web_units, filters)
        if not web_units:
            updated = dict(web_meta)
            updated["used"] = False
            updated["result_count"] = len(web_rows)
            updated["reason"] = "no_results" if not web_rows else "filtered_out_by_query_dsl"
            updated["passes"] = pass_summaries
            updated["web_fetch"] = fetch_meta
            return [], updated, web_rows

        bm25_index = build_bm25_index([u.content for u in web_units])

        ann_index: ANNIndex | None = None
        dense_meta: dict[str, object] = {"enabled": False, "reason": "disabled"}
        try:
            import numpy as np  # type: ignore

            if hasattr(np, "linalg") and callable(getattr(np.linalg, "norm", None)):
                vectors = encode_texts([u.content for u in web_units], model_name=self.embedding_model, text_type="passage")
                ann_index = build_index(vectors, backend="exact")
                dense_meta = {"enabled": True, "backend": "exact", "dim": int(getattr(ann_index, "dim", 0))}
        except Exception as exc:  # pragma: no cover - optional path
            ann_index = None
            dense_meta = {"enabled": False, "reason": f"error:{type(exc).__name__}"}

        hits: list[ScoredEvidence] = []
        for qp in passes:
            hits.extend(
                retrieve(
                    qp,
                    web_units,
                    top_k=top_k,
                    bm25_index=bm25_index,
                    ann_index=ann_index,
                    embedding_model=self.embedding_model,
                    weights=weights,
                    token_boosts=token_boosts,
                    options=retrieval_options,
                )
            )

        adjusted: list[ScoredEvidence] = []
        for item in hits:
            rank = int(item.evidence.metadata.get("web_rank") or 0) or 999
            pass_count = len(item.evidence.metadata.get("web_passes") or [])
            rank_prior = max(0.55, 1.0 - (rank - 1) * 0.05)
            pass_prior = 1.0 + min(0.18, max(0.0, pass_count - 1) * 0.06)
            raw_score = float(item.score)
            final_score = raw_score * rank_prior * pass_prior
            stage = dict(item.stage_scores)
            stage.setdefault("hybrid_raw", float(stage.get("hybrid", raw_score)))
            stage["web_rank"] = float(rank)
            stage["web_rank_prior"] = float(rank_prior)
            stage["web_pass_count"] = float(pass_count)
            stage["web_pass_prior"] = float(pass_prior)
            stage["hybrid"] = final_score
            adjusted.append(
                ScoredEvidence(
                    evidence=item.evidence,
                    score=final_score,
                    verdict=item.verdict,
                    why_it_matches=f"{item.why_it_matches}, web_rank={rank}",
                    stage_scores=stage,
                    source=item.source or build_source_citation(item.evidence),
                )
            )

        if not adjusted and web_units:
            # If hybrid scoring yields nothing (e.g., query language mismatch), still surface web results
            # using provider rank-based priors so users can see external evidence exists.
            fallback_hits: list[ScoredEvidence] = []
            base = max(0.0, min(float(score_base), 1.0))
            for unit in web_units:
                rank = int(unit.metadata.get("web_rank") or 0) or 999
                score = max(0.0, base - (rank - 1) * 0.06)
                fallback_hits.append(
                    ScoredEvidence(
                        evidence=unit,
                        score=score,
                        verdict=Verdict.UNCERTAIN,
                        why_it_matches=f"web_fallback:{unit.metadata.get('provider')}:rank={rank}",
                        stage_scores={"web_fallback": score, "web_rank": float(rank)},
                        source=build_source_citation(unit),
                    )
                )
            adjusted = fallback_hits

        updated = dict(web_meta)
        updated["used"] = bool(web_rows)
        updated["result_count"] = len(web_rows)
        updated["unit_count"] = len(web_units)
        updated["candidate_count"] = len(adjusted)
        updated["passes"] = pass_summaries
        updated["dense_index"] = dense_meta
        updated["web_fetch"] = fetch_meta
        if adjusted:
            updated["reason"] = "applied"
        elif had_error:
            updated["reason"] = "error"
        else:
            updated["reason"] = "no_results"

        return adjusted, updated, web_rows

    def _merge_with_web(
        self,
        local_hits: Sequence[ScoredEvidence],
        web_hits: Sequence[ScoredEvidence],
        *,
        top_n: int,
    ) -> List[ScoredEvidence]:
        merged = list(local_hits) + list(web_hits)
        merged.sort(key=lambda item: item.score, reverse=True)
        return self._dedupe_candidates(merged)[: max(top_n, 1)]

    def _format_answer_for_mode(
        self,
        mode: SearchMode,
        *,
        answer: str,
        best: Sequence[ScoredEvidence],
        supports: Sequence[ScoredEvidence],
        contradictions: Sequence[ScoredEvidence],
    ) -> str:
        if mode == SearchMode.SNIPER:
            return answer
        if mode == SearchMode.RUMOR:
            parts = ["[UNVERIFIED] 다양한 주장/관측을 요약합니다 (진위 미확인)."]
            if supports:
                parts.append("지지 근거:")
                parts.extend(f"- {item.evidence.content}" for item in supports[:3])
            if contradictions:
                parts.append("반대/예외 근거:")
                parts.extend(f"- {item.evidence.content}" for item in contradictions[:3])
            if not supports and not contradictions and best:
                parts.append("참고 근거:")
                parts.extend(f"- {item.evidence.content}" for item in best[:3])
            return "\n".join(parts)
        if mode in {SearchMode.REPORTER, SearchMode.FBI}:
            parts = ["요약:", f"- {answer}"]
            if supports:
                parts.append("주요 근거:")
                parts.extend(f"- {item.evidence.content}" for item in supports[:3])
            if contradictions:
                parts.append("주의/반례:")
                parts.extend(f"- {item.evidence.content}" for item in contradictions[:3])
            return "\n".join(parts)
        if mode == SearchMode.COLLECTION:
            parts = ["자료/링크 후보:"]
            for item in best[: min(len(best), 6)]:
                url = item.evidence.metadata.get("url") or item.evidence.doc_id
                title = item.evidence.metadata.get("title") or item.evidence.section_path
                parts.append(f"- {title} ({url})")
            return "\n".join(parts) if len(parts) > 1 else answer
        return answer

    def _filter_trusted(self, items: Sequence[ScoredEvidence]) -> List[ScoredEvidence]:
        if not self.trusted_domains:
            return [
                item
                for item in items
                if item.evidence.source_type not in {SourceType.WEB_SNIPPET, SourceType.WEB_PAGE_TEXT}
            ]

        trusted: list[ScoredEvidence] = []
        for item in items:
            if item.evidence.source_type not in {SourceType.WEB_SNIPPET, SourceType.WEB_PAGE_TEXT}:
                trusted.append(item)
                continue
            url = str(item.evidence.metadata.get("url") or item.evidence.doc_id)
            domain = urlparse(url).netloc.lower()
            if domain and any(domain == td or domain.endswith("." + td) for td in self.trusted_domains):
                trusted.append(item)
        return trusted

    def _persist_to_library(
        self,
        *,
        mode: str,
        parsed_query: ParsedSearchQuery,
        result: SearchResult,
        extra: Dict[str, object],
    ) -> str | None:
        if self._library is None:
            return None
        session_id = self._library.save_session(
            mode=mode,
            raw_query=parsed_query.raw_query,
            clean_query=parsed_query.clean_query,
            filters=filter_to_dict(parsed_query.filters),
            result=result,
            extra=extra,
        )
        web_units = [
            item.evidence
            for item in [*result.evidence, *result.contradictions]
            if item.evidence.source_type == SourceType.WEB_SNIPPET
        ]
        if web_units:
            self._library.append_evidence_units(web_units, tag="web_snippets")
        web_pages = [
            item.evidence
            for item in [*result.evidence, *result.contradictions]
            if item.evidence.source_type == SourceType.WEB_PAGE_TEXT
        ]
        if web_pages:
            self._library.append_evidence_units(web_pages, tag="web_pages")
        if "osint" in result.diagnostics and isinstance(result.diagnostics.get("osint"), dict):
            self._library.save_osint_artifacts(session_id, artifacts=result.diagnostics["osint"])
        return session_id

    @staticmethod
    def _shard_for_doc(doc_id: str, shard_count: int) -> int:
        digest = hashlib.sha1(doc_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], byteorder="big", signed=False) % shard_count
