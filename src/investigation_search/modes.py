from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from .analyzer import detect_language
from .retrieval import QueryPass, build_passes


class SearchMode(str, Enum):
    INVESTIGATION = "investigation"
    REPORTER = "reporter"
    FBI = "fbi"
    COLLECTION = "collection"
    SNIPER = "sniper"
    RUMOR = "rumor"
    LIBRARY = "library"


@dataclass(frozen=True)
class ModeProfile:
    mode: SearchMode
    description: str
    top_k_per_pass_min: int
    max_rerank_candidates: int
    always_web_search: bool
    enable_web_fallback: bool
    web_max_results: int
    web_score_base: float
    enable_web_fetch: bool
    web_fetch_max_pages: int
    retrieval_options_overrides: Mapping[str, object]
    include_only_trusted_sources: bool = False


_MODE_ALIASES = {
    # English
    "investigation": SearchMode.INVESTIGATION,
    "reporter": SearchMode.REPORTER,
    "journalist": SearchMode.REPORTER,
    "fbi": SearchMode.FBI,
    "osint": SearchMode.FBI,
    "collection": SearchMode.COLLECTION,
    "assets": SearchMode.COLLECTION,
    "sniper": SearchMode.SNIPER,
    "rumor": SearchMode.RUMOR,
    "gossip": SearchMode.RUMOR,
    "library": SearchMode.LIBRARY,
    # Korean
    "조사": SearchMode.INVESTIGATION,
    "조사모드": SearchMode.INVESTIGATION,
    "기자": SearchMode.REPORTER,
    "기자모드": SearchMode.REPORTER,
    "자료": SearchMode.COLLECTION,
    "자료수집": SearchMode.COLLECTION,
    "자료수집모드": SearchMode.COLLECTION,
    "스나이퍼": SearchMode.SNIPER,
    "스나이퍼모드": SearchMode.SNIPER,
    "찌라시": SearchMode.RUMOR,
    "찌라시모드": SearchMode.RUMOR,
    "도서관": SearchMode.LIBRARY,
    "도서관모드": SearchMode.LIBRARY,
    "fbi모드": SearchMode.FBI,
    "osint모드": SearchMode.FBI,
}


def parse_mode(mode: str | SearchMode | None) -> SearchMode:
    if mode is None:
        return SearchMode.INVESTIGATION
    if isinstance(mode, SearchMode):
        return mode
    normalized = str(mode).strip().lower()
    normalized = re.sub(r"\s+", "", normalized)
    return _MODE_ALIASES.get(normalized, SearchMode.INVESTIGATION)


def profile_for_mode(mode: SearchMode) -> ModeProfile:
    if mode == SearchMode.SNIPER:
        return ModeProfile(
            mode=mode,
            description="Only the single best evidence with minimal context.",
            top_k_per_pass_min=1,
            max_rerank_candidates=12,
            always_web_search=False,
            enable_web_fallback=False,
            web_max_results=0,
            web_score_base=0.0,
            enable_web_fetch=False,
            web_fetch_max_pages=0,
            retrieval_options_overrides={},
        )

    if mode == SearchMode.REPORTER:
        return ModeProfile(
            mode=mode,
            description="Broad and detailed collection of relevant information.",
            top_k_per_pass_min=8,
            max_rerank_candidates=64,
            always_web_search=True,
            enable_web_fallback=True,
            web_max_results=6,
            web_score_base=0.55,
            enable_web_fetch=True,
            web_fetch_max_pages=4,
            retrieval_options_overrides={"enable_recency_boost": True},
        )

    if mode == SearchMode.FBI:
        return ModeProfile(
            mode=mode,
            description="OSINT-focused deep collection + link/timeline artifacts (public sources only).",
            top_k_per_pass_min=10,
            max_rerank_candidates=80,
            always_web_search=True,
            enable_web_fallback=True,
            web_max_results=10,
            web_score_base=0.65,
            enable_web_fetch=True,
            web_fetch_max_pages=6,
            retrieval_options_overrides={"enable_recency_boost": True},
        )

    if mode == SearchMode.COLLECTION:
        return ModeProfile(
            mode=mode,
            description="Asset-first search (papers, datasets, repos, images).",
            top_k_per_pass_min=8,
            max_rerank_candidates=32,
            always_web_search=True,
            enable_web_fallback=True,
            web_max_results=12,
            web_score_base=0.6,
            enable_web_fetch=True,
            web_fetch_max_pages=5,
            retrieval_options_overrides={},
        )

    if mode == SearchMode.RUMOR:
        return ModeProfile(
            mode=mode,
            description="Collect diverse opinions regardless of veracity; clearly labeled as unverified.",
            top_k_per_pass_min=10,
            max_rerank_candidates=48,
            always_web_search=True,
            enable_web_fallback=True,
            web_max_results=10,
            web_score_base=0.5,
            enable_web_fetch=True,
            web_fetch_max_pages=4,
            retrieval_options_overrides={},
        )

    if mode == SearchMode.LIBRARY:
        # Prefer trusted sources only: default is local corpus, web excluded unless explicitly allowed.
        return ModeProfile(
            mode=mode,
            description="Trusted-only aggregation (local corpus by default).",
            top_k_per_pass_min=6,
            max_rerank_candidates=32,
            always_web_search=False,
            enable_web_fallback=False,
            web_max_results=0,
            web_score_base=0.0,
            enable_web_fetch=False,
            web_fetch_max_pages=0,
            retrieval_options_overrides={"min_ocr_confidence": 0.55},
            include_only_trusted_sources=True,
        )

    return ModeProfile(
        mode=SearchMode.INVESTIGATION,
        description="Investigation mode (answer + evidence + contradictions + diagnostics).",
        top_k_per_pass_min=5,
        max_rerank_candidates=32,
        always_web_search=False,
        enable_web_fallback=True,
        web_max_results=3,
        web_score_base=0.25,
        enable_web_fetch=True,
        web_fetch_max_pages=2,
        retrieval_options_overrides={},
    )


def build_passes_for_mode(query: str, mode: SearchMode) -> Sequence[QueryPass]:
    base = list(build_passes(query))
    lang = detect_language(query)

    def _add(name: str, suffix: str) -> None:
        base.append(QueryPass(name=name, query=f"{query} {suffix}".strip()))

    if mode in {SearchMode.REPORTER, SearchMode.FBI, SearchMode.COLLECTION, SearchMode.RUMOR}:
        if lang == "en":
            _add("pass_d_background", "overview background context history")
            _add("pass_e_sources", "report analysis data evidence")
        else:
            _add("pass_d_background", "개요 배경 맥락 경과 정리")
            _add("pass_e_sources", "리포트 분석 데이터 근거 출처")

    if mode == SearchMode.COLLECTION:
        if lang == "en":
            _add("pass_f_assets", "dataset github arxiv pdf repository figures")
        else:
            _add("pass_f_assets", "데이터셋 깃허브 arxiv 논문 pdf 리포지토리 이미지")

    if mode == SearchMode.FBI:
        if lang == "en":
            _add("pass_f_osint", "metadata robots sitemap cache archive")
            _add("pass_g_security_docs", "security advisory vulnerability disclosure cve")
        else:
            _add("pass_f_osint", "메타데이터 robots sitemap 캐시 아카이브")
            _add("pass_g_security_docs", "보안 권고 취약점 공지 cve")

    return tuple(base)
