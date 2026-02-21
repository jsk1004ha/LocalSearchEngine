from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SourceType(str, Enum):
    TEXT_SENTENCE = "text_sentence"
    TABLE_CELL = "table_cell"
    CAPTION = "caption"
    OCR_TEXT = "ocr_text"


@dataclass(frozen=True)
class EvidenceUnit:
    doc_id: str
    source_type: SourceType
    content: str
    section_path: str
    char_start: int
    char_end: int
    timestamp: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class Verdict(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class ScoredEvidence:
    evidence: EvidenceUnit
    score: float
    verdict: Verdict
    why_it_matches: str
    stage_scores: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    answer: str
    evidence: List[ScoredEvidence]
    contradictions: List[ScoredEvidence]
    diagnostics: Dict[str, Any]
    reranker_model: Optional[str] = None
    reranker_version: Optional[str] = None
    build_id: Optional[str] = None
