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
class SourceCitation:
    citation_id: str
    doc_id: str
    section_path: str
    char_start: int
    char_end: int
    source_type: SourceType
    timestamp: str


def build_source_citation(unit: EvidenceUnit) -> SourceCitation:
    citation_id = f"{unit.doc_id}:{unit.section_path}:{unit.char_start}-{unit.char_end}"
    return SourceCitation(
        citation_id=citation_id,
        doc_id=unit.doc_id,
        section_path=unit.section_path,
        char_start=unit.char_start,
        char_end=unit.char_end,
        source_type=unit.source_type,
        timestamp=unit.timestamp,
    )


@dataclass(frozen=True)
class ScoredEvidence:
    evidence: EvidenceUnit
    score: float
    verdict: Verdict
    why_it_matches: str
    stage_scores: Dict[str, float] = field(default_factory=dict)
    source: Optional[SourceCitation] = None


@dataclass(frozen=True)
class SearchResult:
    answer: str
    evidence: List[ScoredEvidence]
    contradictions: List[ScoredEvidence]
    diagnostics: Dict[str, Any]
    reranker_model: Optional[str] = None
    reranker_version: Optional[str] = None
    build_id: Optional[str] = None
    answer_sources: List[SourceCitation] = field(default_factory=list)
    sources: List[SourceCitation] = field(default_factory=list)
