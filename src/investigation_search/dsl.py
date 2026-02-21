from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, List, Sequence, Set

from .schema import EvidenceUnit, SourceType


_TOKEN_RE = re.compile(r'"[^"]*"|\S+')

_SOURCE_ALIASES = {
    "text": SourceType.TEXT_SENTENCE,
    "sentence": SourceType.TEXT_SENTENCE,
    "text_sentence": SourceType.TEXT_SENTENCE,
    "table": SourceType.TABLE_CELL,
    "table_cell": SourceType.TABLE_CELL,
    "caption": SourceType.CAPTION,
    "ocr": SourceType.OCR_TEXT,
    "ocr_text": SourceType.OCR_TEXT,
}


@dataclass(frozen=True)
class SearchFilters:
    doc_ids: Set[str] = field(default_factory=set)
    include_source_types: Set[SourceType] = field(default_factory=set)
    exclude_source_types: Set[SourceType] = field(default_factory=set)
    section_prefixes: Set[str] = field(default_factory=set)
    after: str | None = None
    before: str | None = None


@dataclass(frozen=True)
class ParsedSearchQuery:
    raw_query: str
    clean_query: str
    filters: SearchFilters
    phrase_terms: Sequence[str]
    free_terms: Sequence[str]


def parse_search_query(query: str) -> ParsedSearchQuery:
    tokens = _tokenize_query(query)
    phrase_terms = [m.group(1) for m in re.finditer(r'"([^"]+)"', query)]
    free_terms: list[str] = []

    doc_ids: set[str] = set()
    include_sources: set[SourceType] = set()
    exclude_sources: set[SourceType] = set()
    section_prefixes: set[str] = set()
    after: str | None = None
    before: str | None = None

    for token in tokens:
        raw = token
        lowered = raw.lower()

        if lowered.startswith("doc:"):
            value = raw.split(":", 1)[1].strip()
            if value:
                doc_ids.add(value)
            continue
        if lowered.startswith("source:"):
            source = _parse_source(raw.split(":", 1)[1])
            if source is not None:
                include_sources.add(source)
            continue
        if lowered.startswith("-source:"):
            source = _parse_source(raw.split(":", 1)[1])
            if source is not None:
                exclude_sources.add(source)
            continue
        if lowered.startswith("section:"):
            value = raw.split(":", 1)[1].strip()
            if value:
                section_prefixes.add(value)
            continue
        if lowered.startswith("after:"):
            value = raw.split(":", 1)[1].strip()
            if value:
                after = value
            continue
        if lowered.startswith("before:"):
            value = raw.split(":", 1)[1].strip()
            if value:
                before = value
            continue

        free_terms.append(raw)

    clean_terms = list(free_terms)
    clean_query = " ".join(term for term in clean_terms if term).strip()
    if not clean_query:
        clean_query = " ".join(term for term in phrase_terms if term).strip() or query.strip()

    filters = SearchFilters(
        doc_ids=doc_ids,
        include_source_types=include_sources,
        exclude_source_types=exclude_sources,
        section_prefixes=section_prefixes,
        after=after,
        before=before,
    )
    return ParsedSearchQuery(
        raw_query=query,
        clean_query=clean_query,
        filters=filters,
        phrase_terms=phrase_terms,
        free_terms=free_terms,
    )


def apply_search_filters(
    units: Iterable[EvidenceUnit],
    filters: SearchFilters,
) -> List[EvidenceUnit]:
    out: list[EvidenceUnit] = []
    after_dt = _parse_dt(filters.after)
    before_dt = _parse_dt(filters.before)

    for unit in units:
        if filters.doc_ids and unit.doc_id not in filters.doc_ids:
            continue
        if filters.include_source_types and unit.source_type not in filters.include_source_types:
            continue
        if unit.source_type in filters.exclude_source_types:
            continue
        if filters.section_prefixes and not any(unit.section_path.startswith(prefix) for prefix in filters.section_prefixes):
            continue

        unit_dt = _parse_dt(unit.timestamp)
        if after_dt is not None and unit_dt is not None and unit_dt < after_dt:
            continue
        if before_dt is not None and unit_dt is not None and unit_dt > before_dt:
            continue

        out.append(unit)
    return out


def filter_to_dict(filters: SearchFilters) -> dict:
    return {
        "doc_ids": sorted(filters.doc_ids),
        "include_source_types": sorted(source.value for source in filters.include_source_types),
        "exclude_source_types": sorted(source.value for source in filters.exclude_source_types),
        "section_prefixes": sorted(filters.section_prefixes),
        "after": filters.after,
        "before": filters.before,
    }


def _parse_source(raw: str) -> SourceType | None:
    normalized = raw.strip().strip('"').strip("'").lower()
    if not normalized:
        return None
    if normalized in _SOURCE_ALIASES:
        return _SOURCE_ALIASES[normalized]
    try:
        return SourceType(normalized)
    except ValueError:
        return None


def _parse_dt(raw: str | None) -> datetime | None:
    if not raw:
        return None
    value = raw.strip()
    if not value:
        return None

    candidates = [value]
    if len(value) == 10 and value[4] == "-" and value[7] == "-":
        candidates.append(f"{value}T00:00:00+00:00")

    for candidate in candidates:
        normalized = candidate.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(normalized)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def _tokenize_query(query: str) -> list[str]:
    try:
        return shlex.split(query, posix=True)
    except ValueError:
        return _TOKEN_RE.findall(query)
