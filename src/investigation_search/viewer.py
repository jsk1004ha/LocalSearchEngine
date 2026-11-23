from __future__ import annotations

import re
from typing import Iterable, Sequence

from .analyzer import unique_tokens
from .schema import ScoredEvidence, SearchResult


def highlight_text(text: str, query: str, *, max_len: int = 220) -> str:
    tokens = sorted(unique_tokens(query, mode="viewer", include_char_ngrams=False), key=len, reverse=True)
    if not tokens:
        snippet = text.strip()
        return _trim(snippet, max_len=max_len)

    highlighted = text
    for token in tokens:
        if len(token) < 2:
            continue
        pattern = re.compile(re.escape(token), re.IGNORECASE)
        highlighted = pattern.sub(lambda m: f"[[{m.group(0)}]]", highlighted)
    return _trim(highlighted.strip(), max_len=max_len)


def render_result_text(
    result: SearchResult,
    *,
    query: str,
    max_items: int = 5,
    include_diagnostics: bool = True,
) -> str:
    lines: list[str] = []
    lines.append(f"Answer: {result.answer}")
    if result.answer_sources:
        lines.append("Answer Sources: " + ", ".join(src.citation_id for src in result.answer_sources))
    else:
        lines.append("Answer Sources: none")

    evidence_items = list(result.evidence[: max(max_items, 0)])
    if not evidence_items:
        lines.append("Evidence: none")
    else:
        lines.append("Evidence:")
        for idx, item in enumerate(evidence_items, start=1):
            citation = item.source.citation_id if item.source is not None else "unknown"
            snippet = highlight_text(item.evidence.content, query)
            lines.append(
                f"{idx}. [{item.verdict.value}] score={item.score:.4f} citation={citation}\n"
                f"   {snippet}\n"
                f"   why={item.why_it_matches}"
            )

    if include_diagnostics:
        lines.append("Diagnostics:")
        lines.append(f"{result.diagnostics}")
    return "\n".join(lines)


def summarize_stage_scores(
    evidence_items: Sequence[ScoredEvidence] | Iterable[ScoredEvidence],
) -> dict:
    seq = list(evidence_items)
    if not seq:
        return {"count": 0}

    keys: set[str] = set()
    for item in seq:
        keys.update(item.stage_scores.keys())

    summary: dict[str, float] = {"count": float(len(seq))}
    for key in sorted(keys):
        values = [float(item.stage_scores.get(key, 0.0)) for item in seq]
        summary[f"{key}_avg"] = sum(values) / len(values)
    return summary


def _trim(text: str, *, max_len: int) -> str:
    if max_len <= 0 or len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
