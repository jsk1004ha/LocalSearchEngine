from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence
from urllib.parse import urlparse

from .websearch import WebSearchResult


_DATE_PATTERNS = (
    re.compile(r"\b(20\d{2})[-/](0?\d{1,2})[-/](0?\d{1,2})\b"),
    re.compile(r"\b(20\d{2})\.(0?\d{1,2})\.(0?\d{1,2})\b"),
    re.compile(r"\b(20\d{2})년\s*(0?\d{1,2})월\s*(0?\d{1,2})일\b"),
)


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    node_type: str
    label: str


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    edge_type: str


def build_osint_graph(query: str, results: Sequence[WebSearchResult] | Iterable[WebSearchResult]) -> dict:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    seen_nodes: set[str] = set()

    query_id = f"q:{_short_hash(query)}"
    nodes.append(GraphNode(node_id=query_id, node_type="query", label=query))
    seen_nodes.add(query_id)

    for row in results:
        url = row.url
        url_id = f"url:{_short_hash(url)}"
        if url_id not in seen_nodes:
            nodes.append(GraphNode(node_id=url_id, node_type="url", label=url))
            seen_nodes.add(url_id)
        edges.append(GraphEdge(source=query_id, target=url_id, edge_type="query_hits"))

        domain = urlparse(url).netloc.lower()
        if domain:
            domain_id = f"domain:{domain}"
            if domain_id not in seen_nodes:
                nodes.append(GraphNode(node_id=domain_id, node_type="domain", label=domain))
                seen_nodes.add(domain_id)
            edges.append(GraphEdge(source=url_id, target=domain_id, edge_type="hosted_on"))

    return {
        "nodes": [node.__dict__ for node in nodes],
        "edges": [edge.__dict__ for edge in edges],
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


def extract_timeline(results: Sequence[WebSearchResult] | Iterable[WebSearchResult], *, limit: int = 30) -> list[dict]:
    events: list[dict] = []
    for row in results:
        text = f"{row.title} {row.snippet}".strip()
        for date in _extract_dates(text):
            events.append(
                {
                    "date": date,
                    "url": row.url,
                    "title": row.title,
                    "snippet": row.snippet,
                    "rank": row.rank,
                }
            )

    events.sort(key=lambda e: (e["date"], e["rank"]))
    return events[: max(0, int(limit))]


def _extract_dates(text: str) -> List[str]:
    found: set[str] = set()
    for pattern in _DATE_PATTERNS:
        for m in pattern.finditer(text):
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 1 <= mo <= 12 and 1 <= d <= 31:
                found.add(f"{y:04d}-{mo:02d}-{d:02d}")
    return sorted(found)


def _short_hash(text: str) -> str:
    # Cheap stable id without importing hashlib here (caller can store raw too).
    value = 0
    for ch in text:
        value = (value * 131 + ord(ch)) & 0xFFFFFFFF
    return f"{value:08x}"

