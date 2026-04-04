from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence

from .graph_export import GraphExporter, GraphQuerier, build_graph_exporter, build_graph_querier
from .schema import ScoredEvidence


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _command_available(command_name: str) -> bool:
    return shutil.which(command_name) is not None


class TextRedactor(Protocol):
    provider_name: str

    def redact(self, text: str) -> str:
        """Return a best-effort redacted version of text."""


@dataclass(frozen=True)
class OptionalToolStatus:
    name: str
    category: str
    available: bool
    enabled: bool = False
    detail: str | None = None


@dataclass(frozen=True)
class NoopTextRedactor:
    provider_name: str = "off"

    def redact(self, text: str) -> str:
        return text


class PresidioTextRedactor:
    provider_name = "presidio"

    def __init__(self, *, language: str = "en", entities: Sequence[str] | None = None):
        try:
            from presidio_analyzer import AnalyzerEngine  # type: ignore[import-not-found]
            from presidio_anonymizer import AnonymizerEngine  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("presidio_not_installed") from exc

        self.language = str(language or "en")
        self.entities = tuple(str(entity) for entity in (entities or ()))
        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()

    def redact(self, text: str) -> str:
        clean = str(text or "")
        if not clean.strip():
            return clean
        try:
            analyzer_results = self._analyzer.analyze(
                text=clean,
                language=self.language,
                entities=list(self.entities) or None,
            )
        except Exception:
            return clean
        if not analyzer_results:
            return clean
        try:
            return str(self._anonymizer.anonymize(text=clean, analyzer_results=analyzer_results).text)
        except Exception:
            return clean


@dataclass
class RuntimeTooling:
    web_fetch_backend: str = "default"
    pii_redaction_backend: str = "off"
    graph_backend: str = "off"
    text_redactor: TextRedactor = field(default_factory=NoopTextRedactor)
    graph_exporter: GraphExporter = field(default_factory=build_graph_exporter)
    graph_querier: GraphQuerier = field(default_factory=build_graph_querier)
    available_tools: tuple[OptionalToolStatus, ...] = ()

    def redact_text(self, text: str) -> str:
        return self.text_redactor.redact(text)

    def build_graph_preview(
        self,
        *,
        query: str,
        evidence_items: Sequence[ScoredEvidence] | Iterable[ScoredEvidence],
        contradictions: Sequence[ScoredEvidence] | Iterable[ScoredEvidence],
    ) -> dict[str, object] | None:
        backend = str(self.graph_backend or "off")
        if backend in {"", "off", "none"}:
            return None

        nodes: list[dict[str, object]] = [{"id": "query", "type": "query", "label": str(query or "").strip()}]
        edges: list[dict[str, object]] = []
        seen_nodes = {"query"}

        def _add_item(prefix: str, item: ScoredEvidence, relation: str) -> None:
            source_id = item.evidence.doc_id
            evidence_id = f"{prefix}:{source_id}:{item.evidence.char_start}:{item.evidence.char_end}"
            if source_id not in seen_nodes:
                seen_nodes.add(source_id)
                nodes.append(
                    {
                        "id": source_id,
                        "type": "source",
                        "label": source_id,
                        "source_type": item.evidence.source_type.value,
                    }
                )
            if evidence_id not in seen_nodes:
                seen_nodes.add(evidence_id)
                nodes.append(
                    {
                        "id": evidence_id,
                        "type": "evidence",
                        "label": item.evidence.content[:140],
                        "score": float(item.score),
                        "verdict": item.verdict.value,
                    }
                )
            edges.append({"from": "query", "to": evidence_id, "type": relation})
            edges.append({"from": evidence_id, "to": source_id, "type": "from_source"})

        for item in evidence_items:
            _add_item("evidence", item, "supports_candidate")
        for item in contradictions:
            _add_item("contradiction", item, "contradicts_candidate")

        return {
            "backend": backend,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes,
            "edges": edges,
        }

    def export_graph(
        self,
        *,
        query: str,
        evidence_items: Sequence[ScoredEvidence] | Iterable[ScoredEvidence],
        contradictions: Sequence[ScoredEvidence] | Iterable[ScoredEvidence],
    ) -> dict[str, object]:
        return self.graph_exporter.export_search_result(
            query=query,
            evidence_items=evidence_items,
            contradictions=contradictions,
        )

    def query_related_graph(
        self,
        *,
        query: str,
        source_ids: Sequence[str] | Iterable[str],
        limit: int = 5,
    ) -> dict[str, object]:
        return self.graph_querier.query_related_sources(query=query, source_ids=source_ids, limit=limit)

    def to_diagnostics(self) -> dict[str, object]:
        return {
            "web_fetch_backend": self.web_fetch_backend,
            "pii_redaction_backend": self.pii_redaction_backend,
            "graph_backend": self.graph_backend,
            "active_redactor": getattr(self.text_redactor, "provider_name", "off"),
            "available_tools": [
                {
                    "name": tool.name,
                    "category": tool.category,
                    "available": bool(tool.available),
                    "enabled": bool(tool.enabled),
                    "detail": tool.detail,
                }
                for tool in self.available_tools
            ],
        }


def detect_optional_tools(
    *,
    web_fetch_backend: str = "default",
    pii_redaction_backend: str = "off",
    graph_backend: str = "off",
) -> tuple[OptionalToolStatus, ...]:
    requested_fetch = str(web_fetch_backend or "default")
    requested_redaction = str(pii_redaction_backend or "off")
    requested_graph = str(graph_backend or "off")

    crawl4ai_available = _module_available("crawl4ai")
    playwright_available = _module_available("playwright")
    presidio_available = _module_available("presidio_analyzer") and _module_available("presidio_anonymizer")
    neo4j_available = _module_available("neo4j")

    return (
        OptionalToolStatus(
            name="crawl4ai",
            category="web_fetch",
            available=crawl4ai_available,
            enabled=requested_fetch in {"crawl4ai", "auto"} and crawl4ai_available,
            detail="LLM-friendly markdown/structured crawl",
        ),
        OptionalToolStatus(
            name="playwright",
            category="web_fetch",
            available=playwright_available,
            enabled=requested_fetch == "playwright" and playwright_available,
            detail="dynamic page rendering",
        ),
        OptionalToolStatus(
            name="presidio",
            category="redaction",
            available=presidio_available,
            enabled=requested_redaction in {"presidio", "auto"} and presidio_available,
            detail="PII redaction",
        ),
        OptionalToolStatus(
            name="neo4j",
            category="graph",
            available=neo4j_available or _command_available("neo4j"),
            enabled=requested_graph in {"neo4j", "auto"} and (neo4j_available or _command_available("neo4j")),
            detail="graph export/analysis target",
        ),
    )


def build_runtime_tooling(
    *,
    web_fetch_backend: str = "default",
    pii_redaction_backend: str = "off",
    graph_backend: str = "off",
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
    neo4j_database: str | None = None,
) -> RuntimeTooling:
    statuses = detect_optional_tools(
        web_fetch_backend=web_fetch_backend,
        pii_redaction_backend=pii_redaction_backend,
        graph_backend=graph_backend,
    )

    redactor: TextRedactor = NoopTextRedactor()
    redaction_backend = str(pii_redaction_backend or "off")
    if redaction_backend == "auto":
        redaction_backend = "presidio" if any(t.name == "presidio" and t.available for t in statuses) else "off"
    if redaction_backend == "presidio":
        redactor = PresidioTextRedactor()

    fetch_backend = str(web_fetch_backend or "default")
    if fetch_backend == "auto":
        if any(t.name == "crawl4ai" and t.available for t in statuses):
            fetch_backend = "crawl4ai"
        elif any(t.name == "playwright" and t.available for t in statuses):
            fetch_backend = "playwright"
        else:
            fetch_backend = "default"

    resolved_graph_backend = str(graph_backend or "off")
    if resolved_graph_backend == "auto":
        resolved_graph_backend = "neo4j" if any(t.name == "neo4j" and t.available for t in statuses) else "off"

    return RuntimeTooling(
        web_fetch_backend=fetch_backend,
        pii_redaction_backend=redaction_backend,
        graph_backend=resolved_graph_backend,
        text_redactor=redactor,
        graph_exporter=build_graph_exporter(
            backend=resolved_graph_backend,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            neo4j_database=neo4j_database,
        ),
        graph_querier=build_graph_querier(
            backend=resolved_graph_backend,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            neo4j_database=neo4j_database,
        ),
        available_tools=statuses,
    )
