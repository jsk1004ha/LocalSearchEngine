from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Protocol, Sequence

from .schema import ScoredEvidence


class GraphExporter(Protocol):
    backend_name: str

    def export_search_result(
        self,
        *,
        query: str,
        evidence_items: Sequence[ScoredEvidence] | Iterable[ScoredEvidence],
        contradictions: Sequence[ScoredEvidence] | Iterable[ScoredEvidence],
    ) -> dict[str, object]:
        """Export a search result graph and return export metadata."""


class GraphQuerier(Protocol):
    backend_name: str

    def query_related_sources(
        self,
        *,
        query: str,
        source_ids: Sequence[str] | Iterable[str],
        limit: int = 5,
    ) -> dict[str, object]:
        """Return related graph context for current search results."""


@dataclass(frozen=True)
class NoopGraphExporter:
    backend_name: str = "off"

    def export_search_result(
        self,
        *,
        query: str,
        evidence_items: Sequence[ScoredEvidence] | Iterable[ScoredEvidence],
        contradictions: Sequence[ScoredEvidence] | Iterable[ScoredEvidence],
    ) -> dict[str, object]:
        return {"backend": self.backend_name, "exported": False, "reason": "disabled"}


@dataclass(frozen=True)
class NoopGraphQuerier:
    backend_name: str = "off"

    def query_related_sources(
        self,
        *,
        query: str,
        source_ids: Sequence[str] | Iterable[str],
        limit: int = 5,
    ) -> dict[str, object]:
        return {"backend": self.backend_name, "queried": False, "reason": "disabled", "items": []}


@dataclass(frozen=True)
class Neo4jGraphExporter:
    uri: str
    username: str
    password: str
    database: str | None = None
    backend_name: str = "neo4j"

    def export_search_result(
        self,
        *,
        query: str,
        evidence_items: Sequence[ScoredEvidence] | Iterable[ScoredEvidence],
        contradictions: Sequence[ScoredEvidence] | Iterable[ScoredEvidence],
    ) -> dict[str, object]:
        try:
            from neo4j import GraphDatabase  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("neo4j_not_installed") from exc

        query_text = str(query or "").strip()
        created_at = datetime.now(timezone.utc).isoformat()
        query_id = _stable_id("query", query_text, created_at)

        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        exported_evidence = 0
        exported_contradictions = 0
        try:
            driver.verify_connectivity()
            with driver.session(database=self.database) as session:
                session.run(
                    """
                    MERGE (q:SearchQuery {id: $query_id})
                    SET q.text = $text, q.created_at = $created_at
                    """,
                    query_id=query_id,
                    text=query_text,
                    created_at=created_at,
                )
                for item in evidence_items:
                    self._merge_item(session, query_id=query_id, item=item, relation="RETURNED")
                    exported_evidence += 1
                for item in contradictions:
                    self._merge_item(session, query_id=query_id, item=item, relation="CONTRADICTED_BY")
                    exported_contradictions += 1
        finally:
            driver.close()

        return {
            "backend": self.backend_name,
            "exported": True,
            "query_id": query_id,
            "database": self.database or "",
            "evidence_count": exported_evidence,
            "contradiction_count": exported_contradictions,
        }

    def _merge_item(self, session: object, *, query_id: str, item: ScoredEvidence, relation: str) -> None:
        evidence_id = _stable_id(
            "evidence",
            item.evidence.doc_id,
            item.evidence.section_path,
            str(item.evidence.char_start),
            str(item.evidence.char_end),
        )
        payload = {
            "query_id": query_id,
            "evidence_id": evidence_id,
            "source_id": item.evidence.doc_id,
            "content": item.evidence.content,
            "section_path": item.evidence.section_path,
            "char_start": int(item.evidence.char_start),
            "char_end": int(item.evidence.char_end),
            "timestamp": item.evidence.timestamp,
            "source_type": item.evidence.source_type.value,
            "score": float(item.score),
            "verdict": item.verdict.value,
            "relation": relation,
        }
        session.run(
            f"""
            MERGE (q:SearchQuery {{id: $query_id}})
            MERGE (s:Source {{id: $source_id}})
            MERGE (e:Evidence {{id: $evidence_id}})
            SET e.content = $content,
                e.section_path = $section_path,
                e.char_start = $char_start,
                e.char_end = $char_end,
                e.timestamp = $timestamp,
                e.source_type = $source_type,
                e.score = $score,
                e.verdict = $verdict
            MERGE (q)-[r:{relation}]->(e)
            SET r.score = $score,
                r.verdict = $verdict
            MERGE (e)-[:FROM_SOURCE]->(s)
            """,
            **payload,
        )


@dataclass(frozen=True)
class Neo4jGraphQuerier:
    uri: str
    username: str
    password: str
    database: str | None = None
    backend_name: str = "neo4j"

    def query_related_sources(
        self,
        *,
        query: str,
        source_ids: Sequence[str] | Iterable[str],
        limit: int = 5,
    ) -> dict[str, object]:
        try:
            from neo4j import GraphDatabase  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("neo4j_not_installed") from exc

        source_rows = [str(source_id).strip() for source_id in source_ids if str(source_id).strip()]
        if not source_rows:
            return {"backend": self.backend_name, "queried": True, "items": []}

        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        items: list[dict[str, object]] = []
        try:
            driver.verify_connectivity()
            with driver.session(database=self.database) as session:
                result = session.run(
                    """
                    MATCH (s:Source)<-[:FROM_SOURCE]-(e:Evidence)<-[r]-(q:SearchQuery)
                    WHERE s.id IN $source_ids
                    RETURN s.id AS source_id,
                           q.id AS query_id,
                           q.text AS query_text,
                           q.created_at AS created_at,
                           e.id AS evidence_id,
                           e.content AS content,
                           e.verdict AS verdict,
                           type(r) AS relation
                    ORDER BY q.created_at DESC
                    LIMIT $limit
                    """,
                    source_ids=source_rows,
                    limit=max(1, int(limit)),
                )
                for row in result:
                    items.append(
                        {
                            "source_id": row.get("source_id"),
                            "query_id": row.get("query_id"),
                            "query_text": row.get("query_text"),
                            "created_at": row.get("created_at"),
                            "evidence_id": row.get("evidence_id"),
                            "content": row.get("content"),
                            "verdict": row.get("verdict"),
                            "relation": row.get("relation"),
                        }
                    )
        finally:
            driver.close()
        return {
            "backend": self.backend_name,
            "queried": True,
            "query_text": str(query or "").strip(),
            "items": items,
        }


def build_graph_exporter(
    *,
    backend: str = "off",
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
    neo4j_database: str | None = None,
) -> GraphExporter:
    selected = str(backend or "off")
    if selected in {"", "off", "none"}:
        return NoopGraphExporter()
    if selected != "neo4j":
        raise ValueError(f"unsupported graph exporter backend: {selected}")

    uri = str(neo4j_uri or os.environ.get("INVESTIGATION_SEARCH_NEO4J_URI", "")).strip()
    user = str(neo4j_user or os.environ.get("INVESTIGATION_SEARCH_NEO4J_USER", "")).strip()
    password = str(neo4j_password or os.environ.get("INVESTIGATION_SEARCH_NEO4J_PASSWORD", "")).strip()
    database = str(neo4j_database or os.environ.get("INVESTIGATION_SEARCH_NEO4J_DATABASE", "")).strip() or None

    if not uri or not user or not password:
        return NoopGraphExporter()
    return Neo4jGraphExporter(uri=uri, username=user, password=password, database=database)


def build_graph_querier(
    *,
    backend: str = "off",
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
    neo4j_database: str | None = None,
) -> GraphQuerier:
    selected = str(backend or "off")
    if selected in {"", "off", "none"}:
        return NoopGraphQuerier()
    if selected != "neo4j":
        raise ValueError(f"unsupported graph querier backend: {selected}")

    uri = str(neo4j_uri or os.environ.get("INVESTIGATION_SEARCH_NEO4J_URI", "")).strip()
    user = str(neo4j_user or os.environ.get("INVESTIGATION_SEARCH_NEO4J_USER", "")).strip()
    password = str(neo4j_password or os.environ.get("INVESTIGATION_SEARCH_NEO4J_PASSWORD", "")).strip()
    database = str(neo4j_database or os.environ.get("INVESTIGATION_SEARCH_NEO4J_DATABASE", "")).strip() or None

    if not uri or not user or not password:
        return NoopGraphQuerier()
    return Neo4jGraphQuerier(uri=uri, username=user, password=password, database=database)


def _stable_id(*parts: str) -> str:
    payload = "|".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
