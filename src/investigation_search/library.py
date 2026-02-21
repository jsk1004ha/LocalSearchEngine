from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .schema import EvidenceUnit, ScoredEvidence, SearchResult, SourceCitation


def _stable_id(payload: str) -> str:
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


class KnowledgeLibrary:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / "sessions").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "evidence").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "osint").mkdir(parents=True, exist_ok=True)

    def save_session(
        self,
        *,
        mode: str,
        raw_query: str,
        clean_query: str,
        filters: Mapping[str, Any] | None,
        result: SearchResult,
        extra: Mapping[str, Any] | None = None,
    ) -> str:
        now = datetime.now(timezone.utc).isoformat()
        session_id = _stable_id(f"{now}|{mode}|{raw_query}|{clean_query}")[:16]

        record = {
            "session_id": session_id,
            "created_at": now,
            "mode": mode,
            "raw_query": raw_query,
            "clean_query": clean_query,
            "filters": dict(filters or {}),
            "result": _serialize_search_result(result),
            "extra": dict(extra or {}),
        }

        session_path = self.base_dir / "sessions" / f"{session_id}.json"
        session_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

        # Append index line for quick grep.
        index_path = self.base_dir / "sessions.jsonl"
        with index_path.open("a", encoding="utf-8") as fp:
            fp.write(
                json.dumps(
                    {
                        "session_id": session_id,
                        "created_at": now,
                        "mode": mode,
                        "raw_query": raw_query,
                        "clean_query": clean_query,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        return session_id

    def append_evidence_units(self, units: Sequence[EvidenceUnit] | Iterable[EvidenceUnit], *, tag: str) -> int:
        out_path = self.base_dir / "evidence" / f"{tag}.jsonl"
        rows = []
        for unit in units:
            rows.append(_serialize_evidence_unit(unit))
        if not rows:
            return 0

        with out_path.open("a", encoding="utf-8") as fp:
            for row in rows:
                fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return len(rows)

    def save_osint_artifacts(self, session_id: str, artifacts: Mapping[str, Any]) -> Path:
        path = self.base_dir / "osint" / f"{session_id}.json"
        path.write_text(json.dumps(dict(artifacts), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def delete_all(self) -> None:
        # Best-effort delete; avoids shutil for tight sandbox environments.
        for child in self.base_dir.rglob("*"):
            if child.is_file():
                try:
                    child.unlink(missing_ok=True)
                except Exception:
                    pass
        for child in sorted([p for p in self.base_dir.rglob("*") if p.is_dir()], reverse=True):
            try:
                child.rmdir()
            except Exception:
                pass
        try:
            self.base_dir.rmdir()
        except Exception:
            pass


def _serialize_search_result(result: SearchResult) -> dict:
    return {
        "answer": result.answer,
        "answer_sources": [_serialize_source_citation(src) for src in result.answer_sources],
        "sources": [_serialize_source_citation(src) for src in result.sources],
        "evidence": [_serialize_scored_evidence(item) for item in result.evidence],
        "contradictions": [_serialize_scored_evidence(item) for item in result.contradictions],
        "diagnostics": result.diagnostics,
        "reranker_model": result.reranker_model,
        "reranker_version": result.reranker_version,
        "build_id": result.build_id,
    }


def _serialize_source_citation(src: SourceCitation) -> dict:
    return {
        "citation_id": src.citation_id,
        "doc_id": src.doc_id,
        "section_path": src.section_path,
        "char_start": src.char_start,
        "char_end": src.char_end,
        "source_type": src.source_type.value,
        "timestamp": src.timestamp,
    }


def _serialize_scored_evidence(item: ScoredEvidence) -> dict:
    return {
        "evidence": _serialize_evidence_unit(item.evidence),
        "score": item.score,
        "verdict": item.verdict.value,
        "why_it_matches": item.why_it_matches,
        "stage_scores": dict(item.stage_scores),
        "source": _serialize_source_citation(item.source) if item.source is not None else None,
    }


def _serialize_evidence_unit(unit: EvidenceUnit) -> dict:
    payload = asdict(unit)
    payload["source_type"] = unit.source_type.value
    return payload
