from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

from .schema import EvidenceUnit


def _stable_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def knowledge_build_id(corpus_snapshot_id: str, parser_version: str, embedding_model: str) -> str:
    base = f"{corpus_snapshot_id}|{parser_version}|{embedding_model}"
    return _stable_hash(base)[:16]


def build_manifest(
    evidence_units: Iterable[EvidenceUnit],
    corpus_snapshot_id: str,
    parser_version: str,
    embedding_model: str,
) -> dict:
    units: List[EvidenceUnit] = sorted(
        list(evidence_units),
        key=lambda e: (e.doc_id, e.section_path, e.char_start, e.char_end, e.source_type.value),
    )
    serialized = [
        {
            **asdict(unit),
            "source_type": unit.source_type.value,
        }
        for unit in units
    ]
    unit_blob = json.dumps(serialized, ensure_ascii=False, sort_keys=True)
    build_id = knowledge_build_id(corpus_snapshot_id, parser_version, embedding_model)
    return {
        "knowledge_build_id": build_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_snapshot_id": corpus_snapshot_id,
        "parser_version": parser_version,
        "embedding_model": embedding_model,
        "evidence_count": len(serialized),
        "artifact_checksum": _stable_hash(unit_blob),
        "artifacts": {
            "evidence_units.json": _stable_hash(unit_blob),
        },
    }


def write_build(output_dir: Path, evidence_units: Iterable[EvidenceUnit], manifest: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = []
    for unit in evidence_units:
        payload = asdict(unit)
        payload["source_type"] = unit.source_type.value
        serializable.append(payload)

    evidence_path = output_dir / "evidence_units.json"
    manifest_path = output_dir / "manifest.json"

    evidence_path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
