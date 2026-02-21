from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from .embedding import DEFAULT_EMBEDDING_MODEL, encode_texts
from .index_ann import ANNIndex, build_index, load_index, save_index
from .schema import EvidenceUnit, SourceType


def _stable_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def write_build(
    output_dir: Path,
    evidence_units: Iterable[EvidenceUnit],
    manifest: dict,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ann_backend: str = "hnsw",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = []
    units = list(evidence_units)
    for unit in units:
        payload = asdict(unit)
        payload["source_type"] = unit.source_type.value
        serializable.append(payload)

    evidence_path = output_dir / "evidence_units.json"
    manifest_path = output_dir / "manifest.json"

    evidence_path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    vectors = encode_texts([u.content for u in units], model_name=embedding_model, text_type="passage")
    embedding_path = output_dir / "evidence_embeddings.npy"
    np.save(embedding_path, vectors)

    ann = build_index(vectors, backend=ann_backend)
    ann_index_path, ann_meta_path = save_index(ann, output_dir / "ann_index")

    manifest = dict(manifest)
    manifest.setdefault("artifacts", {})
    manifest["artifacts"]["evidence_units.json"] = _file_hash(evidence_path)
    manifest["artifacts"][embedding_path.name] = _file_hash(embedding_path)
    manifest["artifacts"][ann_index_path.name] = _file_hash(ann_index_path)
    manifest["artifacts"][ann_meta_path.name] = _file_hash(ann_meta_path)
    manifest["embedding_model"] = embedding_model
    manifest["ann_backend"] = ann.backend

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_build(input_dir: Path) -> Tuple[List[EvidenceUnit], dict, np.ndarray, ANNIndex]:
    evidence_path = input_dir / "evidence_units.json"
    manifest_path = input_dir / "manifest.json"
    embedding_path = input_dir / "evidence_embeddings.npy"
    ann_meta_path = input_dir / "ann_index.meta.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_units = json.loads(evidence_path.read_text(encoding="utf-8"))
    units = [
        EvidenceUnit(
            doc_id=row["doc_id"],
            source_type=SourceType(row["source_type"]),
            content=row["content"],
            section_path=row["section_path"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            timestamp=row["timestamp"],
            confidence=row["confidence"],
            metadata=row.get("metadata", {}),
        )
        for row in raw_units
    ]

    embeddings = np.load(embedding_path)
    ann_backend = manifest.get("ann_backend", "hnsw")
    ann_suffix = ".npy" if ann_backend == "exact" else ".bin"
    ann_index_path = input_dir / f"ann_index{ann_suffix}"
    ann = load_index(ann_index_path, ann_meta_path)
    return units, manifest, embeddings, ann
