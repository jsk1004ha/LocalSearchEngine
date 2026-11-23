from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .bm25 import BM25Index, build_bm25_index, load_bm25_index, save_bm25_index
from .embedding import DEFAULT_EMBEDDING_MODEL, encode_texts
from .index_ann import ANNIndex, build_index, load_index, save_index
from .parser import DocumentParser, parse_documents
from .schema import EvidenceUnit, SourceType


def _stable_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unit_cache_key(unit: EvidenceUnit) -> str:
    payload = {
        "doc_id": unit.doc_id,
        "source_type": unit.source_type.value,
        "content": unit.content,
        "section_path": unit.section_path,
        "char_start": unit.char_start,
        "char_end": unit.char_end,
        "metadata": unit.metadata,
    }
    return _stable_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _shard_for_doc(doc_id: str, shard_count: int) -> int:
    digest = hashlib.sha1(doc_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False) % shard_count


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
    previous_build_dir: Path | None = None,
    shard_count: int = 1,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = []
    units = list(evidence_units)
    if not units:
        raise ValueError("evidence_units가 비어 있습니다.")

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

    vectors, inc_stats = _encode_with_incremental_reuse(
        units,
        embedding_model=embedding_model,
        previous_build_dir=previous_build_dir,
    )
    embedding_path = output_dir / "evidence_embeddings.npy"
    np.save(embedding_path, vectors)

    ann = build_index(vectors, backend=ann_backend)
    ann_index_path, ann_meta_path = save_index(ann, output_dir / "ann_index")
    bm25 = build_bm25_index([u.content for u in units])
    bm25_path = save_bm25_index(bm25, output_dir / "bm25_index.json")

    manifest = dict(manifest)
    manifest.setdefault("artifacts", {})
    manifest["artifacts"]["evidence_units.json"] = _file_hash(evidence_path)
    manifest["artifacts"][embedding_path.name] = _file_hash(embedding_path)
    manifest["artifacts"][ann_index_path.name] = _file_hash(ann_index_path)
    manifest["artifacts"][ann_meta_path.name] = _file_hash(ann_meta_path)
    manifest["artifacts"][bm25_path.name] = _file_hash(bm25_path)
    manifest["embedding_model"] = embedding_model
    manifest["ann_backend"] = ann.backend
    manifest["incremental"] = inc_stats

    shard_count = max(1, int(shard_count))
    manifest["shard_count"] = shard_count
    if shard_count > 1:
        shard_entries = _write_shards(
            base_output_dir=output_dir,
            units=units,
            vectors=vectors,
            ann_backend=ann_backend,
            shard_count=shard_count,
        )
        manifest["shards"] = shard_entries

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


def load_sharded_build(input_dir: Path) -> list[tuple[list[EvidenceUnit], np.ndarray, ANNIndex | None]]:
    manifest_path = input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_meta = manifest.get("shards", [])
    bundles: list[tuple[list[EvidenceUnit], np.ndarray, ANNIndex | None]] = []

    for shard in shard_meta:
        shard_dir = input_dir / shard["path"]
        evidence_path = shard_dir / "evidence_units.json"
        embedding_path = shard_dir / "evidence_embeddings.npy"
        ann_meta_path = shard_dir / "ann_index.meta.json"
        units = _load_units(evidence_path)
        embeddings = np.load(embedding_path)

        ann_backend = shard.get("ann_backend", "empty")
        if ann_backend == "empty" or not ann_meta_path.exists():
            bundles.append((units, embeddings, None))
            continue
        ann_suffix = ".npy" if ann_backend == "exact" else ".bin"
        ann_index_path = shard_dir / f"ann_index{ann_suffix}"
        ann = load_index(ann_index_path, ann_meta_path)
        bundles.append((units, embeddings, ann))
    return bundles


def load_bm25_from_build(input_dir: Path) -> BM25Index:
    path = input_dir / "bm25_index.json"
    return load_bm25_index(path)


def load_sharded_bm25_indices(input_dir: Path) -> list[BM25Index]:
    manifest_path = input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_meta = manifest.get("shards", [])
    out: list[BM25Index] = []
    for shard in shard_meta:
        path = input_dir / shard["path"] / "bm25_index.json"
        if not path.exists():
            out.append(build_bm25_index([]))
            continue
        out.append(load_bm25_index(path))
    return out


def write_build_from_documents(
    output_dir: Path,
    document_paths: Sequence[str | Path] | Iterable[str | Path],
    manifest: dict,
    *,
    parser: DocumentParser | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ann_backend: str = "hnsw",
    previous_build_dir: Path | None = None,
    shard_count: int = 1,
) -> List[EvidenceUnit]:
    units = parse_documents(document_paths, parser=parser)
    write_build(
        output_dir=output_dir,
        evidence_units=units,
        manifest=manifest,
        embedding_model=embedding_model,
        ann_backend=ann_backend,
        previous_build_dir=previous_build_dir,
        shard_count=shard_count,
    )
    return units


def _load_units(evidence_path: Path) -> list[EvidenceUnit]:
    raw_units = json.loads(evidence_path.read_text(encoding="utf-8"))
    return [
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


def _encode_with_incremental_reuse(
    units: list[EvidenceUnit],
    *,
    embedding_model: str,
    previous_build_dir: Path | None,
) -> tuple[np.ndarray, dict]:
    reused_vectors: dict[str, np.ndarray] = {}
    if previous_build_dir is not None:
        prev_evidence = previous_build_dir / "evidence_units.json"
        prev_embeddings = previous_build_dir / "evidence_embeddings.npy"
        if prev_evidence.exists() and prev_embeddings.exists():
            prev_units = _load_units(prev_evidence)
            prev_matrix = np.load(prev_embeddings)
            for idx, unit in enumerate(prev_units):
                if idx >= len(prev_matrix):
                    break
                reused_vectors[_unit_cache_key(unit)] = np.asarray(prev_matrix[idx], dtype=np.float32)

    vectors: list[np.ndarray | None] = [None] * len(units)
    new_texts: list[str] = []
    new_indices: list[int] = []
    reused_count = 0

    for idx, unit in enumerate(units):
        key = _unit_cache_key(unit)
        prev_vec = reused_vectors.get(key)
        if prev_vec is not None:
            vectors[idx] = prev_vec
            reused_count += 1
        else:
            new_indices.append(idx)
            new_texts.append(unit.content)

    new_vectors = np.zeros((0, 0), dtype=np.float32)
    if new_texts:
        new_vectors = encode_texts(new_texts, model_name=embedding_model, text_type="passage")
        for idx, vec in zip(new_indices, new_vectors):
            vectors[idx] = np.asarray(vec, dtype=np.float32)

    first = next((vec for vec in vectors if vec is not None), None)
    if first is None:
        matrix = np.zeros((len(units), 0), dtype=np.float32)
    else:
        dim = int(np.asarray(first, dtype=np.float32).shape[0])
        matrix = np.zeros((len(units), dim), dtype=np.float32)
        for i, vec in enumerate(vectors):
            if vec is None:
                continue
            matrix[i] = np.asarray(vec, dtype=np.float32)

    stats = {
        "previous_build_dir": str(previous_build_dir) if previous_build_dir is not None else None,
        "reused_vectors": reused_count,
        "new_vectors": len(new_indices),
        "total_vectors": len(units),
    }
    return matrix, stats


def _write_shards(
    *,
    base_output_dir: Path,
    units: list[EvidenceUnit],
    vectors: np.ndarray,
    ann_backend: str,
    shard_count: int,
) -> list[dict]:
    shard_root = base_output_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []

    shard_indices: list[list[int]] = [[] for _ in range(shard_count)]
    for idx, unit in enumerate(units):
        shard_id = _shard_for_doc(unit.doc_id, shard_count)
        shard_indices[shard_id].append(idx)

    for shard_id, indices in enumerate(shard_indices):
        shard_dir = shard_root / f"shard_{shard_id:03d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_units = [units[i] for i in indices]
        shard_vectors = vectors[indices]

        serializable = []
        for unit in shard_units:
            payload = asdict(unit)
            payload["source_type"] = unit.source_type.value
            serializable.append(payload)

        evidence_path = shard_dir / "evidence_units.json"
        embedding_path = shard_dir / "evidence_embeddings.npy"
        evidence_path.write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        np.save(embedding_path, shard_vectors)

        shard_entry = {
            "shard_id": shard_id,
            "path": str((Path("shards") / f"shard_{shard_id:03d}").as_posix()),
            "evidence_count": len(shard_units),
            "artifacts": {
                "evidence_units.json": _file_hash(evidence_path),
                "evidence_embeddings.npy": _file_hash(embedding_path),
            },
        }

        if len(shard_units) == 0:
            shard_entry["ann_backend"] = "empty"
        else:
            ann = build_index(shard_vectors, backend=ann_backend)
            ann_index_path, ann_meta_path = save_index(ann, shard_dir / "ann_index")
            shard_entry["ann_backend"] = ann.backend
            shard_entry["artifacts"][ann_index_path.name] = _file_hash(ann_index_path)
            shard_entry["artifacts"][ann_meta_path.name] = _file_hash(ann_meta_path)

        bm25 = build_bm25_index([unit.content for unit in shard_units])
        bm25_path = save_bm25_index(bm25, shard_dir / "bm25_index.json")
        shard_entry["artifacts"][bm25_path.name] = _file_hash(bm25_path)

        entries.append(shard_entry)

    return entries
