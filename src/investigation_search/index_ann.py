from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass
class ANNIndex:
    backend: str
    dim: int
    ids: np.ndarray
    index: object


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


def build_index(
    vectors: np.ndarray,
    ids: Sequence[int] | None = None,
    *,
    backend: str = "hnsw",
    ef_construction: int = 200,
    m: int = 32,
    ef_search: int = 64,
    nlist: int = 256,
) -> ANNIndex:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("vectors는 shape=(n, dim) 이어야 합니다.")

    n, dim = vectors.shape
    ids_arr = np.arange(n, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
    if len(ids_arr) != n:
        raise ValueError("ids 길이는 vectors 행 수와 같아야 합니다.")

    if backend == "hnsw":
        try:
            import hnswlib

            index = hnswlib.Index(space="cosine", dim=dim)
            index.init_index(max_elements=n, ef_construction=ef_construction, M=m)
            index.add_items(vectors, ids_arr)
            index.set_ef(ef_search)
            return ANNIndex(backend="hnsw", dim=dim, ids=ids_arr, index=index)
        except ImportError:
            backend = "faiss"

    if backend == "faiss":
        try:
            import faiss

            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            train_vecs = _normalize_rows(vectors)
            if not index.is_trained:
                index.train(train_vecs)
            id_index = faiss.IndexIDMap2(index)
            id_index.add_with_ids(train_vecs, ids_arr)
            return ANNIndex(backend="faiss", dim=dim, ids=ids_arr, index=id_index)
        except ImportError:
            backend = "exact"

    if backend == "exact":
        # fallback: ANN 라이브러리가 없을 때 동작하는 내적 기반 검색
        return ANNIndex(backend="exact", dim=dim, ids=ids_arr, index=_normalize_rows(vectors))

    raise ValueError(f"지원하지 않는 backend: {backend}")


def search_index(index: ANNIndex, query_vectors: np.ndarray, top_k: int) -> list[list[tuple[int, float]]]:
    queries = np.asarray(query_vectors, dtype=np.float32)
    if queries.ndim == 1:
        queries = queries[None, :]

    if index.backend == "hnsw":
        labels, distances = index.index.knn_query(queries, k=top_k)
        return [
            [(int(doc_id), float(1.0 - dist)) for doc_id, dist in zip(row_ids, row_dist)]
            for row_ids, row_dist in zip(labels, distances)
        ]

    if index.backend == "faiss":
        scores, labels = index.index.search(_normalize_rows(queries), top_k)
        return [
            [
                (int(doc_id), float(score))
                for doc_id, score in zip(row_ids, row_scores)
                if doc_id >= 0
            ]
            for row_ids, row_scores in zip(labels, scores)
        ]

    matrix = index.index
    norm_queries = _normalize_rows(queries)
    sim = norm_queries @ matrix.T
    order = np.argsort(-sim, axis=1)[:, :top_k]
    results: list[list[tuple[int, float]]] = []
    for qi, row in enumerate(order):
        results.append([(int(index.ids[col]), float(sim[qi, col])) for col in row])
    return results


def save_index(index: ANNIndex, output_prefix: Path) -> tuple[Path, Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output_prefix.with_suffix(".meta.json")
    index_path = output_prefix.with_suffix(".bin")

    meta = {"backend": index.backend, "dim": index.dim, "ids": index.ids.tolist()}
    if index.backend == "hnsw":
        index.index.save_index(str(index_path))
    elif index.backend == "faiss":
        import faiss

        faiss.write_index(index.index, str(index_path))
    else:
        np.save(index_path.with_suffix(".npy"), index.index)
        index_path = index_path.with_suffix(".npy")

    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return index_path, meta_path


def load_index(index_path: Path, meta_path: Path) -> ANNIndex:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    backend = meta["backend"]
    dim = int(meta["dim"])
    ids = np.asarray(meta["ids"], dtype=np.int64)

    if backend == "hnsw":
        import hnswlib

        index = hnswlib.Index(space="cosine", dim=dim)
        index.load_index(str(index_path))
        return ANNIndex(backend=backend, dim=dim, ids=ids, index=index)
    if backend == "faiss":
        import faiss

        index = faiss.read_index(str(index_path))
        return ANNIndex(backend=backend, dim=dim, ids=ids, index=index)

    matrix = np.load(index_path)
    return ANNIndex(backend="exact", dim=dim, ids=ids, index=matrix)
