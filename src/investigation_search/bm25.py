from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


_WORD_RE = re.compile(r"[a-zA-Z0-9가-힣]+")


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


@dataclass(frozen=True)
class BM25Index:
    postings: Dict[str, List[Tuple[int, int]]]
    doc_len: List[int]
    doc_count: int
    avg_doc_len: float
    k1: float = 1.5
    b: float = 0.75


def build_bm25_index(
    texts: Sequence[str] | Iterable[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> BM25Index:
    seq = list(texts)
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    doc_len: list[int] = []

    for doc_id, text in enumerate(seq):
        terms = tokenize(text)
        doc_len.append(len(terms))
        tf = Counter(terms)
        for term, freq in tf.items():
            postings[term].append((doc_id, int(freq)))

    avg_doc_len = (sum(doc_len) / len(doc_len)) if doc_len else 0.0
    return BM25Index(
        postings=dict(postings),
        doc_len=doc_len,
        doc_count=len(seq),
        avg_doc_len=avg_doc_len,
        k1=k1,
        b=b,
    )


def _idf(doc_count: int, doc_freq: int) -> float:
    return math.log1p((doc_count - doc_freq + 0.5) / (doc_freq + 0.5))


def search_bm25(
    index: BM25Index,
    query: str,
    *,
    top_k: int = 10,
    token_boosts: Mapping[str, float] | None = None,
) -> List[Tuple[int, float]]:
    if top_k <= 0 or index.doc_count == 0:
        return []

    q_terms = tokenize(query)
    if not q_terms:
        return []

    scores: dict[int, float] = defaultdict(float)
    k1 = index.k1
    b = index.b
    avg_doc_len = index.avg_doc_len if index.avg_doc_len > 0 else 1.0

    for term in q_terms:
        postings = index.postings.get(term)
        if not postings:
            continue
        term_weight = 1.0
        if token_boosts is not None:
            term_weight += max(0.0, min(float(token_boosts.get(term, 0.0)), 2.0))
        idf = _idf(index.doc_count, len(postings))
        for doc_id, tf in postings:
            dl = index.doc_len[doc_id]
            denom = tf + k1 * (1 - b + b * dl / avg_doc_len)
            if denom <= 0:
                continue
            scores[doc_id] += term_weight * idf * ((tf * (k1 + 1)) / denom)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
