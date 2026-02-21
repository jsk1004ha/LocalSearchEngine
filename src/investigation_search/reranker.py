from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Protocol

from .analyzer import unique_tokens
from .schema import ScoredEvidence


class Reranker(Protocol):
    model_name: str
    model_version: str

    def score(self, query: str, candidates: Iterable[ScoredEvidence]) -> List[float]:
        """Returns rerank score per candidate in the original order."""


@dataclass(frozen=True)
class LocalCrossEncoderReranker:
    """Small local cross-encoder style reranker (token overlap heuristic)."""

    model_name: str = "local-cross-encoder-mini"
    model_version: str = "0.1"

    def score(self, query: str, candidates: Iterable[ScoredEvidence]) -> List[float]:
        query_tokens = set(_tokenize(query))
        scored: List[float] = []
        for candidate in candidates:
            evidence_tokens = set(_tokenize(candidate.evidence.content))
            if not query_tokens or not evidence_tokens:
                scored.append(0.0)
                continue
            overlap = len(query_tokens & evidence_tokens) / len(query_tokens)
            length_norm = min(len(evidence_tokens) / 32.0, 1.0)
            scored.append(overlap * 0.85 + length_norm * 0.15)
        return scored


@dataclass(frozen=True)
class OnnxRerankerAdapter:
    """ONNX inference interface adapter.

    runtime must expose `predict_pairs(pairs: list[tuple[str, str]]) -> list[float]`.
    """

    runtime: object
    model_name: str = "onnx-cross-encoder"
    model_version: str = "unknown"

    def score(self, query: str, candidates: Iterable[ScoredEvidence]) -> List[float]:
        pairs = [(query, candidate.evidence.content) for candidate in candidates]
        predict_pairs = getattr(self.runtime, "predict_pairs", None)
        if not callable(predict_pairs):
            raise TypeError("runtime must implement predict_pairs(pairs)")
        raw_scores = predict_pairs(pairs)
        return [float(score) for score in raw_scores]


def _tokenize(text: str) -> List[str]:
    return list(unique_tokens(text, mode="rerank", include_char_ngrams=True))
