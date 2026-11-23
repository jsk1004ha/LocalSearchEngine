from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Protocol

from .analyzer import unique_tokens
from .entity import entity_match_features
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
            match_features = entity_match_features(
                query,
                candidate.evidence.content,
                title=str(candidate.evidence.metadata.get("title") or candidate.evidence.section_path or ""),
            )
            score = overlap * 0.78 + length_norm * 0.12
            score += 0.22 * match_features.get("exact_query_match", 0.0)
            score += 0.18 * match_features.get("exact_phrase_match", 0.0)
            score += 0.32 * match_features.get("exact_name_match", 0.0)
            score += 0.14 * match_features.get("title_exact_query_match", 0.0)
            score += 0.18 * match_features.get("title_exact_name_match", 0.0)
            score += 0.12 * match_features.get("role_name_cooccurrence", 0.0)
            score += 0.16 * match_features.get("all_term_coverage", 0.0)
            score += 0.10 * match_features.get("ordered_span_match", 0.0)
            if match_features.get("exact_name_match", 0.0) <= 0:
                score -= 0.25 * match_features.get("similar_name_penalty", 0.0)
            scored.append(max(score, 0.0))
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
