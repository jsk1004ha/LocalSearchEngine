from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Protocol

from .schema import ScoredEvidence, Verdict


_WORD_RE = re.compile(r"[a-zA-Z0-9가-힣]+")


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text)}


@dataclass(frozen=True)
class ContradictionPrediction:
    verdict: Verdict
    score: float
    rationale: str


class ContradictionDetector(Protocol):
    model_name: str
    model_version: str

    def predict(self, query: str, candidates: Iterable[ScoredEvidence]) -> List[ContradictionPrediction]:
        """Predict contradiction/support/uncertain labels for each candidate in order."""


@dataclass(frozen=True)
class HeuristicContradictionDetector:
    model_name: str = "heuristic-contradiction-detector"
    model_version: str = "0.1"
    contradiction_threshold: float = 0.55

    def predict(self, query: str, candidates: Iterable[ScoredEvidence]) -> List[ContradictionPrediction]:
        contradiction_hints = {"아니다", "없다", "제한", "예외", "불가", "반면", "not", "never", "cannot"}
        support_hints = {"가능", "효과", "개선", "충족", "성공", "improve", "support"}
        query_tokens = _tokenize(query)

        out: list[ContradictionPrediction] = []
        for candidate in candidates:
            text = candidate.evidence.content
            evidence_tokens = _tokenize(text)
            overlap = 0.0
            if query_tokens and evidence_tokens:
                overlap = len(query_tokens & evidence_tokens) / len(query_tokens)

            has_contradiction_hint = any(hint in text for hint in contradiction_hints)
            has_support_hint = any(hint in text for hint in support_hints)

            if has_contradiction_hint:
                score = min(0.55 + overlap * 0.45, 0.99)
                verdict = Verdict.CONTRADICTS
                rationale = "contradiction_hint+overlap"
            elif has_support_hint:
                score = min(0.5 + overlap * 0.45, 0.95)
                verdict = Verdict.SUPPORTS
                rationale = "support_hint+overlap"
            else:
                score = max(0.15, overlap * 0.5)
                verdict = Verdict.UNCERTAIN
                rationale = "weak_semantic_signal"

            out.append(ContradictionPrediction(verdict=verdict, score=score, rationale=rationale))
        return out


@dataclass(frozen=True)
class OnnxContradictionDetectorAdapter:
    """ONNX NLI adapter.

    runtime must implement:
      predict_nli(pairs: list[tuple[str, str]]) -> list[dict|tuple|list]
    """

    runtime: object
    model_name: str = "onnx-contradiction-detector"
    model_version: str = "unknown"

    def predict(self, query: str, candidates: Iterable[ScoredEvidence]) -> List[ContradictionPrediction]:
        pairs = [(query, candidate.evidence.content) for candidate in candidates]
        predict_nli = getattr(self.runtime, "predict_nli", None)
        if not callable(predict_nli):
            raise TypeError("runtime must implement predict_nli(pairs)")

        raw = predict_nli(pairs)
        out: list[ContradictionPrediction] = []
        for row in raw:
            label, score = _parse_nli_output(row)
            out.append(
                ContradictionPrediction(
                    verdict=_map_label_to_verdict(label),
                    score=score,
                    rationale=f"nli:{label}",
                )
            )
        return out


def _parse_nli_output(row: object) -> tuple[str, float]:
    if isinstance(row, dict):
        label = str(row.get("label", "neutral"))
        score = float(row.get("score", 0.0))
        return label, score
    if isinstance(row, (tuple, list)) and len(row) >= 2:
        return str(row[0]), float(row[1])
    return "neutral", 0.0


def _map_label_to_verdict(label: str) -> Verdict:
    normalized = label.strip().lower()
    if normalized in {"contradiction", "contradicts", "refutes"}:
        return Verdict.CONTRADICTS
    if normalized in {"entailment", "supports", "support"}:
        return Verdict.SUPPORTS
    return Verdict.UNCERTAIN
