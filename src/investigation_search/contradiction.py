from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Protocol

from .schema import ScoredEvidence, Verdict
from .verification import infer_claim_evidence_relation


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
    model_version: str = "0.2"
    contradiction_threshold: float = 0.55

    def predict(self, query: str, candidates: Iterable[ScoredEvidence]) -> List[ContradictionPrediction]:
        out: list[ContradictionPrediction] = []
        for candidate in candidates:
            relation = infer_claim_evidence_relation(query, candidate.evidence.content)
            if relation.contradict >= self.contradiction_threshold:
                verdict, score, rationale = Verdict.CONTRADICTS, relation.contradict, "pair_contradiction"
            elif relation.entail >= 0.45 and relation.entail > relation.contradict:
                verdict, score, rationale = Verdict.SUPPORTS, relation.entail, "pair_entailment"
            else:
                verdict, score, rationale = Verdict.UNCERTAIN, relation.neutral, "pair_neutral"
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
