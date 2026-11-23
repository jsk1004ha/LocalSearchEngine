from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Sequence

from .contradiction import ContradictionPrediction
from .schema import ScoredEvidence, Verdict


_SCORE_RE = re.compile(r"([01](?:\.\d+)?)")


@dataclass(frozen=True)
class OllamaClient:
    base_url: str = field(default_factory=lambda: os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    timeout_sec: float = 60.0

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None = None,
        options: Mapping[str, object] | None = None,
    ) -> str:
        payload: dict[str, object] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if options:
            payload["options"] = dict(options)
        res = self._post_json("/api/generate", payload)
        text = res.get("response")
        return str(text or "")

    def embed(self, *, model: str, texts: Sequence[str] | Iterable[str]) -> List[List[float]]:
        seq = list(texts)
        if not seq:
            return []

        try:
            res = self._post_json("/api/embed", {"model": model, "input": seq})
            rows = res.get("embeddings")
            if isinstance(rows, list) and rows:
                return [[float(v) for v in row] for row in rows]
        except RuntimeError:
            # Fallback for older Ollama versions.
            pass

        vectors: list[list[float]] = []
        for text in seq:
            res = self._post_json("/api/embeddings", {"model": model, "prompt": text})
            row = res.get("embedding")
            if not isinstance(row, list):
                raise RuntimeError("Ollama embedding response does not contain `embedding` field.")
            vectors.append([float(v) for v in row])
        return vectors

    def _post_json(self, path: str, payload: Mapping[str, object]) -> dict:
        url = self.base_url.rstrip("/") + path
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Ollama HTTP error ({exc.code}) for {path}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama connection failed for {path}. base_url={self.base_url}. "
                "Ollama server 실행 상태를 확인하세요."
            ) from exc

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Ollama response is not valid JSON for {path}: {raw[:200]}") from exc


@dataclass(frozen=True)
class OllamaRerankerAdapter:
    model: str = "llama3.1:8b"
    client: OllamaClient = field(default_factory=OllamaClient)
    model_name: str = "ollama-reranker"
    model_version: str = "0.1"

    def score(self, query: str, candidates: Iterable[ScoredEvidence]) -> List[float]:
        scores: list[float] = []
        for candidate in candidates:
            prompt = (
                "다음 query와 passage의 검색 관련성을 0.0~1.0 실수 하나로만 답하세요.\n"
                f"query: {query}\n"
                f"passage: {candidate.evidence.content}\n"
                "output:"
            )
            raw = self.client.generate(model=self.model, prompt=prompt)
            scores.append(_parse_score(raw))
        return scores


@dataclass(frozen=True)
class OllamaContradictionDetector:
    model: str = "llama3.1:8b"
    client: OllamaClient = field(default_factory=OllamaClient)
    model_name: str = "ollama-contradiction-detector"
    model_version: str = "0.1"

    def predict(self, query: str, candidates: Iterable[ScoredEvidence]) -> List[ContradictionPrediction]:
        out: list[ContradictionPrediction] = []
        for candidate in candidates:
            prompt = (
                "질의와 근거 문장을 보고 verdict를 supports|contradicts|uncertain 중 하나로 판단하세요.\n"
                "JSON으로만 답하세요: {\"verdict\":\"...\",\"score\":0.0~1.0,\"rationale\":\"짧게\"}\n"
                f"query: {query}\n"
                f"evidence: {candidate.evidence.content}"
            )
            raw = self.client.generate(model=self.model, prompt=prompt)
            verdict, score, rationale = _parse_contradiction_json(raw)
            out.append(
                ContradictionPrediction(
                    verdict=verdict,
                    score=score,
                    rationale=rationale,
                )
            )
        return out


def _parse_score(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    try:
        return max(0.0, min(float(stripped), 1.0))
    except ValueError:
        pass
    match = _SCORE_RE.search(stripped)
    if not match:
        return 0.0
    try:
        return max(0.0, min(float(match.group(1)), 1.0))
    except ValueError:
        return 0.0


def _parse_contradiction_json(text: str) -> tuple[Verdict, float, str]:
    raw = text.strip()
    if not raw:
        return Verdict.UNCERTAIN, 0.0, "empty_response"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return Verdict.UNCERTAIN, _parse_score(raw), "invalid_json"

    label = str(payload.get("verdict", "uncertain")).strip().lower()
    if label == "supports":
        verdict = Verdict.SUPPORTS
    elif label == "contradicts":
        verdict = Verdict.CONTRADICTS
    else:
        verdict = Verdict.UNCERTAIN
    score = _parse_score(str(payload.get("score", "0")))
    rationale = str(payload.get("rationale", "ollama_prediction"))
    return verdict, score, rationale
