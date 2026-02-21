from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

from .retrieval import RetrievalWeights, tokenize
from .schema import ScoredEvidence


_WEIGHT_KEYS: tuple[str, ...] = ("bm25", "dense", "lexical", "rrf")


@dataclass(frozen=True)
class LearningConfig:
    enabled: bool = True
    learning_rate: float = 0.08
    min_weight: float = 0.05
    token_gain: float = 0.04
    token_decay: float = 0.995
    max_token_boost: float = 1.25
    autosave_every: int = 20
    state_path: str | None = None


@dataclass
class LearningState:
    weights: Dict[str, float]
    token_boosts: Dict[str, float] = field(default_factory=dict)
    searches: int = 0
    updates: int = 0
    version: int = 0
    last_updated: str | None = None


class OnlineLearningManager:
    def __init__(self, base_weights: RetrievalWeights, config: LearningConfig | None = None):
        self.config = config or LearningConfig()
        self.state = LearningState(weights=_weights_from_dataclass(base_weights))
        self._normalize_weights()
        if self.config.state_path:
            self._load_state(self.config.state_path)

    def current_weights(self) -> RetrievalWeights:
        w = self.state.weights
        return RetrievalWeights(
            bm25=float(w.get("bm25", 0.0)),
            dense=float(w.get("dense", 0.0)),
            lexical=float(w.get("lexical", 0.0)),
            rrf=float(w.get("rrf", 0.0)),
            rrf_k=60,
        )

    def current_token_boosts(self) -> Mapping[str, float]:
        return dict(self.state.token_boosts)

    def learn(self, query: str, ranked: Sequence[ScoredEvidence] | Iterable[ScoredEvidence]) -> dict:
        self.state.searches += 1
        if not self.config.enabled:
            return self.snapshot()

        seq = list(ranked)
        if not seq:
            return self.snapshot()

        positives = self._pick_positive_samples(seq)
        target = self._target_weight_distribution(positives)
        if target:
            self._update_weights(target)
        self._update_token_boosts(query, positives)

        self.state.updates += 1
        self.state.version += 1
        self.state.last_updated = datetime.now(timezone.utc).isoformat()

        if self.config.state_path and self.state.updates % max(self.config.autosave_every, 1) == 0:
            self.save(self.config.state_path)
        return self.snapshot()

    def snapshot(self) -> dict:
        return {
            "enabled": self.config.enabled,
            "version": self.state.version,
            "searches": self.state.searches,
            "updates": self.state.updates,
            "weights": {k: round(self.state.weights.get(k, 0.0), 6) for k in _WEIGHT_KEYS},
            "token_boost_count": len(self.state.token_boosts),
            "state_path": self.config.state_path,
        }

    def save(self, path: str | Path | None = None) -> None:
        if path is None:
            path = self.config.state_path
        if not path:
            return
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "weights": self.state.weights,
            "token_boosts": self.state.token_boosts,
            "searches": self.state.searches,
            "updates": self.state.updates,
            "version": self.state.version,
            "last_updated": self.state.last_updated,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    @property
    def version(self) -> int:
        return self.state.version

    def _load_state(self, path: str | Path) -> None:
        src = Path(path)
        if not src.exists():
            return
        payload = json.loads(src.read_text(encoding="utf-8"))
        weights = payload.get("weights")
        if isinstance(weights, dict):
            for key in _WEIGHT_KEYS:
                if key in weights:
                    self.state.weights[key] = float(weights[key])
            self._normalize_weights()

        token_boosts = payload.get("token_boosts")
        if isinstance(token_boosts, dict):
            self.state.token_boosts = {str(k): float(v) for k, v in token_boosts.items() if float(v) > 0}
        self.state.searches = int(payload.get("searches", 0))
        self.state.updates = int(payload.get("updates", 0))
        self.state.version = int(payload.get("version", 0))
        self.state.last_updated = payload.get("last_updated")

    def _pick_positive_samples(self, ranked: Sequence[ScoredEvidence]) -> list[ScoredEvidence]:
        positives: list[ScoredEvidence] = []
        for item in ranked[:5]:
            if item.score <= 0:
                continue
            positives.append(item)
        return positives[:3]

    def _target_weight_distribution(self, positives: Sequence[ScoredEvidence]) -> Dict[str, float]:
        if not positives:
            return {}
        accum = {key: 0.0 for key in _WEIGHT_KEYS}
        total = 0.0
        for item in positives:
            row = item.stage_scores
            row_sum = 0.0
            row_vals: dict[str, float] = {}
            for key in _WEIGHT_KEYS:
                val = float(row.get(key, 0.0))
                if val < 0:
                    val = 0.0
                row_vals[key] = val
                row_sum += val
            if row_sum <= 0:
                continue
            row_weight = max(item.score, 0.0)
            total += row_weight
            for key, value in row_vals.items():
                accum[key] += row_weight * (value / row_sum)

        if total <= 0:
            return {}
        return {key: accum[key] / total for key in _WEIGHT_KEYS}

    def _update_weights(self, target: Mapping[str, float]) -> None:
        lr = min(max(self.config.learning_rate, 0.0), 1.0)
        for key in _WEIGHT_KEYS:
            prev = float(self.state.weights.get(key, 0.0))
            tgt = float(target.get(key, prev))
            self.state.weights[key] = (1.0 - lr) * prev + lr * tgt
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        min_w = max(0.0, self.config.min_weight)
        for key in _WEIGHT_KEYS:
            self.state.weights[key] = max(float(self.state.weights.get(key, 0.0)), min_w)
        s = sum(self.state.weights.values())
        if s <= 0:
            fallback = 1.0 / len(_WEIGHT_KEYS)
            for key in _WEIGHT_KEYS:
                self.state.weights[key] = fallback
            return
        for key in _WEIGHT_KEYS:
            self.state.weights[key] /= s

    def _update_token_boosts(self, query: str, positives: Sequence[ScoredEvidence]) -> None:
        if not positives:
            return
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return

        decay = min(max(self.config.token_decay, 0.0), 1.0)
        for token in query_tokens:
            prev = float(self.state.token_boosts.get(token, 0.0))
            if prev <= 0:
                continue
            self.state.token_boosts[token] = prev * decay

        token_gain = max(0.0, self.config.token_gain)
        max_boost = max(0.0, self.config.max_token_boost)
        for item in positives:
            evidence_tokens = set(tokenize(item.evidence.content))
            overlap = query_tokens & evidence_tokens
            if not overlap:
                continue
            quality = max(0.0, min(item.score, 1.0))
            for token in overlap:
                prev = float(self.state.token_boosts.get(token, 0.0))
                self.state.token_boosts[token] = min(max_boost, prev + token_gain * (0.5 + quality))

        stale = [token for token, val in self.state.token_boosts.items() if val < 0.005]
        for token in stale:
            self.state.token_boosts.pop(token, None)


def _weights_from_dataclass(weights: RetrievalWeights) -> Dict[str, float]:
    return {
        "bm25": float(weights.bm25),
        "dense": float(weights.dense),
        "lexical": float(weights.lexical),
        "rrf": float(weights.rrf),
    }
