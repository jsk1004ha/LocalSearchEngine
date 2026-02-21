from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, List, Sequence


_MASK_PATTERNS = (
    (re.compile(r"([a-zA-Z])[a-zA-Z*]{1,}([a-zA-Z])"), r"\1*\2"),
    (re.compile(r"([가-힣])[가-힣*]{1,}([가-힣])"), r"\1*\2"),
    (re.compile(r"(\d{2,})[-_]?[*xX]{2,}(\d{0,})"), r"\1***\2"),
)
_EMAIL_RE = re.compile(r"([a-zA-Z0-9._%+-]{1,3})[a-zA-Z0-9._%+-]*(@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})")
_TOKEN_RE = re.compile(r"[a-zA-Z0-9가-힣@._+-]+")

_KEYBOARD_ADJACENCY = {
    "q": "wa",
    "w": "qase",
    "e": "wsdr",
    "r": "edft",
    "t": "rfgy",
    "y": "tghu",
    "u": "yhji",
    "i": "ujko",
    "o": "iklp",
    "p": "ol",
    "a": "qwsz",
    "s": "qweadzx",
    "d": "wersfxc",
    "f": "ertdgcv",
    "g": "rtyfhvb",
    "h": "tyugjbn",
    "j": "yuihknm",
    "k": "uiojlm",
    "l": "iopk",
    "z": "asx",
    "x": "zsdc",
    "c": "xdfv",
    "v": "cfgb",
    "b": "vghn",
    "n": "bhjm",
    "m": "njk",
}


@dataclass(frozen=True)
class AliasCandidate:
    alias: str
    penalty: float
    reason: str


def normalize_masked_entity(text: str) -> str:
    """Normalize partially masked identifiers for legal data integration (non-identification use)."""
    normalized = text.strip().lower()
    normalized = _EMAIL_RE.sub(r"\1***\2", normalized)
    for pattern, replacement in _MASK_PATTERNS:
        normalized = pattern.sub(replacement, normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def generate_alias_candidates(entity: str) -> List[AliasCandidate]:
    normalized = normalize_masked_entity(entity)
    if not normalized:
        return []

    candidates: dict[str, AliasCandidate] = {
        normalized: AliasCandidate(alias=normalized, penalty=0.0, reason="original_normalized")
    }

    for token in _TOKEN_RE.findall(normalized):
        for alias in _char_ngram_aliases(token, n=2):
            _register_candidate(candidates, alias, 0.08, "ngram")
        for alias in _char_ngram_aliases(token, n=3):
            _register_candidate(candidates, alias, 0.12, "ngram")

        for typo in _keyboard_typo_aliases(token):
            _register_candidate(candidates, typo, 0.18, "keyboard_typo")

    for alias in _edit_distance_aliases(normalized):
        _register_candidate(candidates, alias, 0.22, "edit_distance")

    ranked = sorted(candidates.values(), key=lambda c: (c.penalty, -len(c.alias), c.alias))
    return ranked


def expansion_factor(query: str, alias: str) -> float:
    ratio = SequenceMatcher(None, query, alias).ratio()
    length_gap = abs(len(query) - len(alias)) / max(len(query), 1)
    return max(ratio - length_gap * 0.2, 0.0)


def _register_candidate(
    store: dict[str, AliasCandidate],
    alias: str,
    penalty: float,
    reason: str,
) -> None:
    cleaned = normalize_masked_entity(alias)
    if len(cleaned) < 2:
        return
    prev = store.get(cleaned)
    if prev is None or penalty < prev.penalty:
        store[cleaned] = AliasCandidate(alias=cleaned, penalty=penalty, reason=reason)


def _char_ngram_aliases(text: str, n: int) -> Sequence[str]:
    if len(text) <= n:
        return [text]
    grams = [text[i : i + n] for i in range(len(text) - n + 1)]
    return [" ".join(grams), "".join(grams)]


def _keyboard_typo_aliases(text: str) -> Iterable[str]:
    lowered = text.lower()
    for idx, ch in enumerate(lowered):
        for near in _KEYBOARD_ADJACENCY.get(ch, ""):
            yield lowered[:idx] + near + lowered[idx + 1 :]


def _edit_distance_aliases(text: str) -> Iterable[str]:
    if len(text) < 3:
        return []

    aliases: List[str] = []
    for idx in range(len(text)):
        aliases.append(text[:idx] + text[idx + 1 :])

    for idx in range(len(text) - 1):
        swapped = list(text)
        swapped[idx], swapped[idx + 1] = swapped[idx + 1], swapped[idx]
        aliases.append("".join(swapped))

    return aliases
