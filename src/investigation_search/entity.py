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
_HANGUL_TOKEN_RE = re.compile(r"[가-힣]{2,6}")
_WHITESPACE_RE = re.compile(r"\s+")

_ROLE_HINTS = {
    "강사",
    "교수",
    "선생님",
    "원장",
    "대표",
    "의사",
    "박사",
    "변호사",
    "기자",
    "작가",
    "배우",
    "가수",
    "감독",
    "코치",
    "목사",
    "신부",
    "법사",
    "강연자",
    "유튜버",
    "교사",
}
_VERIFICATION_HINTS = {"사실", "진실", "맞나", "맞아", "확인", "검증", "팩트", "fact", "verify", "true", "false"}
_COMPARISON_HINTS = {"비교", "차이", "vs", "versus", "대비", "다른점", "difference"}
_TIMELINE_HINTS = {"언제", "시기", "연혁", "timeline", "history", "변천", "순서", "년도", "날짜"}

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


@dataclass(frozen=True)
class QueryIntent:
    kind: str = "general"
    focus_name: str = ""
    role_terms: tuple[str, ...] = ()
    exact_terms: tuple[str, ...] = ()
    quoted_terms: tuple[str, ...] = ()
    required_terms: tuple[str, ...] = ()
    allow_alias_expansion: bool = True

    @property
    def is_entity_lookup(self) -> bool:
        return self.kind in {"entity_lookup", "person_lookup"}

    @property
    def requires_exact_name(self) -> bool:
        return bool(self.focus_name) and self.is_entity_lookup

    @property
    def is_exact_lookup(self) -> bool:
        return self.kind in {"entity_lookup", "person_lookup", "exact_phrase"}


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


def classify_query_intent(query: str) -> QueryIntent:
    raw = str(query or "").strip()
    if not raw:
        return QueryIntent()

    tokens = [token for token in _WHITESPACE_RE.split(raw) if token]
    hangul_tokens = [token for token in tokens if _HANGUL_TOKEN_RE.fullmatch(token)]
    role_terms = tuple(token for token in tokens if token in _ROLE_HINTS)
    quoted_terms = tuple(match.group(1) for match in re.finditer(r'"([^"]+)"', raw))

    focus_name = ""
    if hangul_tokens:
        candidate_names = [token for token in hangul_tokens if token not in role_terms]
        if candidate_names:
            focus_name = max(candidate_names, key=len)

    exact_terms = tuple(dict.fromkeys([raw, *quoted_terms, *(term for term in (focus_name,) if term)]))

    exact_terms = tuple(dict.fromkeys([raw, *quoted_terms, *(term for term in (focus_name,) if term)]))
    required_terms = tuple(dict.fromkeys([token for token in tokens if token not in {"vs", "versus"}]))

    lowered_tokens = {token.lower() for token in tokens}
    if quoted_terms:
        return QueryIntent(
            kind="exact_phrase",
            focus_name=focus_name,
            role_terms=role_terms,
            exact_terms=exact_terms,
            quoted_terms=quoted_terms,
            required_terms=required_terms,
            allow_alias_expansion=False,
        )
    if lowered_tokens & _COMPARISON_HINTS:
        return QueryIntent(
            kind="comparison",
            focus_name=focus_name,
            role_terms=role_terms,
            exact_terms=exact_terms,
            quoted_terms=quoted_terms,
            required_terms=required_terms,
            allow_alias_expansion=False,
        )
    if lowered_tokens & _TIMELINE_HINTS:
        return QueryIntent(
            kind="timeline",
            focus_name=focus_name,
            role_terms=role_terms,
            exact_terms=exact_terms,
            quoted_terms=quoted_terms,
            required_terms=required_terms,
            allow_alias_expansion=False,
        )
    if lowered_tokens & _VERIFICATION_HINTS:
        return QueryIntent(
            kind="verification",
            focus_name=focus_name,
            role_terms=role_terms,
            exact_terms=exact_terms,
            quoted_terms=quoted_terms,
            required_terms=required_terms,
            allow_alias_expansion=False,
        )
    if focus_name and role_terms:
        return QueryIntent(
            kind="person_lookup",
            focus_name=focus_name,
            role_terms=role_terms,
            exact_terms=exact_terms,
            quoted_terms=quoted_terms,
            required_terms=required_terms,
            allow_alias_expansion=False,
        )
    if focus_name and len(tokens) <= 3 and len(focus_name) <= 4:
        return QueryIntent(
            kind="entity_lookup",
            focus_name=focus_name,
            role_terms=role_terms,
            exact_terms=exact_terms,
            quoted_terms=quoted_terms,
            required_terms=required_terms,
            allow_alias_expansion=False,
        )
    return QueryIntent(
        kind="general",
        focus_name=focus_name if len(focus_name) >= 2 else "",
        role_terms=role_terms,
        exact_terms=exact_terms if quoted_terms else (),
        quoted_terms=quoted_terms,
        required_terms=required_terms,
        allow_alias_expansion=True,
    )


def extract_name_candidates(text: str) -> list[str]:
    clean = str(text or "")
    return list(dict.fromkeys(match.group(0) for match in _HANGUL_TOKEN_RE.finditer(clean)))


def entity_match_features(query: str, text: str, *, title: str = "") -> dict[str, float]:
    intent = classify_query_intent(query)
    combined = f"{title} {text}".strip()
    features: dict[str, float] = {
        "intent_entity_lookup": 1.0 if intent.is_entity_lookup else 0.0,
        "exact_query_match": 0.0,
        "exact_phrase_match": 0.0,
        "exact_name_match": 0.0,
        "title_exact_query_match": 0.0,
        "title_exact_name_match": 0.0,
        "role_name_cooccurrence": 0.0,
        "all_term_coverage": 0.0,
        "ordered_span_match": 0.0,
        "similar_name_penalty": 0.0,
    }
    if not combined:
        return features

    lowered = combined.lower()
    raw_query = str(query or "").strip().lower()
    title_lower = str(title or "").strip().lower()
    if raw_query and raw_query in lowered:
        features["exact_query_match"] = 1.0
    if raw_query and raw_query in title_lower:
        features["title_exact_query_match"] = 1.0
    for phrase in intent.quoted_terms or intent.exact_terms:
        phrase_lower = phrase.strip().lower()
        if phrase_lower and phrase_lower in lowered:
            features["exact_phrase_match"] = 1.0
            break

    required_terms = [term.lower() for term in intent.required_terms if term.strip()]
    if required_terms:
        matched_terms = [term for term in required_terms if term in lowered]
        features["all_term_coverage"] = len(matched_terms) / max(len(required_terms), 1)
        positions = [lowered.find(term) for term in matched_terms if lowered.find(term) >= 0]
        if len(positions) >= 2 and (max(positions) - min(positions)) <= 48:
            features["ordered_span_match"] = 1.0

    if not intent.focus_name:
        return features

    focus_name = intent.focus_name.lower()
    if focus_name in lowered:
        features["exact_name_match"] = 1.0
    if focus_name and focus_name in title_lower:
        features["title_exact_name_match"] = 1.0
    if intent.role_terms and all(term.lower() in lowered for term in intent.role_terms) and focus_name in lowered:
        features["role_name_cooccurrence"] = 1.0

    if features["exact_name_match"] < 1.0:
        for candidate in extract_name_candidates(combined):
            candidate_lower = candidate.lower()
            if candidate_lower == focus_name:
                continue
            ratio = SequenceMatcher(None, focus_name, candidate_lower).ratio()
            if ratio >= 0.66:
                features["similar_name_penalty"] = max(features["similar_name_penalty"], ratio)
    return features
