from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, List, Sequence


_WORD_RE = re.compile(r"[a-zA-Z0-9가-힣]+")
_KO_RE = re.compile(r"[가-힣]")
_EN_RE = re.compile(r"[a-zA-Z]")
_SPACE_RE = re.compile(r"\s+")


def detect_language(text: str) -> str:
    if not text:
        return "unknown"
    ko = len(_KO_RE.findall(text))
    en = len(_EN_RE.findall(text))
    if ko == 0 and en == 0:
        return "unknown"
    if ko > 0 and en == 0:
        return "ko"
    if en > 0 and ko == 0:
        return "en"
    ratio = ko / max(en, 1)
    if ratio > 1.6:
        return "ko"
    if ratio < 0.6:
        return "en"
    return "mixed"


def tokenize(
    text: str,
    *,
    mode: str = "search",
    use_morphology: bool = False,
    include_char_ngrams: bool | None = None,
    ngram_range: Sequence[int] = (2, 3),
) -> List[str]:
    normalized = _SPACE_RE.sub(" ", text.strip().lower())
    if not normalized:
        return []

    tokens = [m.group(0) for m in _WORD_RE.finditer(normalized)]
    if use_morphology:
        morph = _morphological_tokens(normalized)
        if morph:
            tokens.extend(morph)

    lang = detect_language(normalized)
    if include_char_ngrams is None:
        include_char_ngrams = mode in {"search", "bm25", "lexical", "rerank"}

    if include_char_ngrams and lang in {"ko", "mixed"}:
        grams: list[str] = []
        for token in tokens:
            if len(token) < 2:
                continue
            if not _KO_RE.search(token):
                continue
            for n in ngram_range:
                if n <= 1 or len(token) < n:
                    continue
                for i in range(len(token) - n + 1):
                    grams.append(f"cg:{token[i:i+n]}")
        tokens.extend(grams)

    return tokens


def unique_tokens(
    text: str,
    *,
    mode: str = "search",
    use_morphology: bool = False,
    include_char_ngrams: bool | None = None,
    ngram_range: Sequence[int] = (2, 3),
) -> set[str]:
    return set(
        tokenize(
            text,
            mode=mode,
            use_morphology=use_morphology,
            include_char_ngrams=include_char_ngrams,
            ngram_range=ngram_range,
        )
    )


@lru_cache(maxsize=1)
def _load_okt():
    try:
        from konlpy.tag import Okt  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        return Okt()
    except Exception:
        return None


def _morphological_tokens(text: str) -> Iterable[str]:
    okt = _load_okt()
    if okt is None:
        return []
    try:
        nouns = okt.nouns(text)
    except Exception:
        return []
    return [token.lower() for token in nouns if len(token) >= 2]
