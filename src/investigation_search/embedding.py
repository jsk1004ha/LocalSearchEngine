from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Literal, Sequence

import numpy as np


DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


@lru_cache(maxsize=4)
def load_local_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """CPU 환경에서 동작 가능한 sentence-transformers 모델을 로드한다."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "sentence-transformers가 필요합니다. `pip install sentence-transformers` 후 재시도하세요."
        ) from exc

    return SentenceTransformer(model_name, device="cpu")


def _prefix_for(model_name: str, text_type: Literal["query", "passage"]) -> str:
    lower = model_name.lower()
    if "e5" in lower:
        return "query: " if text_type == "query" else "passage: "
    return ""


def encode_texts(
    texts: Sequence[str] | Iterable[str],
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    text_type: Literal["query", "passage"] = "passage",
    batch_size: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    model = load_local_model(model_name)
    seq = list(texts)
    if not seq:
        return np.zeros((0, 0), dtype=np.float32)

    prefix = _prefix_for(model_name, text_type)
    model_inputs = [f"{prefix}{text}" for text in seq]
    vectors = model.encode(
        model_inputs,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return np.asarray(vectors, dtype=np.float32)
