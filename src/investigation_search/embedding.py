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
        from .bootstrap import auto_install_enabled, ensure_installed, requirements_path

        if auto_install_enabled():
            req = requirements_path("requirements-search.txt")
            ensure_installed(
                requirements_files=[req] if req is not None else None,
                packages=("sentence-transformers",),
                auto_install=True,
            )
            from sentence_transformers import SentenceTransformer
        else:
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
    seq = list(texts)
    if not seq:
        return np.zeros((0, 0), dtype=np.float32)

    if model_name.lower().startswith("ollama:"):
        from .ollama import OllamaClient

        ollama_model = model_name.split(":", 1)[1].strip()
        if not ollama_model:
            raise ValueError("ollama 모델명은 `ollama:<model>` 형식이어야 합니다.")
        client = OllamaClient()
        vectors = np.asarray(client.embed(model=ollama_model, texts=seq), dtype=np.float32)
        if normalize and vectors.size > 0:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            vectors = vectors / norms
        return vectors

    model = load_local_model(model_name)
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
