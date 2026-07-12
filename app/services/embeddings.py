from __future__ import annotations

import hashlib
import logging

import diskcache
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.settings import settings

logger = logging.getLogger(__name__)

cache = diskcache.Cache(str(settings.embeddings.cache_dir))

_model = None

_retry_on_download_error = retry(
    retry=retry_if_exception_type((OSError, ConnectionError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    reraise=True,
)


@_retry_on_download_error
def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("embeddings_model_loading", extra={"model": settings.embeddings.model})
        _model = SentenceTransformer(settings.embeddings.model, device=settings.embeddings.device)
    return _model


def _cache_key(text: str) -> str:
    raw = f"{settings.embeddings.model}:{text}"
    return "emb:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    results: list[list[float] | None] = [None] * len(texts)
    to_encode: list[str] = []
    positions: list[int] = []

    for i, text in enumerate(texts):
        cached = cache.get(_cache_key(text))
        if cached is not None:
            results[i] = cached
        else:
            to_encode.append(text)
            positions.append(i)

    logger.info(
        "embed_texts",
        extra={"total": len(texts), "cache_hits": len(texts) - len(to_encode), "to_encode": len(to_encode)},
    )

    if to_encode:
        model = _get_model()
        vectors = model.encode(
            to_encode,
            batch_size=settings.embeddings.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        for pos, text, vector in zip(positions, to_encode, vectors):
            vector_list = [float(x) for x in vector]
            results[pos] = vector_list
            cache.set(_cache_key(text), vector_list)

    return results  # type: ignore[return-value]


def embed_query(text: str) -> list[float]:
    return embed_texts([f"query: {text}"])[0]


def embed_documents(texts: list[str]) -> list[list[float]]:
    return embed_texts([f"passage: {t}" for t in texts])