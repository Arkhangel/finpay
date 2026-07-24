# syntax=docker/dockerfile:1.7
FROM python:3.13-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.6.10 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Эмбеддинг-модель (М5.1) — качаем на этапе сборки образа, а не при первом
# запросе в проде: контейнер стартует без сетевой зависимости от HuggingFace.
# Модель должна совпадать с default в app/settings/embeddings.py.
# Путь HF_HOME должен совпадать в runtime-стадии ниже, иначе кеш не найдётся.
ENV HF_HOME=/app/.cache/huggingface
RUN /app/.venv/bin/python -c \
    "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base')"

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.13-slim-bookworm AS runtime

RUN useradd --create-home --uid 1000 appuser

# .venv (torch/llama-index, GB+) и .cache (эмбеддинг-модель) — отдельными
# слоями от кода: при изменении .py-файлов пересоздаётся только маленький
# слой ниже, а не вся venv целиком. Без этого разделения любой рестарт
# исходников заново материализует ~11GB на диск (было: 79G -> 100% диска
# за две пересборки подряд).
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/.cache /app/.cache
COPY --chown=appuser:appuser . /app

USER appuser

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface

EXPOSE 8000

CMD ["python", "main.py"]
