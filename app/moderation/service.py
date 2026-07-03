from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

import yaml
from openai import AsyncOpenAI
from pydantic import BaseModel

from app.observability.pii import prompt_hash, redact_pii

logger = logging.getLogger(__name__)

_OPENAI_MODEL = "omni-moderation-latest"
_DEFAULT_THRESHOLD = 0.5
_LOG_TEXT_MAX_CHARS = 200


class ModerationResult(BaseModel):
    allowed: bool
    categories: list[str] = []
    reasons: list[str] = []
    blocked_by: str = ""


@lru_cache
def _load_keyword_categories(path: Path) -> dict[str, list[re.Pattern[str]]]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return {
        category: [re.compile(re.escape(kw), re.IGNORECASE) for kw in spec.get("keywords", [])]
        for category, spec in raw.get("categories", {}).items()
    }


class ModerationService:
    def __init__(
        self,
        keywords_path: Path,
        use_openai_api: bool = False,
        openai_client: AsyncOpenAI | None = None,
        category_thresholds: dict[str, float] | None = None,
    ) -> None:
        self._keyword_categories = _load_keyword_categories(keywords_path)
        self._use_openai_api = use_openai_api
        self._client = openai_client
        self._thresholds = category_thresholds or {}

    async def check_input(self, content: str) -> ModerationResult:
        return await self._check(content, direction="input")

    async def check_output(self, content: str) -> ModerationResult:
        return await self._check(content, direction="output")

    async def _check(self, content: str, direction: str) -> ModerationResult:
        result = self._check_keywords(content)
        if not result.allowed:
            self._log_incident(content, result, direction)
            return result

        if self._use_openai_api and self._client is not None:
            try:
                result = await self._check_openai_api(content)
            except Exception as exc:  # noqa: BLE001 - moderation API outage must not block chat
                logger.warning("moderation_api_unavailable error=%s", exc)
                return ModerationResult(allowed=True)
            if not result.allowed:
                self._log_incident(content, result, direction)
                return result

        return ModerationResult(allowed=True)

    def _check_keywords(self, content: str) -> ModerationResult:
        matched_categories: list[str] = []
        reasons: list[str] = []
        for category, patterns in self._keyword_categories.items():
            for pattern in patterns:
                if pattern.search(content):
                    matched_categories.append(category)
                    reasons.append(f"keyword match in category {category!r}")
                    break

        if matched_categories:
            return ModerationResult(
                allowed=False, categories=matched_categories, reasons=reasons, blocked_by="keyword"
            )
        return ModerationResult(allowed=True)

    async def _check_openai_api(self, content: str) -> ModerationResult:
        response = await self._client.moderations.create(model=_OPENAI_MODEL, input=content)
        result = response.results[0]

        flagged_categories: list[str] = []
        for category, score in result.category_scores.model_dump().items():
            threshold = self._thresholds.get(category, _DEFAULT_THRESHOLD)
            if score >= threshold:
                flagged_categories.append(category)

        if flagged_categories:
            return ModerationResult(
                allowed=False,
                categories=flagged_categories,
                reasons=[f"omni-moderation score above threshold for {c!r}" for c in flagged_categories],
                blocked_by="openai_moderation",
            )
        return ModerationResult(allowed=True)

    def _log_incident(self, content: str, result: ModerationResult, direction: str) -> None:
        masked = redact_pii(content)[:_LOG_TEXT_MAX_CHARS]
        logger.warning(
            "moderation_blocked",
            extra={
                "direction": direction,
                "text_hash": prompt_hash(content),
                "text_masked": masked,
                "categories": result.categories,
                "blocked_by": result.blocked_by,
            },
        )
