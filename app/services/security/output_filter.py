"""Output filter: detect system-prompt leakage and mask PII in LLM responses.

Raises ValueError on leakage (caller should return HTTP 502).
Returns the (possibly masked) answer on success.
"""
from __future__ import annotations

import re
from typing import Final

from app.observability.pii import redact_pii

# ── PII patterns (aligned with app/observability/pii.py placeholders) ────────
# We re-use the same placeholder names so logs and responses are consistent.

_EMAIL_RE: Final = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RU_RE: Final = re.compile(
    r"(?<!\w)(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\w)"
)
_PASSPORT_RU_RE: Final = re.compile(r"\b\d{4}\s?\d{6}\b")
# FinPay-specific: card numbers are already covered by pii.py [CARD],
# but INN (10/12 digits) needs a separate pass here to avoid false positives.
_INN_RE: Final = re.compile(r"\b(?:\d{10}|\d{12})\b")


def filter_output(answer: str, system_prompt: str, canary: str) -> str:
    """Redact PII and detect system-prompt leakage.

    Raises:
        ValueError: if the canary token or the system-prompt prefix is found
                    in the answer — indicates prompt leakage.

    Usage in /chat handler:
        try:
            content = filter_output(response.content, system_prompt, app.state.canary)
        except ValueError as e:
            raise HTTPException(status_code=502, detail=str(e))
    """
    # 1. Canary check — exact token planted in system prompt
    if canary and canary in answer:
        raise ValueError("system_prompt leakage: canary detected")

    # 2. Prefix check — first 80 normalised chars of system prompt in answer
    head = " ".join(system_prompt.split())[:80]
    if head and head.lower() in " ".join(answer.split()).lower():
        raise ValueError("system_prompt leakage: prefix detected")

    # 3. PII masking — use shared redact_pii for email/phone/card/inn,
    #    then add passport pattern not covered by pii.py
    masked = redact_pii(answer)
    masked = _PASSPORT_RU_RE.sub("[PASSPORT]", masked)

    return masked
