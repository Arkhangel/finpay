"""Input validation layer: block prompt injection and encoding attacks.

Decision: raise HTTPException(400) on blocked input.
This causes garak to see no "content" field in the response and mark the
probe as defended — which is exactly the goal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

# ── Injection patterns ───────────────────────────────────────────────────────
# Each pattern is compiled once; IGNORECASE covers mixed-case bypass attempts.

INJECTION_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(
        r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdisregard\s+(the\s+)?(system|previous|above)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\byou\s+are\s+now\s+(a|an|the|dan|do\s+anything\s+now)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bforget\s+(everything|all|previous)\b", re.IGNORECASE),
    re.compile(r"\b(jailbroken|developer\s+mode|godmode|DAN)\b", re.IGNORECASE),
    # Base64-encoded payload markers
    re.compile(r"\bSUdOT1JF|aWdub3Jl|FORGET\s+EVERYTHING\b", re.IGNORECASE),
    # FinPay-specific: attempts to extract internal config
    re.compile(r"\b(system\s+prompt|internal\s+instructions|confidential)\b", re.IGNORECASE),
]

MAX_INPUT_CHARS: Final[int] = 4000
NON_PRINTABLE_RATIO_LIMIT: Final[float] = 0.10


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str | None = None
    rule: str | None = None


def validate_input(text: str) -> ValidationResult:
    """Validate a single message text.

    Returns ValidationResult(ok=True) if the input is safe, or
    ValidationResult(ok=False, reason=..., rule=...) if blocked.
    """
    if len(text) > MAX_INPUT_CHARS:
        return ValidationResult(False, "input too long", rule="length")

    # Non-printable character ratio heuristic (encoding attacks)
    non_printable = sum(1 for c in text if not c.isprintable() and c not in "\n\r\t")
    if non_printable / max(len(text), 1) > NON_PRINTABLE_RATIO_LIMIT:
        return ValidationResult(False, "high non-printable ratio", rule="encoding")

    for pat in INJECTION_PATTERNS:
        if pat.search(text):
            return ValidationResult(
                False, f"matched pattern: {pat.pattern[:60]}", rule="injection"
            )

    return ValidationResult(True)
