class LLMError(Exception):
    code: str = "llm_error"
    message: str = "LLM error"


class LLMRateLimitError(LLMError):
    code = "llm_rate_limit"
    message = "Rate limit exceeded"


class LLMTimeoutError(LLMError):
    code = "llm_timeout"
    message = "LLM request timed out"


class LLMAuthError(LLMError):
    code = "llm_auth"
    message = "Authentication failed"
