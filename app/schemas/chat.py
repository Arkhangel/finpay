from __future__ import annotations

from pydantic import BaseModel, Field

from app.observability.pii import redact_pii


class Message(BaseModel):
    role: str
    content: str = Field(min_length=1, max_length=32000)

    def __repr__(self) -> str:
        return f"Message(role={self.role!r}, content={redact_pii(self.content)!r})"


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    model: str | None = None
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1024, ge=1, le=16000)
    user_id: str | None = None
    session_id: str | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "messages": [{"role": "user", "content": "Что такое event loop?"}],
                    "temperature": 0.7,
                    "max_tokens": 512,
                },
                {
                    "messages": [
                        {"role": "system", "content": "Ты ассистент техподдержки FinPay."},
                        {"role": "user", "content": "Какой статус у транзакции TXN-1002?"},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 256,
                    "session_id": "ses_abc123",
                },
            ]
        }
    }


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    content: str
    model: str
    usage: Usage
    finish_reason: str
    cached: bool = False

    @classmethod
    def from_openai(cls, response, *, cached: bool = False) -> ChatResponse:
        choice = response.choices[0]
        u = response.usage
        return cls(
            content=(choice.message.content or "").strip(),
            model=response.model,
            usage=Usage(
                prompt_tokens=u.prompt_tokens if u else 0,
                completion_tokens=u.completion_tokens if u else 0,
                total_tokens=u.total_tokens if u else 0,
            ),
            finish_reason=choice.finish_reason or "stop",
            cached=cached,
        )


class ChatDelta(BaseModel):
    content: str | None = None
    usage: Usage | None = None
