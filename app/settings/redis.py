from pydantic import BaseModel, field_validator


class RedisSettings(BaseModel):
    url: str = "redis://localhost:6379"
    password: str | None = None
    ttl: int = 3600

    @field_validator("password", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: str | None) -> str | None:
        return v or None
