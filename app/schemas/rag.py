from pydantic import BaseModel, Field


class RagQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class RagSource(BaseModel):
    text: str
    source: str | None
    score: float | None


class RagQueryResponse(BaseModel):
    answer: str
    top_score: float
    sources: list[RagSource]
