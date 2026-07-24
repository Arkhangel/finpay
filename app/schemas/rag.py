from pydantic import BaseModel, Field


class RagQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class RagSource(BaseModel):
    id: int
    file_name: str | None
    page: int | None
    score: float | None
    snippet: str


class RagQueryResponse(BaseModel):
    answer: str
    top_score: float
    confident: bool
    sources: list[RagSource]
