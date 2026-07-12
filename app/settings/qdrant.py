from pydantic import BaseModel


class QdrantSettings(BaseModel):
    url: str = "http://localhost:6333"
    api_key: str | None = None
    collection: str = "documents"
