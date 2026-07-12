from pathlib import Path

from pydantic import BaseModel


class EmbeddingsSettings(BaseModel):
    model: str = "intfloat/multilingual-e5-base"
    device: str = "cpu"
    batch_size: int = 32
    cache_dir: Path = Path(".cache/embeddings")
    dim: int = 768