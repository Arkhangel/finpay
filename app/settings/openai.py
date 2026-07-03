from pydantic import BaseModel


class OpenAISettings(BaseModel):
    api_key: str = ""
    host: str = ""
    model: str = ""
    whisper_model: str = "whisper-1"
