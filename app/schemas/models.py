from pydantic import BaseModel


class ModelInfo(BaseModel):
    id: str
    name: str
    context_window: int
    input_price_per_1k: float
    output_price_per_1k: float

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Return total USD cost for a single API call based on token usage."""
        return (
            prompt_tokens / 1000 * self.input_price_per_1k
            + completion_tokens / 1000 * self.output_price_per_1k
        )


AVAILABLE_MODELS: list[ModelInfo] = [
    ModelInfo(id="gpt-4o",      name="GPT-4o",      context_window=128000, input_price_per_1k=0.005,   output_price_per_1k=0.015),
    ModelInfo(id="gpt-4o-mini", name="GPT-4o Mini", context_window=128000, input_price_per_1k=0.00015, output_price_per_1k=0.0006),
    ModelInfo(id="gpt-3.5-turbo", name="GPT-3.5 Turbo", context_window=16385, input_price_per_1k=0.0005, output_price_per_1k=0.0015),
    ModelInfo(id="llama-3.3-70b-versatile", name="Llama 3.3 70B (Groq)", context_window=128000, input_price_per_1k=0.00059, output_price_per_1k=0.00079),
]
