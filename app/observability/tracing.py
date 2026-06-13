import os

from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.otel import register


def setup_tracing(project_name: str = "finpay") -> None:
    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:4317")
    tracer_provider = register(
        project_name=project_name,
        endpoint=endpoint,
        protocol="grpc",
    )
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
