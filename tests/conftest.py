import os

# app/llm/client.py создаёт OpenAI(api_key=...) на уровне модуля.
# openai>=2.0 выбрасывает OpenAIError при пустом ключе ещё до создания объекта.
# Этот setdefault выполняется при сборе тестов (до импорта test_llm_agent.py),
# поэтому фиктивный ключ предотвращает ошибку — реальных сетевых вызовов нет.
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")
