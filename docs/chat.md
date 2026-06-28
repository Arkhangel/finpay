# Модуль чата — архитектура и API

## Архитектура

```mermaid
graph TD
    Client -->|HTTP| routes["app/chat/routes.py\n/chats endpoints"]
    routes --> ChatService["ChatService\napp/chat/service.py"]
    ChatService --> ChatRepository["ChatRepository\n(Protocol)"]
    ChatService --> llm_client["AsyncOpenAI\n(LLM)"]
    ChatRepository -->|json| JsonRepo["JsonChatRepository\nфайлы: var/chats/"]
    ChatRepository -->|postgres| PgRepo["PostgresChatRepository\nasync SQLAlchemy 2.x"]
    JsonRepo --> FS[(Файловая система)]
    PgRepo --> PG[(PostgreSQL)]
    llm_client --> Groq["Groq / OpenAI API"]
```

## Стратегия контекста

**Выбранная стратегия: Скользящее окно (Sliding Window)**

На каждом шаге сервис берёт последние `CHAT_CONTEXT_WINDOW` сообщений (по умолчанию 10)
и добавляет в начало `system_prompt` чата (если задан).
Если итоговое количество токенов превышает бюджет
(`CONTEXT_WINDOW − RESPONSE_TOKENS − SAFETY_MARGIN`), старые сообщения обрезаются,
системное сообщение при этом всегда сохраняется.

**Обоснование для FinPay:**
FinPay — сервис поддержки платёжного шлюза: FAQ и статусы транзакций.
Диалоги короткие (1–4 хода), вопросы самодостаточны, контекст из далёкого прошлого
практически не нужен. Скользящее окно с N=10 покрывает все реальные сценарии,
не добавляя ни задержки, ни лишних токенов.
Гибридная стратегия (summary + последние M) потребовала бы дополнительного LLM-вызова
на каждом ходу — избыточно для FAQ-бота.

## Эндпоинты

### POST `/chats`
Создать новый чат.

```bash
curl -X POST http://localhost:8000/chats \
  -H 'Content-Type: application/json' \
  -d '{"owner_external_id": "tg-12345", "interface": "telegram"}'
# → {"chat_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"}
```

### GET `/chats/{chat_id}`
Получить метаданные чата.

```bash
curl http://localhost:8000/chats/<chat_id>
```

### POST `/chats/{chat_id}/messages`
Отправить сообщение пользователя; возвращает SSE-стрим токенов ответа ассистента.

```bash
curl -N -X POST http://localhost:8000/chats/<chat_id>/messages \
  -H 'Content-Type: application/json' \
  -d '{"content": "Привет, меня зовут Аня"}'
# data: Привет
# data: , Аня
# ...
# data: [DONE]
```

### GET `/chats/{chat_id}/messages?limit=50`
Получить список сообщений в хронологическом порядке.

```bash
curl "http://localhost:8000/chats/<chat_id>/messages?limit=20"
```

### DELETE `/chats/{chat_id}/messages`
Мягкое удаление всей истории (чат остаётся, сообщения скрываются).

```bash
curl -X DELETE http://localhost:8000/chats/<chat_id>/messages
# → {"status": "ok"}
```

## Переключение хранилища

Отредактируй `.config/local.toml`:

```toml
[chat]
repository = "json"    # поменяй на "postgres" для Postgres
storage_dir = "./var/chats"
context_strategy = "sliding"
context_window = 10
database_url = "postgresql+asyncpg://neto_chat:neto_chat@localhost:5432/neto_chat"
```

Запуск в обоих случаях одинаковый:

```bash
uv run main.py
```

Перед первым запуском с Postgres применить миграцию:

```bash
uv run alembic upgrade head
```

Переменные окружения (`CHAT__REPOSITORY`, `CHAT__DATABASE_URL` и др.) работают
как переопределение конфига — удобно для Docker/CI без правки файлов.

### Структура JSON-хранилища

```
var/chats/
  <chat_id>/
    chat.json          ← метаданные чата
    messages.jsonl     ← по одному сообщению на строку; маркер soft-delete отдельной строкой
```

### Схема PostgreSQL

```sql
CREATE TABLE chats (
    id UUID PRIMARY KEY,
    owner_external_id TEXT NOT NULL,
    interface TEXT NOT NULL,
    system_prompt TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE chat_messages (
    id UUID PRIMARY KEY,
    chat_id UUID NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tokens INT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMPTZ  -- NULL = активное, NOT NULL = soft-deleted
);
```

Строки физически не удаляются — soft-delete выставляет `deleted_at`.
