# Модуль чата — архитектура и API

## Архитектура

```mermaid
graph TD
    Client -->|HTTP| routes["app/chat/routes.py\n/chats endpoints"]
    routes --> ChatService["ChatService\napp/chat/service.py"]
    ChatService --> ChatRepository["ChatRepository\n(Protocol)"]
    ChatService --> llm_client["AsyncOpenAI\n(LLM)"]
    ChatService -->|retrieve| RAGService["RAGService.retrieve()\napp/services/rag.py"]
    ChatRepository -->|json| JsonRepo["JsonChatRepository\nфайлы: var/chats/"]
    ChatRepository -->|postgres| PgRepo["PostgresChatRepository\nasync SQLAlchemy 2.x"]
    JsonRepo --> FS[(Файловая система)]
    PgRepo --> PG[(PostgreSQL)]
    llm_client --> Groq["Groq / OpenAI API"]
    RAGService --> Qdrant[(Qdrant\nfinpay_kb)]
```

## Системный промпт по умолчанию

`ChatService.create_chat` без явного `system_prompt` (например, звонки из
бота — `backend_client.get_or_create_chat` его не передаёт) раньше оставлял
чат с «голой» LLM без персоны FinPay и без правила честного отказа —
`render_system_prompt`/`app/prompts/system_v1.j2` были подключены только в
старом `app/routers/chat.py` (М1-М3), но не в этом (М4) сервисе. Теперь
`create_chat` при `system_prompt is None` рендерит
`render_system_prompt(project_name=settings.project_name)` и сохраняет его
в `chats.system_prompt` — эффект наблюдался вживую: до фикса бот отвечал на
«Привет» как безликий ассистент («Я — языковая модель AI...»), после —
персоной FinPay.

## RAG в чате (блок 5.5)

`ChatService.send_message` на каждый ход (если подключён `RAGService`,
`app.state.rag_service is not None`) дополнительно:

1. Переписывает follow-up в самостоятельный вопрос для retrieval
   (`_condense_query`, включается `chat.rag_condense_enabled`, по умолчанию
   `true`) — генерации это не касается, туда история уходит целиком.
2. Вызывает `RAGService.retrieve()` — retrieval → опциональный re-ranking →
   score-guard.
3. **Низкий score НЕ обрывает генерацию** (в отличие от `/rag/query`, где
   код-гард полностью пропускает LLM-вызов) — вместо этого просто не
   добавляется RAG-контекст, и LLM отвечает по обычному системному промпту
   с его собственным правилом честного отказа. Причина отклонения от
   `/rag/query`: в чат приходят не только фактические вопросы к базе, но и
   приветствия/small-talk («Привет», «тест», «спасибо») — жёсткий код-гард
   отвечал бы на них фиксированной фразой «По базе не нашёл, могу
   эскалировать», даже не спросив LLM. Живая проверка (прод-логи): именно
   так и происходило до этого исправления — на «Привет»/«тест»/«расскажи о
   сервисе» бот всегда отвечал одной и той же фразой. Числовая калибровка
   самого порога (`app/settings/rag.py`) и почему он такой низкий (0.005) —
   `docs/rag.md`, раздел «Threshold для отказа».
4. Если уверенно — в сообщения перед последним (текущим) вопросом
   добавляется системный блок с пронумерованным контекстом `[1]`, `[2]` и
   инструкцией цитировать; дальше — обычная генерация и стриминг.

Источники всегда прокидываются наружу в `result["sources"]` (даже при
низком score — видно, что было найдено, но не прошло порог, RAG-контекст
в генерацию не пошёл), и SSE-стрим дополняется финальным именованным
событием:

```
event: sources
data: {"sources": [{"id":1,"file_name":"05_refunds.md","page":1,"score":0.9,"snippet":"..."}]}
```

Если `RAGService` не подключён (например, `scripts/ingest.py` ещё не
запускался и коллекция `finpay_kb` не существует), `event: sources` просто
не отправляется — остальной чат работает как раньше, без RAG.

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
Тело — `multipart/form-data`: `content` (обязательно) и `media` (опционально,
файл — картинка/голос/PDF/DOCX). Отдельного `/messages/with-media` нет.

```bash
curl -N -X POST http://localhost:8000/chats/<chat_id>/messages \
  -F 'content=Привет, меня зовут Аня'
# data: {"type":"token","delta":"Привет"}
# data: {"type":"token","delta":", Аня"}
# ...
# event: sources
# data: {"sources":[...]}          ← только если подключён RAGService
# data: {"type":"done","message_id":"<uuid ассистентского сообщения>"}
```

`\n` внутри `delta` не ломает формат SSE: `json.dumps` экранирует его как
два символа `\n` в JSON-строке, сырой перевод строки никогда не попадает в
`data:`-строку целиком (см. `app/chat/routes.py::generator`).

`message_id` из `done` используется, например, ботом для клавиатуры фидбека
👍/👎 (`POST /messages/{message_id}/feedback`, см. ниже).

Перед стримом content проверяется `ModerationService.check_input` — если
заблокирован, ответ `403 {"code": "moderation_blocked", "categories": [...]}`
и SSE-стрим даже не открывается. Ответ модели после сборки полностью
проверяется `check_output`: если нарушает правила, в историю пишется и
дополнительно стримится заглушка "Не могу показать ответ — он мог нарушить
правила" (см. `app/moderation/`, `app/chat/service.py`).

С вложением (фото/голос/PDF/DOCX — см. `app/chat/media.py::media_to_part`):

```bash
curl -N -X POST http://localhost:8000/chats/<chat_id>/messages \
  -F 'content=что на фото?' \
  -F 'media=@photo.jpg;type=image/jpeg'
```

MIME-диспатч (без FFmpeg/subprocess):

| MIME | Content-part |
|------|--------------|
| `image/*` | `image_url` напрямую в `chat.completions` |
| `audio/*`, `application/ogg` | `text` — расшифровка через Whisper-1 |
| `application/pdf` | `text` — извлечённый текст (`pypdf`, до 50 страниц) |
| `*wordprocessingml.document` (docx) | `text` — извлечённый текст (`python-docx`) |
| неизвестный MIME | `415 Unsupported media type` |

Вложение сохраняется в `ChatMessage.media_refs` (mime/size/filename/`part`) и
восстанавливается как мультимодальный `content` при последующих LLM-вызовах
того же чата — модель может «пересмотреть» исходное фото в следующей реплике.

### POST `/chats/{chat_id}/system-message`
Демо-эндпоинт для симуляции завершения фоновой задачи (статус заявки,
подписка и т.п.). Дописывает сообщение от лица ассистента и, если
`notify: true`, шлёт проактивное уведомление в Telegram через бота (см. ниже).

```bash
curl -X POST http://localhost:8000/chats/<chat_id>/system-message \
  -H 'Content-Type: application/json' \
  -d '{"text": "Ваша заявка #123 обработана", "notify": true}'
```

## Telegram-бот: стриминг ответа (отклонение от Б5.5-задания)

Задание Б5.5 описывает стриминг в Telegram через `editMessageText` с ручным
дросселированием (`last_edit_at` per `chat_id`, ≥700–1000мс между вызовами,
чтобы не упереться в лимит Telegram ~1 edit/сек на сообщение → `429 Too Many
Requests`).

В проекте эта механика уже была реализована иначе — через нативный
`sendMessageDraft` (Bot API 10.0, `app/bot/services/streaming.py::stream_to_chat`):
Telegram сам буферизует черновик на своей стороне и не считает вызовы
`sendMessageDraft` per-message rate-limit'ом `editMessageText`, поэтому
проблема 429 не возникает в принципе, и вручную реализованный debounce не
нужен — `stream_to_chat` вызывает `send_message_draft` на каждый чанк без
троттлинга. Итоговое сообщение отправляется один раз, обычным
`send_message`, после чего на него вешается клавиатура фидбека.

Осознанно оставлено как есть: `editMessageText`+debounce и
`sendMessageDraft` решают одну и ту же задачу (плавный вывод ответа по
токенам), но `sendMessageDraft` — более новый и надёжный механизм именно
потому, что убирает источник 429 полностью, а не смягчает его частотой
вызовов. Бот отвечает через RAG-контур (`POST /chats/{id}/messages`,
`app/bot/handlers/text.py`) так же, как и REST-клиенты; после стрима боту
приходит `result["sources"]` (см. `app/bot/services/backend_client.py`) —
источники кэшируются по `message_id` в `app/bot/services/sources_cache.py`
(callback-кнопки Telegram ограничены ~64 байтами и не могут нести список
источников внутри себя) и прикладываются к `POST .../feedback` при нажатии
👍/👎.

## Обратный канал backend → bot (`/notify`)

Бот поднимает внутренний FastAPI рядом с polling (`app/bot/web.py`,
`modes/bot.py`), на порту `bot.bot_api_port` (по умолчанию 9000). Backend
дёргает его через `app/services/notifier.py::notify_user` для проактивных
уведомлений.

```bash
curl -X POST http://localhost:9000/notify \
  -H 'X-Internal-Token: <bot.internal_token>' \
  -H 'Content-Type: application/json' \
  -d '{"chat_id": 123456789, "text": "Готово!"}'
```

Без верного `X-Internal-Token` эндпоинт отвечает `401`. Токен задаётся через
`BOT__INTERNAL_TOKEN` (env) или `.config/local.toml`, в репозиторий не
коммитится.

### POST `/chats/{chat_id}/messages/{message_id}/feedback`
Сохранить оценку ответа ассистента (`up`/`down`). Дедуп по
`(owner_external_id, message_id)` — `owner_external_id` берётся из чата, а не
из тела запроса; повторный голос того же пользователя по тому же сообщению
не создаёт вторую запись (`recorded: false`).

`sources` (блок 5.5, опционально) — список источников, показанных вместе с
оценённым ответом; сохраняется как есть для аудита ("какой ответ с какими
source получил дизлайк"), в дедупликации не участвует.

```bash
curl -X POST http://localhost:8000/chats/<chat_id>/messages/<message_id>/feedback \
  -H 'Content-Type: application/json' \
  -d '{"value": "up", "sources": [{"id": 1, "file_name": "05_refunds.md", "page": 1, "score": 0.9, "snippet": "..."}]}'
```

### Admin API

`/chats/admin/*` (stats/users/broadcast) требует Postgres и заголовок
`X-Admin-Token` — подробности в разделе «Production-обвязка» в [`README.md`](../README.md).

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
    media_refs JSONB,       -- {mime, size, filename, part} — см. app/chat/media.py
    latency_ms INT,         -- время генерации ответа ассистента (для admin/stats)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMPTZ  -- NULL = активное, NOT NULL = soft-deleted
);

-- Б4.4: production-обвязка
CREATE TABLE message_feedback (
    id UUID PRIMARY KEY,
    message_id UUID NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
    owner_external_id TEXT NOT NULL,
    value TEXT NOT NULL,    -- "up" | "down"
    sources JSONB,          -- Б5.5: источники, показанные вместе с ответом (nullable)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (owner_external_id, message_id)
);

CREATE TABLE moderation_incidents (
    id UUID PRIMARY KEY,
    chat_id UUID NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    direction TEXT NOT NULL,     -- "input" | "output"
    blocked_by TEXT NOT NULL,    -- "keyword" | "openai_moderation"
    categories JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE broadcast_queue (
    id UUID PRIMARY KEY,
    message TEXT NOT NULL,
    interface TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | sent | failed
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMPTZ
);
```

Строки физически не удаляются — soft-delete выставляет `deleted_at`.
