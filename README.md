# FinPay — AI-ассистент техподдержки

FastAPI-сервис на базе LLM для поддержки платёжного процессинга: RAG по
корпоративной базе знаний с цитированием источников, агент с function
calling (включая human-in-the-loop подтверждение опасных действий),
персистентность между рестартами, наблюдаемость через OpenTelemetry и
security-слой с защитой от prompt injection.

## Архитектура

```mermaid
flowchart LR
    U["Пользователь\n(Telegram / REST)"] --> BOT["bot (aiogram)"]
    U --> API["app (FastAPI)"]
    BOT -->|"BackendClient"| API
    API --> MOD["Модерация\n(keyword + опц. OpenAI)"]
    MOD --> CHAT["ChatService\n(история, sliding window)"]
    CHAT --> RAG["RAGService\nQdrant retrieval → reranker → score-guard"]
    CHAT --> AGENT["Agent graph (LangGraph)\ncheckpointer + HIL"]
    RAG --> QDRANT[(Qdrant\nfinpay_kb)]
    AGENT --> TOOLS["search_knowledge_base /\nget_current_time /\nsend_telegram_message"]
    TOOLS --> RAG
    AGENT --> PG[(Postgres\nchat + checkpoints)]
    CHAT --> PG
    CHAT --> REDIS[(Redis\nresponse cache)]
    API --> PHOENIX["Phoenix\n(OpenTelemetry-трейсинг)"]
```

Запрос идёт: пользователь (Telegram-бот или напрямую REST) → модерация входа
→ `ChatService` собирает контекст истории → либо прямой RAG-ответ
(`/rag/query`, одноразовый вопрос-ответ), либо агентный граф (`/agent/stream`,
LangGraph, tool calling + HIL) → LLM (Groq, OpenAI-совместимый API) → ответ с
цитированием источников `[1]`, `[2]`. Состояние диалога и чекпоинты агента —
в Postgres (переживают рестарт контейнера), ответы кешируются в Redis, весь
путь запроса виден в Phoenix через OpenTelemetry.

Подробные ADR и обоснования решений — [`docs/architecture.md`](docs/architecture.md).

## Быстрый старт

```bash
# Установить зависимости
uv sync --all-groups

# Скопировать конфиг и вписать API-ключ
cp .env.example .env
# отредактировать .env: OPENAI__API_KEY, OPENAI__HOST, OPENAI__MODEL

# Запустить сервер
uv run main.py
```

## Конфигурация

Единственный источник конфига — `.env` в корне проекта (шаблон — `.env.example`).
Он используется дважды: `docker compose` подставляет эти переменные в
`compose.yaml` через `${...}`, а `app/settings/__init__.py` читает тот же файл
напрямую (`SettingsConfigDict(env_file=".env")`) при запуске скриптов на
хосте без Docker — никакого отдельного шага/окружения не нужно, `.env`
подхватывается всегда, если файл существует. Внутри Docker-контейнера `.env`
физически нет (`.dockerignore`) — там только переменные, которые реально
передал сам compose.

Любой параметр можно переопределить настоящей env-переменной — она всегда
приоритетнее значения из `.env`. Вложенные поля разделяются `__`:

```bash
OPENAI__API_KEY=gsk_... \
OPENAI__HOST=https://api.groq.com/openai/v1 \
OPENAI__MODEL=openai/gpt-oss-120b \
uv run main.py
```

### Таблица переменных окружения

| Переменная | По умолчанию | Описание |
|---|---|---|
| `OPENAI__API_KEY` | — (обязательно) | Ключ провайдера. Groq API key (console.groq.com), не ключ OpenAI |
| `OPENAI__HOST` | `https://api.groq.com/openai/v1` | Base URL — Groq отдаёт OpenAI-совместимый API |
| `OPENAI__MODEL` | `openai/gpt-oss-120b` | Продакшен-модель (генерация, агент) |
| `REDIS__URL` | `redis://localhost:6379` | Кеш ответов |
| `REDIS__TTL` | `3600` | TTL кеша, секунд |
| `CHAT__REPOSITORY` | `json` | `json` (файловое) \| `postgres` |
| `CHAT__DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/finpay` | asyncpg-URL для SQLAlchemy (чат) |
| `CHAT__CONTEXT_WINDOW` | `10` | N последних сообщений в контексте |
| `CHAT__RAG_CONDENSE_ENABLED` | `true` | Переписывать follow-up в самостоятельный вопрос перед retrieval |
| `BOT__TOKEN` | — (обязательно для бота) | Токен от @BotFather |
| `BOT__ADMIN_IDS` | `[]` | Telegram user_id с доступом к `/stats`/`/broadcast` и подтверждению HIL-действий |
| `BOT__INTERNAL_TOKEN` | — | Секрет backend → bot (`/notify`) |
| `QDRANT_API_KEY` | — | Секрет Qdrant (контейнер + клиент) |
| `QDRANT__URL` | `http://localhost:6333` | Хостовое значение; в Docker хардкожено `http://qdrant:6333` |
| `QDRANT__COLLECTION` | `documents` | Legacy-коллекция (Б5.2); прод использует `RAG__KB_COLLECTION=finpay_kb` |
| `EMBEDDINGS__DIM` | `768` | Размерность вектора (`intfloat/multilingual-e5-base`) |
| `RAG__RERANKER_ENABLED` | `true` | BAAI/bge-reranker-v2-m3 после retrieval |
| `RAG__SCORE_THRESHOLD` | `0.005` | Порог "не найдено" после re-ranking |
| `AGENT__CHECKPOINTER` | `sqlite` | `memory` (тесты) \| `sqlite` (локально) \| `postgres` (прод) |
| `AGENT__POSTGRES_URI` | `postgresql://postgres:postgres@localhost:5433/finpay` | psycopg (v3) DSN для чекпоинтера — НЕ `CHAT__DATABASE_URL` |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | Разрешённые origin для REST API |
| `SECURITY_ENABLED` | `true` | Prompt-injection защита |
| `ADMIN_TOKEN` | — | Секрет `/chats/admin/*` |
| `MODERATION__ENABLED` | `true` | Keyword-модерация (всегда дёшево включена) |
| `MODERATION__USE_OPENAI_API` | `false` | Второй слой — `omni-moderation-latest` |
| `EVAL__TESTSET_LLM_API_KEY` | — | Настоящий OpenAI-ключ, только для `scripts/generate_testset.py` |

Полный и всегда актуальный список — `.env.example` (значения-заглушки вместо
секретов).

## Режимы запуска

Сервис запускается через единую точку входа `main.py`. Режим передаётся аргументом:

```bash
uv run main.py rest   # HTTP API (по умолчанию, можно без аргумента)
uv run main.py bot    # Telegram-бот
```

## Запуск

### Локально (без Docker)

Требуется запущенный Redis и Qdrant:

```bash
redis-server --daemonize yes
docker compose up -d qdrant   # либо локальный Qdrant

uv run main.py rest
```

### Docker Compose (рекомендуется)

Поднимает FastAPI + Telegram-бот + Redis + PostgreSQL + Qdrant + Phoenix одной
командой (backend использует Postgres по умолчанию, миграции применяются
автоматически при старте `app`):

```bash
cp .env.example .env
# отредактировать .env: OPENAI__API_KEY (обязательно), BOT__TOKEN (если нужен бот)

docker compose up -d --build
```

`pg_data`/`qdrant_storage`/`redis_data` — именованные volumes, данные
переживают `docker compose down` (без флага `-v`).

Остановка:

```bash
docker compose down       # сохранить данные
docker compose down -v    # удалить все volumes
```

### Проверка

```bash
curl http://localhost:8000/health   # liveness  — 200
curl http://localhost:8000/ready    # readiness — 200 если Redis жив, 503 если нет
curl http://localhost:8000/docs     # Swagger UI
```

## Чат-модуль (stateful история)

Модуль `app/chat/` реализует серверную историю диалогов с двумя бэкендами хранилища.

### Переключение хранилища

В `.env`:

```bash
CHAT__REPOSITORY=json      # файловое JSONL-хранилище (по умолчанию)
# CHAT__REPOSITORY=postgres  # PostgreSQL (нужны Docker и миграция)
```

При первом использовании Postgres — применить миграцию:

```bash
uv run alembic upgrade head
```

### Chat API

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/chats` | Создать чат |
| `GET` | `/chats/{id}` | Метаданные чата |
| `POST` | `/chats/{id}/messages` | Отправить сообщение (multipart, опционально с `media`) → SSE-стрим JSON-событий |
| `GET` | `/chats/{id}/messages` | История сообщений |
| `DELETE` | `/chats/{id}/messages` | Мягкое удаление истории |
| `POST` | `/chats/{id}/system-message` | Демо: фоновая задача завершилась (+ опциональный `/notify` в Telegram) |

Мультимодальность (фото/голос/PDF/DOCX) и формат SSE-событий — подробнее в [`docs/chat.md`](docs/chat.md).
RAG внутри чата (retrieval, score-guard, событие `sources`) — [`docs/rag.md`](docs/rag.md).

## Telegram-бот

Тонкий клиент к Chat API. Сценарий: `/ask` → выбор темы → вопрос → стрим в чат
через нативный `sendMessageDraft`. Принимает фото, голос и PDF/DOCX-документы
(конвертация в content-part происходит на backend — бот не импортирует
`openai`/`pypdf`/`python-docx`). Все запросы к backend идут через единый
`BackendClient.send_message(chat_id, content, media=None, mime=None)`.

Бот также поднимает внутренний HTTP-эндпоинт `POST /notify` (порт
`bot.bot_api_port`, защищён заголовком `X-Internal-Token`) — backend
использует его для проактивных уведомлений (см. `app/services/notifier.py`)
и для реальной отправки сообщений агентом после HIL-подтверждения (см. раздел
«Агенты» ниже).

### Запуск бота

```bash
# Прописать в .env:
# BOT__TOKEN=123456:ABC-...
# BOT__BACKEND_URL=http://localhost:8000
# BOT__INTERNAL_TOKEN=локальный-секрет-для-notify

uv run main.py bot
```

REST API должен быть запущен отдельно:

```bash
uv run main.py rest
```

## База знаний и RAG (М5)

`app/services/rag.py::RAGService` — LlamaIndex + Qdrant: retrieval (top-k) →
опциональный re-ranking (`BAAI/bge-reranker-v2-m3`, `app/services/reranker.py`)
→ код-гард по итоговому score ДО вызова LLM → генерация с нумерованными
цитатами `[1]`, `[2]`. Параллельная bare-metal реализация без re-ranking/
цитат/score-guard-до-LLM — исторический baseline Б5.3,
`app/services/rag_baremetal.py` (сравнение — `docs/rag.md`).

**Корпус** — `data/<категория>/` (tariffs, support, security, api, webhooks,
legal, compliance, onboarding, incidents, integrations), 42 документа →
коллекция Qdrant `finpay_kb`. Индексация — `scripts/ingest.py`
(`IngestionPipeline` + `DocstoreStrategy.UPSERTS`, идемпотентно: повторный
запуск не плодит дубликаты, определяет изменённые/неизменные документы по
хешу).

**Чанкинг** — `app/services/chunking.py`, три стратегии (`fixed_size`,
`recursive`, `semantic`); `semantic` победил в grid search на golden dataset
(`docs/chunking_experiment.md`, MRR@10 0.917 vs равные Hit Rate/Recall).

```bash
# Индексация базы знаний (идемпотентно)
uv run scripts/ingest.py

# Прямой вопрос к базе (без агента/чата)
curl -X POST http://localhost:8000/rag/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"Какая стандартная комиссия за транзакцию?"}'
```

Обоснование метрики Qdrant (cosine vs dot), HNSW-параметров и фильтров —
[`docs/vector_store.md`](docs/vector_store.md).

## Агенты (М6)

Прогрессия агентного слоя от простого к прод-готовому — все версии оставлены
в `app/services/` как исторические срезы (не удалялись при переходе к
следующей):

| Модуль | Блок | Что добавляет |
|---|---|---|
| `agent_naive.py` | Б6.1 | Голый `for`-цикл tool-calling, 3 tools |
| `agent_react.py` | Б6.2 | ReAct + self-reflection (Reflexion-light critic), жёсткие лимиты (`max_iterations`, `timeout`) |
| `agent_graph.py` | Б6.3 | Тот же ReAct на LangGraph `StateGraph` (`custom_graph`) + `create_agent` (`prebuilt_graph`) для сравнения |
| `agent_persistent.py` | Б6.4 | + checkpointer (sqlite/postgres) — диалог переживает рестарт; + human-in-the-loop (`interrupt()`+`Command(resume=...)`) на реальной отправке в Telegram; + SSE-стриминг |

**Инструменты**: `search_knowledge_base` (обёртка над `RAGService`),
`get_current_time`, `send_telegram_message` — единственный опасный (реальный
внешний side-effect), поэтому единственный, требующий подтверждения
пользователя перед выполнением.

```bash
# SSE-стрим агента с HIL
curl -N -X POST http://localhost:8000/agent/stream \
  -H 'Content-Type: application/json' \
  -d '{"thread_id":"demo-1","input":{"messages":[{"role":"user","content":"Отправь клиенту в чат 123 сводку по тарифам"}]}}'

# После получения __interrupt__ в потоке — подтверждение тем же thread_id
curl -N -X POST http://localhost:8000/agent/stream \
  -d '{"thread_id":"demo-1","input":{"resume":true}}'
```

Отчёты по каждому шагу с реальными числами (latency/токены/баги, найденные
при отладке) — [`docs/agent-react-report.md`](docs/agent-react-report.md),
[`docs/agent-graph-report.md`](docs/agent-graph-report.md),
[`docs/agent-persistent-report.md`](docs/agent-persistent-report.md). Схемы
графов — `docs/agent-graph-custom.mmd`/`.png`.

Более ранний, отдельный от RAG-агента набор function-calling инструментов
(`app/tools/` — `get_payment_system_status`, `check_transaction_status`) —
демо простого tool-calling без LangGraph из ранних блоков курса, оставлен как
референс, в проде не используется.

### Мультиагентность: решение — не используется (Б6.5)

Сравнение supervisor-графа (researcher+writer, LangGraph) с single-agent
baseline на одинаковых 5 вопросах и одном tool — `experiments/`. Итог:
качество не выросло в среднем (4.4/5 у обоих), стоимость выросла в 8.6× по
токенам и в 29.9× по медианной задержке, а на тривиальном вопросе
координация улетела в 25 LLM-вызовов и ответ на английском вместо русского.
Полное обоснование и решение — [`docs/multi-agent-report.md`](docs/multi-agent-report.md).
Финальный agent-слой — single-agent (`agent_persistent.py`), `experiments/`
остаётся прототипной площадкой, не переехал в прод.

## Оценка качества (eval)

Качество RAG оценивается через RAGAS на golden dataset —
`tests/eval/golden_dataset.json` (36 вопросов, каждый вручную сверен на
дословное соответствие корпусу) и `scripts/run_eval.py`
(Faithfulness/AnswerRelevancy/ContextPrecision/ContextRecall/`has_citation`,
судья — отдельная от продакшена модель через Groq). Методология, 14 найденных
по ходу багов и отклонения от задания разобраны в
[`docs/rag_evaluation.md`](docs/rag_evaluation.md). На момент этого README
прогон не завершён — упёрлись в дневную квоту Groq, итоговые числа по всем 36
строкам и обоим A/B-экспериментам (chunking, reranker on/off) остаются
открытым пунктом, см. «Ограничения» ниже.

```bash
uv run scripts/run_eval.py --label baseline
```

## Наблюдаемость

Все запросы трассируются через OpenTelemetry в Arize Phoenix
(`compose.yaml`, порт `6006` UI / `4317` gRPC-коллектор) —
`app/observability/tracing.py` инструментирует OpenAI- и LlamaIndex-вызовы.
PII маскируется до записи в логи/трейсы (`app/observability/pii.py`).

```bash
open http://localhost:6006
```

## Production-обвязка (Б4.4)

### Модерация (`app/moderation/`)

`ModerationService.check_input` / `check_output` — два слоя:

1. keyword/regex по `app/moderation/moderation_keywords.yaml` (дёшево, включён всегда);
2. опционально OpenAI Moderation API (`omni-moderation-latest`) — включается
   флагом `MODERATION__USE_OPENAI_API=true`.

Вход проверяется до старта SSE-стрима: заблокированный запрос — `403` с
`detail.code == "moderation_blocked"`. Выход проверяется на собранном полном
ответе; если он нарушает правила, в историю пишется заглушка "Не могу показать
ответ — он мог нарушить правила" (токены, уже показанные во время стрима,
задним числом скрыть нельзя — это заранее известное ограничение стриминга).
Инциденты логируются (хеш + маскированный текст, без сырого текста) и пишутся
в таблицу `moderation_incidents` (только Postgres).

### Admin API (`app/admin/`, только Postgres)

Префикс `/chats/admin`, авторизация — заголовок `X-Admin-Token` (сверяется с
`ADMIN_TOKEN` из env).

| Метод | Путь | Описание |
|-------|------|----------|
| `GET` | `/chats/admin/stats` | сообщения/DAU/латентность/block-rate за 24ч |
| `GET` | `/chats/admin/users?limit=50` | последние пользователи |
| `POST` | `/chats/admin/broadcast` | поставить рассылку в очередь |
| `GET` | `/chats/admin/broadcast/pending` | internal: бот забирает рассылки |
| `POST` | `/chats/admin/broadcast/{id}/ack` | internal: бот отчитывается о статусе |

Без Postgres (`CHAT__REPOSITORY=json`) эндпоинты отвечают `503`.

### Admin-команды бота

Доступны только `message.from_user.id` из `BOT__ADMIN_IDS` (фильтр `IsAdmin`
на уровне роутера — не внутри хендлеров):

```
/stats               — статистика за 24ч
/users               — первые 10 последних пользователей
/broadcast <текст>    — поставить рассылку в очередь (interface=telegram)
```

Бот сам вытягивает рассылки из `broadcast_queue` фоновым воркером
(`app/bot/services/broadcast.py`, опрос раз в 10 сек) и шлёт их через свою
сессию — backend не хранит токен бота и не обращается к Telegram напрямую.

### Фидбек 👍/👎

После каждого ответа ассистента бот прикрепляет инлайн-клавиатуру
(`fb:<up|down>:<message_id>`). Голос сохраняется в `message_feedback`
(`UNIQUE (owner_external_id, message_id)` — повторный голос того же
пользователя по тому же сообщению просто игнорируется), после чего бот
убирает клавиатуру через `edit_reply_markup(reply_markup=None)`.

## Security-тестирование (garak)

Тестирование ведётся в два прогона: baseline (без защиты) и after (с защитой).
Garak обращается к серверу через throttle-прокси, чтобы не превысить лимиты провайдера.

### Throttle-прокси

`eval/security/throttle_proxy.py` — тонкий reverse-proxy между garak и сервисом.
Слушает на `:8001`, форвардит на `:8000`.

**Зачем нужен:** garak шлёт ~255 запросов на три пробы. Free-tier Groq имеет
лимит по токенам в минуту, а DAN-промпты весят ~900 токенов каждый — при
неограниченной скорости сразу летят 429.

**Параметры:**

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--rpm` | 20 | Максимум запросов в минуту |
| `--backend` | `http://localhost:8000` | Адрес реального сервиса |

**Поведение при 400:** если security-слой заблокировал запрос до LLM — токены не
потрачены, таймер сбрасывается и следующий запрос идёт без ожидания. After-прогон
поэтому проходит быстрее baseline.

**Подавить шум OTLP** (ускоряет каждый запрос на ~8 сек если коллектор не запущен):
```bash
OTEL_TRACES_EXPORTER=none uv run main.py
```

### Baseline (security отключена)

```bash
# Терминал 1 — сервер без security-слоя
SECURITY_ENABLED=false uv run main.py

# Терминал 2 — throttle-прокси (:8001 → :8000)
uv run python eval/security/throttle_proxy.py --rpm 5

# Терминал 3 — garak
uv run garak \
  --target_type rest \
  -G eval/security/rest_config.json \
  --probes promptinject.HijackHateHumans,encoding.InjectBase64,dan.Ablation_Dan_11_0 \
  --parallel_requests 1 --parallel_attempts 1 \
  --generations 1
```

### After (security включена)

```bash
# Терминал 1 — сервер с security-слоем (по умолчанию)
uv run main.py

# Терминал 2 — throttle-прокси
uv run python eval/security/throttle_proxy.py --rpm 20

# Терминал 3 — те же probe-ы
uv run garak \
  --target_type rest \
  -G eval/security/rest_config.json \
  --probes promptinject.HijackHateHumans,encoding.InjectBase64,dan.Ablation_Dan_11_0 \
  --parallel_requests 1 --parallel_attempts 1 \
  --generations 1
```

### Отчёты garak

Garak сохраняет результаты в `~/.local/share/garak/garak_runs/`.
После каждого прогона скопировать в проект:

```bash
cp ~/.local/share/garak/garak_runs/baseline.* docs/security/reports/baseline/
cp ~/.local/share/garak/garak_runs/after.* docs/security/reports/after/
```

Извлечь attack_success_rate по пробам:

```bash
cat ~/.local/share/garak/garak_runs/baseline.report.jsonl \
  | jq 'select(.entry_type=="eval") | {probe: .probe, asr: (1 - (.passed / .total))}'
```

Шаблоны отчётов: `docs/security/garak_baseline_2026-06-20.md`, `docs/security/garak_after_2026-06-20.md`.

## Тесты

```bash
# Все тесты (без вызовов LLM и без Docker)
uv run pytest tests/ -m "not llm" -v

# Только chat-модуль (Postgres-тесты пропускаются без Docker)
uv run pytest tests/chat/ -v

# Только агенты (checkpointer/HIL — на InMemorySaver/AsyncSqliteSaver(:memory:), без Postgres)
uv run pytest tests/test_agent_persistent.py tests/unit/test_agent_*.py -v

# Только бот
uv run pytest tests/bot/ -v

# Только security-тесты
uv run pytest tests/unit/test_security.py -v

# С реальным LLM (требует API-ключ)
uv run pytest tests/ -m llm -v
```

Postgres-тесты автоматически пропускаются (`s`) если Docker недоступен.

## Ограничения и известные пробелы

Честно, без приукрашивания:

- **RAG-эвал (RAGAS) не доведён до конца** — 12 из 36 строк golden dataset с
  реальными числами, оба A/B-эксперимента (chunking, reranker on/off)
  подготовлены, но не прогнаны до конца — упирались в дневную квоту Groq на
  бесплатном тарифе, не в технические ограничения (пайплайн работает,
  проверено на частичных данных). Детали — `docs/rag_evaluation.md`.
- **Мультиагентность сознательно не используется** — не пробел, а
  обоснованное решение по итогам сравнения (см. «Агенты» выше), но лежит
  готовый прототип на случай, если для развития проекта decomposition
  когда-нибудь окупится.
- **`user_role="read-only"` в агенте** — параметр политики доступа объявлен
  (`app/services/agent_persistent.py`), но реально влияет пока только на
  пропуск HIL для роли `full`; отдельного enforcement-узла, блокирующего ЛЮБОЙ
  tool-вызов для `read-only`, нет — актуально только при добавлении второго
  опасного tool.
- **Coordination overhead агента непредсказуем в худшем случае** — на
  тривиальном запросе supervisor-граф (эксперимент Б6.5) один раз ушёл в 25
  LLM-вызовов вместо ожидаемых 5 — нет бюджетного гарда на число хендоффов.
- **`_send_real_telegram` создаёт новый `aiogram.Bot` на каждый вызов** —
  рабочее, но не самое экономное решение; для высокой частоты HIL-действий
  стоило бы держать один `Bot` в `app.state`.
- **Поведение при недоступности Groq**: RAG/чат — деградация без падения
  (сервис поднимается, retrieval работает, генерация вернёт ошибку на
  конкретный запрос); агентные `/agent/stream`-вызовы — сетевые ошибки от
  `httpx`/`openai` пробрасываются наверх без отдельного retry-слоя поверх
  встроенных ретраев SDK.

## Структура проекта

```
app/
  chat/             # M4Б1/Б4.3/Б4.4: история диалогов, мультимодальность, модерация, RAG в чате (Б5.5)
  moderation/       # Б4.4: keyword-слой + опционально OpenAI Moderation API
  admin/            # Б4.4: /chats/admin/* (только Postgres)
  bot/              # M4Б2/Б4.3/Б4.4: Telegram-бот
  routers/          # FastAPI endpoints: chat, health, models, rag (Б5.3), agent (Б6.4), documents
  services/
    llm.py, notifier.py                 # оркестрация LLM, backend -> bot уведомления
    embeddings.py, vector_store.py      # М5.1/М5.2
    rag.py, rag_baremetal.py            # Б5.3-5.5: RAGService (LlamaIndex+Qdrant+reranker+citations)
    chunking.py, reranker.py, ingestion.py  # Б5.4/Б5.5: стратегии чанкинга, re-ranking, индексация
    retrieval_eval.py                   # вспомогательное для оценки retrieval
    agent_naive.py                      # Б6.1: наивный tool-calling loop
    agent_react.py                      # Б6.2: ReAct + self-reflection
    agent_graph.py                      # Б6.3: тот же ReAct на LangGraph
    agent_persistent.py                 # Б6.4: + checkpointer + HIL + SSE
    security/                           # input_validator, output_filter
  deps/providers.py   # FastAPI Depends: get_llm_service, get_vector_store, get_agent_graph, ...
  tools/              # ранние function-calling демо-инструменты (до RAG-агента)
  prompts/            # Jinja2-шаблоны системного промпта
  schemas/            # Pydantic-модели запросов и ответов
  observability/      # structlog + OpenTelemetry + PII-маскирование
  settings/           # pydantic-settings, .env — по модулю на область (agent.py — Б6.4)

experiments/        # Б6.5: supervisor multi-agent vs single-agent, прототип, не в проде
  common.py, multi_agent_langgraph.py, single_agent_baseline.py, results.json

modes/               # rest.py (uvicorn), bot.py (aiogram polling)

scripts/
  ingest.py, load_to_qdrant.py, qdrant_experiments.py       # М5.2/М5.5: индексация, эксперименты
  chunking_experiment.py, generate_testset.py, run_eval.py  # Б5.4/Б5.6: чанкинг, golden dataset, RAGAS
  run_naive_agent.py, run_agent_comparison.py, bench_agents.py  # Б6.1-6.3: прогоны и бенчмарки агентов
  visualize_graph.py, time_travel_demo.py                   # Б6.3/Б6.4: mermaid-схемы, time travel
  estimate_embedding_cost.py, embeddings_smoke.py           # М5.1

alembic/versions/    # миграции: chat-таблицы, media_refs, moderation/feedback/broadcast
                     # (checkpoint*-таблицы Б6.4 создаёт AsyncPostgresSaver.setup(), не Alembic —
                     #  alembic/env.py::include_name явно исключает их из autogenerate)

eval/security/       # throttle_proxy.py + rest_config.json для garak (единственное, что осталось от раннего eval/)
tests/
  unit/, chat/, admin/, bot/, integration/   # по модулям
  eval/                                       # golden_dataset.json (36 Б5.6), mini_benchmark.json (М5.1)
  test_agent_persistent.py                    # Б6.4: HIL smoke-тесты на AsyncSqliteSaver(:memory:)

docs/
  architecture.md, chat.md, vector_store.md, chunking_experiment.md   # М5
  rag.md, rag_evaluation.md, data_inventory.md                        # Б5.3-5.6
  agent-react-report.md, agent-graph-report.md, agent-persistent-report.md,
  agent-graph-custom.mmd/.png                                         # Б6.2-6.4
  multi-agent-report.md, architecture-multi-agent.md                  # Б6.5
  security/                                                           # garak-отчёты baseline и after
```
