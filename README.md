# FinPay — AI-ассистент техподдержки

FastAPI-сервис на базе LLM для поддержки платёжного процессинга.
Поддерживает function calling, кеширование ответов в Redis, observability через OpenTelemetry и security-слой с защитой от prompt injection.

## Быстрый старт

```bash
# Установить зависимости
uv sync --all-groups

# Скопировать конфиг и вписать API-ключ
cp .config/example.toml .config/local.toml
# отредактировать .config/local.toml: openai.api_key, openai.host, openai.model

# Запустить сервер
ENVIRONMENT=local uv run main.py
```

## Окружения

Конфиги хранятся в `.config/<env>.toml`. Окружение задаётся переменной `ENVIRONMENT`:

| Значение | Файл конфига |
|----------|-------------|
| `local` | `.config/local.toml` |
| `dev` | `.config/dev.toml` |
| `test` | `.config/test.toml` |

Любой параметр из TOML можно переопределить env-переменной. Вложенные поля разделяются `__`:

```bash
ENVIRONMENT=local \
OPENAI__API_KEY=gsk_... \
OPENAI__HOST=https://api.groq.com/openai/v1 \
OPENAI__MODEL=llama-3.3-70b-versatile \
uv run main.py
```

### Ключевые параметры конфига

```toml
[openai]
api_key   = ""                              # ключ провайдера
host      = "https://api.groq.com/openai/v1"
model     = "llama-3.3-70b-versatile"

[redis]
url       = "redis://localhost:6379"
ttl       = 3600                            # секунд, TTL кеша ответов

security_enabled = true                     # false — отключить security-слой
```

## Запуск

### Локально (без Docker)

Требуется запущенный Redis:

```bash
redis-server --daemonize yes

ENVIRONMENT=local uv run main.py
```

### Docker Compose (рекомендуется)

Поднимает FastAPI + Redis одной командой:

```bash
export OPENAI__API_KEY=gsk_...
export OPENAI__HOST=https://api.groq.com/openai/v1
export OPENAI__MODEL=llama-3.3-70b-versatile

docker compose up -d --build
```

Остановка:

```bash
docker compose down       # сохранить данные Redis
docker compose down -v    # удалить volume Redis
```

### Проверка

```bash
curl http://localhost:8000/health   # liveness  — 200
curl http://localhost:8000/ready    # readiness — 200 если Redis жив, 503 если нет
curl http://localhost:8000/docs     # Swagger UI
```

## Инструменты (function calling)

| Tool | Описание |
|------|----------|
| `get_payment_system_status(component)` | Статус компонентов платёжной системы |
| `check_transaction_status(transaction_id)` | Статус транзакции из локальной SQLite |

## Тесты

```bash
# Все unit-тесты (без вызовов LLM)
ENVIRONMENT=test uv run pytest tests/ -m "not llm" -v

# Только security-тесты
ENVIRONMENT=test uv run pytest tests/unit/test_security.py -v

# С реальным LLM (требует API-ключ)
ENVIRONMENT=local uv run pytest tests/ -m llm -v
```

## Оценка качества (eval)

### Прогон на golden dataset

```bash
ENVIRONMENT=local uv run python eval/run_evaluation.py \
  --golden eval/golden_dataset.json \
  --judge  gpt-4o-mini \
  --out    eval/runs/$(date +%Y-%m-%d).json
```

Результат сохраняется в `eval/runs/<YYYY-MM-DD>.json`.

### Проверка порогов (CI)

```bash
# Читает последний файл из eval/runs/, завершается с exit 1 при нарушении
ENVIRONMENT=test uv run python eval/check_thresholds.py
```

Пороги задаются в `eval/thresholds.yaml`:

```yaml
correctness_avg: 4.0   # средняя оценка правильности по всем вопросам
min_correctness: 2.0   # минимально допустимая оценка для одного вопроса
```

## Security-тестирование (garak)

Тестирование ведётся в два прогона: baseline (без защиты) и after (с защитой).
Garak обращается к серверу через throttle-прокси, чтобы не превысить лимиты провайдера.

### Throttle-прокси

`eval/security/throttle_proxy.py` — тонкий reverse-proxy между garak и сервисом.
Слушает на `:8001`, форвардит на `:8000`.

**Зачем нужен:** garak шлёт ~255 запросов на три пробы. Free-tier Groq/xAI имеет лимит
~6 000 TPM, а DAN-промпты весят ~900 токенов каждый — при неограниченной скорости
сразу летят 429.

**Как считать RPM под свой лимит:**
```
RPM = TPM_limit / avg_tokens_per_request
# Groq free tier, llama-3.1-8b-instant: 6000 / 900 ≈ 5 RPM
```

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
OTEL_TRACES_EXPORTER=none ENVIRONMENT=local uv run main.py
```

### Baseline (security отключена)

```bash
# Терминал 1 — сервер без security-слоя
ENVIRONMENT=local SECURITY_ENABLED=false uv run main.py

# Терминал 2 — throttle-прокси (:8001 → :8000), для Groq free tier --rpm 5
ENVIRONMENT=local uv run python eval/security/throttle_proxy.py --rpm 5

# Терминал 3 — garak
ENVIRONMENT=local uv run garak \
  --target_type rest \
  -G eval/security/rest_config.json \
  --probes promptinject.HijackHateHumans,encoding.InjectBase64,dan.Ablation_Dan_11_0 \
  --parallel_requests 1 --parallel_attempts 1 \
  --generations 1
```

### After (security включена)

```bash
# Терминал 1 — сервер с security-слоем (по умолчанию)
ENVIRONMENT=local uv run main.py

# Терминал 2 — throttle-прокси
ENVIRONMENT=local uv run python eval/security/throttle_proxy.py --rpm 20

# Терминал 3 — те же probe-ы
ENVIRONMENT=local uv run garak \
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
# После baseline
cp ~/.local/share/garak/garak_runs/baseline.* \
   docs/security/reports/baseline/

# После after
cp ~/.local/share/garak/garak_runs/after.* \
   docs/security/reports/after/
```

Извлечь attack_success_rate по пробам:

```bash
cat ~/.local/share/garak/garak_runs/baseline.report.jsonl \
  | jq 'select(.entry_type=="eval") | {probe: .probe, asr: (1 - (.passed / .total))}'
```

Шаблоны отчётов: `docs/security/garak_baseline_2026-06-20.md`, `docs/security/garak_after_2026-06-20.md`.

## Структура проекта

```
app/
  routers/          # FastAPI endpoints (/chat, /health, /models)
  services/
    llm.py          # оркестрация LLM + tool calls
    security/       # input_validator, output_filter
  llm/client.py     # синхронный OpenAI-клиент (module-level)
  prompts/          # Jinja2-шаблоны системного промпта
  tools/            # handlers и схемы для function calling
  schemas/          # Pydantic-модели запросов и ответов
  observability/    # structlog + OpenTelemetry + PII-маскирование
  settings/         # pydantic-settings, TOML + env

eval/
  golden_dataset.json     # 26 вопросов, 4 категории
  run_evaluation.py       # G-Eval judge CLI
  check_thresholds.py     # CI-проверка порогов
  thresholds.yaml
  security/
    rest_config.json      # конфиг garak REST target (:8001)
    throttle_proxy.py     # rate-limiting прокси

tests/
  unit/             # 56 unit-тестов, без сетевых вызовов
  conftest.py

docs/
  architecture.md
  security/         # garak-отчёты baseline и after
```
