# Payment Support — Function Calling

Ассистент техподдержки платёжного процессинга с двумя инструментами.

## Инструменты

| Tool | Источник данных |
|---|---|
| `get_payment_system_status(component)` | Stripe Status API (публичный, без ключа) |
| `check_transaction_status(transaction_id)` | Локальная SQLite с тестовыми данными |

## Запуск

### Локально

```bash
uv run examples.run_tool_call
```

### Docker

```bash
sudo systemctl start docker
```

Запуск всего стека (FastAPI + Redis) одной командой:

```bash
export OPENAI__API_KEY=gsk_...
docker compose up -d --build
```

`OPENAI__HOST` и `OPENAI__MODEL` имеют дефолты в `compose.yaml`. Переопределить:

```bash
export OPENAI__HOST=https://api.groq.com/openai/v1
export OPENAI__MODEL=llama-3.1-8b-instant
docker compose up -d --build
```

Проверка:

```bash
curl http://localhost:8000/health   # liveness — всегда 200
curl http://localhost:8000/ready    # readiness — 200 если Redis жив, 503 если нет
curl http://localhost:8000/docs     # Swagger UI
```

Остановка:

```bash
docker compose down          # остановить, сохранить данные Redis
docker compose down -v       # остановить и удалить volume Redis
```

## Три тест-кейса

### (а) Запрос, который точно требует tool

**Запрос:** `"Что случилось с транзакцией TXN-1002? Почему платёж не прошёл?"`

Модель вызвала `check_transaction_status` с аргументом `{"transaction_id": "TXN-1002"}`.
Функция обратилась к SQLite и вернула статус `failed`, причину — `"Недостаточно средств"`.
Финальный ответ содержал конкретную причину отказа и рекомендацию проверить баланс карты.

---

### (б) Запрос, который точно не требует tool

**Запрос:** `"Как работает процесс возврата средств при отмене заказа?"`

Модель НЕ вызвала ни один инструмент — ответила текстом напрямую.
Объяснила общий процесс рефанда: инициация, срок зачисления, участие банка.
Код не упал, ветка `if not message.tool_calls` отработала корректно.

---

### (в) Пограничный случай

**Запрос:** `"У меня проблемы с платежами сегодня, всё ли нормально с системой?"`

Модель вызвала `get_payment_system_status` с аргументом `{"component": "payments"}`.
Несмотря на размытую формулировку, модель интерпретировала вопрос как запрос о статусе сервиса.
Получила реальные данные с внутренних данных и сообщила текущее состояние компонента.
