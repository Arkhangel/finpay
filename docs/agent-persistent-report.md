# Блок 6.4 — LangGraph: персистентность, HIL, стриминг, time travel

`app/services/agent_persistent.py` берёт `custom_graph` из `agent_graph.py`
(Б6.3, не тронут — остаётся in-memory вариантом для unit-тестов) и добавляет
checkpointer, human-in-the-loop для одного опасного tool
(`send_telegram_message` — теперь настоящая отправка через aiogram, не
print), SSE-стриминг через FastAPI и time travel по чек-пойнтам.

## 1. Backend checkpointer’а: где какой

`AGENT__CHECKPOINTER` (env, `app/settings/agent.py`):

| Режим | Когда | Saver |
|-------|-------|-------|
| `memory` | unit-тесты, где даже sqlite-файл не нужен | `InMemorySaver` |
| `sqlite` | локальная разработка без Docker (дефолт) | `AsyncSqliteSaver`, файл `storage/agent_checkpoints.sqlite` (тот же volume, что `docstore_kb.json` из Б5.5 — переживает рестарт контейнера) |
| `postgres` | прод, `docker-compose` | `AsyncPostgresSaver`, тот же Postgres-сервис, что чат-модуль (М3Б5) |

Отклонение от буквального имени `AGENT_CHECKPOINTER` из ТЗ: используется
`AGENT__CHECKPOINTER` (nested-delimiter `__`) — ради консистентности с тем,
как организована ВСЯ остальная конфигурация проекта
(`settings.<область>.<поле>`, см. `app/settings/__init__.py`). Осознанное
отклонение, не влияет на суть критерия (переключение backend через env
всё равно есть).

`await checkpointer.setup()` вызывается ровно один раз — внутри
`agent_lifespan()`, который держится открытым на весь lifespan FastAPI
(`app/lifespan.py`), не на каждый запрос. Проверено вживую по логу старта:

```
Persistent agent graph ready: checkpointer=sqlite
```

появляется один раз при `Application startup complete`, не повторяется ни
на одном последующем запросе.

## 2. Postgres в docker-compose

`compose.yaml`, сервис `app` — добавлены `AGENT__CHECKPOINTER=postgres` и
`AGENT__POSTGRES_URI=postgresql://postgres:postgres@postgres:5432/finpay`
(psycopg v3 DSN — отдельный от `CHAT__DATABASE_URL`, который asyncpg/
SQLAlchemy). Новый Postgres-сервис не создавался — тот же `postgres` из
М3Б5, та же база `finpay`. Также добавлен `BOT__TOKEN` в env сервиса `app`
(раньше был только у сервиса `bot` — нужен `app`, потому что теперь
`/agent/stream` сам шлёт реальные сообщения в Telegram).

Проверено вживую: `AsyncPostgresSaver(...).setup()` против
`postgresql://postgres:postgres@localhost:5433/finpay` (хостовый порт
compose-сервиса), затем:

```
$ docker compose exec -T postgres psql -U postgres -d finpay -c '\dt'
                 List of relations
 Schema |         Name          | Type  |  Owner
--------+-----------------------+-------+----------
 public | alembic_version       | table | postgres
 public | broadcast_queue       | table | postgres
 public | chat_messages         | table | postgres
 public | chats                 | table | postgres
 public | checkpoint_blobs      | table | postgres
 public | checkpoint_migrations | table | postgres
 public | checkpoint_writes     | table | postgres
 public | checkpoints           | table | postgres
 public | message_feedback      | table | postgres
 public | moderation_incidents  | table | postgres
(10 rows)
```

Ровно 4 чекпоинт-таблицы рядом с доменными, в одной БД — критерий "новая БД
не плодится" выполнен буквально (checklist упоминает `-d agent_db`, но
задача 2 явно требует не создавать отдельную БД — используем `finpay`,
буквальное имя из checklist — генерик-заготовка шаблона задания, не
специфика этого проекта).

Отдельно замечено: **SQLite-чекпоинтер использует ДРУГУЮ схему** —
`checkpoints`/`writes` (2 таблицы, не 4) — имена `checkpoint_writes`/
`checkpoint_blobs`/`checkpoint_migrations` специфичны именно для
`AsyncPostgresSaver`. `include_name` в Alembic (раздел 8) написан под
Postgres-схему сознательно — с sqlite Alembic вообще не работает
(`chat.database_url` — Postgres-only путь).

## 3. Опасный tool: что до interrupt, что после

`send_telegram_message` — потому что это единственный из 3 tools с реальным
внешним побочным эффектом на СТОРОННИЙ сервис (в отличие от
`get_current_time` — чистая функция, и `search_knowledge_base` — read-only).

Разнесено на два узла с обязательным edge между ними:

- **`prepare_send_telegram`** (до interrupt, idempotent) — читает
  `tool_call` последнего сообщения модели, кладёт `{tool_call_id, args}` в
  `state["pending_action"]`. Никакого сетевого вызова, никакого
  побочного эффекта — безопасно выполнить сколько угодно раз (при resume
  граф проигрывает узлы заново от последнего checkpoint).
- **`confirm_and_execute_send_telegram`** (после) — `decision =
  interrupt({"preview": payload, "type": "approve_send_telegram_message"})`;
  ТОЛЬКО после этой строки (то есть только после реального `resume`) —
  настоящий вызов `aiogram.Bot.send_message`.

Если бы `bot.send_message(...)` стоял ДО `interrupt()` — при каждом resume
(включая повторные попытки после сбоя сети на этапе доставки resume) узел
перезапустился бы с начала и отправил сообщение ещё раз. До самого
`interrupt()` в узле нет ничего, кроме чтения уже готового `payload` из
state — пока resume не доставлен (или доставка ретраится), повторный запуск
узла безопасен: `interrupt()` просто снова паузит граф.

**Честная поправка (global-аудит проекта, после написания этого раздела):**
"структурно невозможно" было переоценкой — есть более узкий, но реальный
сценарий гонки уже ПОСЛЕ успешного resume: `decision = interrupt(...)`
возвращает `True`, но клиентское SSE-соединение обрывается ровно во время
исполнения `await bot.send_message(...)` (сам вызов ещё не вернулся, узел
ещё не успел закоммитить `pending_action=None` в checkpoint). Поскольку
`asyncio.CancelledError` — не `Exception`, а `BaseException`, голый `except
Exception` его не ловит, и узел прерывается ДО фиксации результата. Со
стороны checkpointer'а граф выглядит так, будто он всё ещё стоит на паузе —
повторный `Command(resume=true)` на том же `thread_id` заново запускает узел
с нуля и реально отправляет сообщение в Telegram второй раз.

Смягчено (не устранено формально): `_send_real_telegram(...)` теперь
оборачивается в `asyncio.shield(...)` с явным перехватом
`asyncio.CancelledError` в `confirm_and_execute_send_telegram` — реальный
HTTP-вызов к Telegram всегда либо успевает завершиться и узел фиксирует
результат, либо не начинается вовсе. Строгой гарантии "ровно один раз" это
не даёт (у Telegram Bot API нет idempotency-key на отправку сообщений), но
закрывает конкретно этот воспроизводимый сценарий обрыва клиента.

## 4. Живой прогон: interrupt → resume (curl + Telegram)

`POST /agent/stream`, задача "Отправь клиенту в чат 351696260 сообщение:
привет из curl-демо блока 6.4, попытка два" (реальный chat_id, реальная
отправка — подтверждено получением сообщения в Telegram):

Момент паузы (стрим останавливается, `__interrupt__` виден и в потоке
`updates`, и в отдельном явном событии):

```
data: {"type": "updates", "payload": {"prepare_send_telegram": {"pending_action": {"tool_call_id": "fc_f194274a-...", "args": {"chat_id": "351696260", "text": "привет из curl-демо блока 6.4, попытка два"}}}}}

data: {"type": "updates", "payload": {"__interrupt__": [{"value": {"preview": {...}, "type": "approve_send_telegram_message"}, "id": "dcbd20da..."}]}}

data: {"type": "__interrupt__", "payload": [{"preview": {...}, "type": "approve_send_telegram_message"}]}
```

Момент после `curl -d '{"thread_id":"curl-hil-2","input":{"resume":true}}'`
(тот же `/agent/stream`, тот же thread_id):

```
data: {"type": "updates", "payload": {"confirm_and_execute_send_telegram": {"messages": [...], "tool_results": [{"name": "send_telegram_message", "args": {...}, "result": "Сообщение отправлено в 351696260"}], "pending_action": null}}}

data: {"type": "messages", "payload": {"content": "Со", "node": "call_model"}}
... (токены генерации финального ответа)
data: {"type": "updates", "payload": {"call_model": {"messages": [{"content": "Сообщение успешно отправлено клиенту.", ...}], "iteration_count": 2}}}
data: {"type": "updates", "payload": {"force_finish": null}}
```

## 5. Time travel

`scripts/time_travel_demo.py` (офлайн: модель и Telegram-отправка замоканы
— реальная отправка уже подтверждена в разделе 4, здесь цель — детерминизм
демонстрации, не повторный расход Groq-квоты). Реальный вывод:

```
=== (2) История чек-пойнтов thread A (сразу после interrupt) ===
  checkpoint_id=1f191a93…cd0a45 next=('confirm_and_execute_send_telegram',) outcome='?'
  checkpoint_id=1f191a93…9954eb next=('prepare_send_telegram',) outcome='?'
  checkpoint_id=1f191a93…702c34 next=('call_model',) outcome='?'
  checkpoint_id=1f191a93…f3fdd8 next=('__start__',) outcome='?'

=== (3) Time travel: читаем СТАРЫЙ checkpoint_id ПОСЛЕ того, как тред уже продолжился ===
  next: ('confirm_and_execute_send_telegram',) (узел подтверждения — ещё ничего не отправлено)
  pending_action: {'tool_call_id': 'call_1', 'args': {...}}
  tool_results на этом чек-пойнте: []

=== (4) Две ветки из ОДИНАКОВОГО входа — на РАЗНЫХ thread_id ===
  demo-B (resume=True):  Сообщение отправлено в 555
  demo-C (resume=False): Отклонено пользователем — сообщение не отправлено.

  Bot.send_message вызван всего 2 раза за весь прогон (thread A + demo-B, оба resume=True) — demo-C (resume=False) вклада не внёс
```

Ключевой момент раздела (3): thread A **уже был резюмирован**
(`Command(resume=True)`, сообщение реально "отправлено") ДО того, как мы
читаем старый `checkpoint_id` — и он всё равно показывает
`tool_results: []` и `next=('confirm_and_execute_send_telegram',)`, то есть
состояние ДО отправки. Это и есть путешествие во времени: чек-пойнт
неизменен, хотя сам тред давно ушёл вперёд.

Две ветки (approve/deny) из идентичного входа получены на **разных**
`thread_id` (demo-B, demo-C), не повторным resume одного треда — потому что
значение `resume` фиксируется в чекпоинтере как pending-write на весь
thread-lineage: второй `Command(resume=...)` с другим значением на том же
interrupt-чекпоинте вернул бы ПЕРВОЕ зафиксированное значение, а не новое
(проверено на практике при отладке — см. раздел 8).

## 6. Streaming mode

Выбран `astream(stream_mode=["updates", "messages"])`, не
`astream_events(version="v2")`. Причина: `updates` даёт ровно то, что нужно
для UI прогресса (какой узел отработал, что изменилось в state — включая
естественную видимость `__interrupt__` прямо в потоке, без отдельного
парсинга event-tree), `messages` — токены генерации для стриминга текста.
`astream_events(v2)` даёт более гранулярные события
(`on_chat_model_stream`, `on_tool_start` и т.п.), но это лишняя
детализация и лишний объём для задачи "показать прогресс + токены +
момент паузы" — два явных stream_mode читаются и форматируются в SSE проще
и прозрачнее, чем разбор универсального event-tree.

## 7. Permission policy

`config["configurable"]["user_role"]` — `read-only` / `write-with-approve`
(дефолт) / `full`. `confirm_and_execute_send_telegram` для `full` пропускает
`interrupt()` целиком (сразу `decision=True`) — остальные роли всегда ждут
подтверждения. `read-only` в текущей версии не даёт дополнительных гарантий
на уровне графа (нет отдельного узла, который бы блокировал ЛЮБОЙ tool для
этой роли) — это осознанно оставлено как TODO, см. раздел 8.

## 8. Баги, найденные при отладке

1. **`Interrupt` не JSON-сериализуется — ронял SSE-стрим молча ровно в
   момент паузы.** LangGraph сам эмитит `updates`-событие
   `{"__interrupt__": (Interrupt(...),)}` при паузе — `Interrupt` не dict и
   не pydantic-модель, `json.dumps` падал `TypeError` внутри async-генератора
   `StreamingResponse`, соединение просто обрывалось без валидного
   `__interrupt__`-события на клиенте (curl показывал последний успешный
   `updates` и тишину). Нашёл через прямой curl-прогон (не через юнит-тест
   — там модель и Bot замоканы синтетически, ошибка не воспроизводилась,
   пока не попробовал живой конец-в-конец). Исправлено: `_json_safe`
   разворачивает `Interrupt` в `{"value": ..., "id": ...}` явно.
2. **Роутер SSE-эндпоинта падал `KeyError('iteration_count')` на первом
   запросе нового треда.** Curl-пример из самого ТЗ (`{"input":
   {"messages":[...]}}`) не содержит остальных полей `AgentState`
   (`iteration_count`/`tool_results`/`pending_action`) — вызывающий не
   обязан знать о внутренней схеме state. Исправлено: эндпоинт дефолтит
   отсутствующие поля для нового треда (и заодно автоматически добавляет
   `SYSTEM_PROMPT`, если в `messages` нет системного сообщения) — резюме
   (`{"input": {"resume": ...}}`) идёт отдельной веткой, без этих полей.
3. **`Command(resume=...)` не переигрывает решение — фиксируется на
   thread-lineage.** Пытался сначала получить "две ветки" повторным resume
   ОДНОГО thread_id с разными значениями (`True`, потом `False`) — второй
   вызов возвращал результат первого (`True`), логика ветвления в
   `time_travel_demo.py` даже до его нынешнего вида молча "не работала" (обе
   ветки показывали один и тот же исход). Только после этого стало ясно,
   что нужны два разных thread_id — задокументировано и в разделе 5, и в
   докстринге `agent_persistent.py`.

## 9. Что осталось хрупким / чего не хватает

- `user_role="read-only"` не имеет собственного enforcement-узла — сейчас
  влияет только на `confirm_and_execute_send_telegram`'s пропуск interrupt
  (`full`), а не на общий запрет ЛЮБОГО tool-вызова для read-only. Для
  диплома этого достаточно (единственный опасный tool — telegram), но при
  добавлении второго опасного tool policy стоит вынести в отдельный узел
  перед роутером.
- `_send_real_telegram` создаёт новый `aiogram.Bot(token=...)` на каждый
  вызов подтверждённого действия — рабочий, но не самый экономный вариант;
  для высокой частоты вызовов стоило бы держать один `Bot` в
  `app.state` (как остальные клиенты в `app/lifespan.py`), а не создавать
  заново каждый раз.
- SSE-эндпоинт не валидирует `chat_id`/содержимое `text` перед постановкой
  на подтверждение — модель теоретически может подготовить payload с
  некорректным `chat_id` (не число), и это всплывёт только в момент
  реальной отправки (после resume), а не на этапе `prepare_send_telegram`.
- Постоянного расписания очистки старых checkpoint-записей нет — таблица
  `checkpoints` растёт неограниченно на активных тредах; для прод-нагрузки
  нужен TTL/retention (LangGraph сам такого механизма из коробки не даёт).
- Тесты (`tests/test_agent_persistent.py`) гоняются только на
  `AsyncSqliteSaver(":memory:")` — Postgres-путь проверен вручную (раздел
  2), но не покрыт автоматическим тестом (testcontainers для Postgres уже
  есть в проекте для чата — можно переиспользовать по тому же паттерну).
