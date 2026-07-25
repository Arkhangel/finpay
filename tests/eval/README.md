# tests/eval — golden dataset (блок 5.6)

## Как получить `golden_dataset.json`

1. Сгенерировать сырые Q/A через RAGAS `TestsetGenerator`:

   ```
   python scripts/generate_testset.py --provider groq --max-files 4 --size 5 --file-offset 0
   ```

   `--provider groq` — бесплатно, но медленно (Groq free tier TPM) — гонять
   маленькими батчами со сдвигом `--file-offset`, каждый следующий вызов
   дописывается в `--out` (дедуп по `user_input`), а не перезаписывает его.
   `--provider openai` — настоящий OpenAI, быстрее и надёжнее, но платно и
   требует `EVAL__TESTSET_LLM_API_KEY` (см. `app/settings/eval.py`).

2. Результат — `tests/eval/golden_dataset_raw.csv`, колонки
   `user_input`/`reference`/`reference_contexts`.

3. **Обязательная ручная вычитка** сырого CSV перед тем, как он станет
   golden dataset:
   - убрать дубли и вопросы-перефразы одного и того же факта;
   - убрать слишком общие/бессмысленные вопросы (RAGAS иногда генерирует
     вопросы вида "О чём этот документ?");
   - проверить, что `reference` фактически соответствует корпусу
     (`data/`, без `data/rag-block-03/` — отдельный исторический корпус
     Б5.3/Б5.4, не пересекается с продакшен-коллекцией `finpay_kb`).

4. Сохранить вычитанный результат как `tests/eval/golden_dataset.json` —
   список объектов `{"user_input": str, "reference": str, "reference_contexts": [str, ...], "source": "ragas"|"manual"}`
   (путь настраивается через `settings.eval.golden_dataset_path`).

**Текущее состояние (2026-07-25): датасет готов, 36 строк.** RAGAS-батчи
уперлись в дневной (TPD) лимит токенов Groq на `openai/gpt-oss-120b`
(200k/день, см. `feedback_ragas_testset_rate_limits` в памяти) и параллельно
обнажили баг RAGAS: `PersonaGenerationPrompt` по умолчанию `language=english`
— автогенерируемые персоны, а с ними и вопросы, оказались в основном на
английском для полностью русскоязычного корпуса (15 из 23 строк). Решение:
переключились на `openai/gpt-oss-20b` (свежий дневной лимит) с вручную
заданными русскоязычными `persona_list` (см. `scripts/generate_testset.py`),
но и это упёрлось в бюджет по времени раньше, чем набралось 30+ строк.
Оставшиеся строки дописаны вручную (`source: "manual"`) — вопрос+ответ+
`reference_contexts` собраны из реальных файлов `data/` (offset 16+, не
затронутых RAGAS-батчами), каждый `reference_contexts` — дословная выдержка
из документа, а не пересказ. Итог: 4 строки `source: "ragas"` (прошли
проверку на русский язык), 32 строки `source: "manual"`, включая один
намеренный edge-case вопрос вне базы (проверка score-guard-отказа).

## Как прогнать метрики

```
python scripts/run_eval.py
```

Для каждой строки золотого датасета: `RAGService.evaluate_inputs()` (реальный
retrieval + генерация системы) → RAGAS-метрики (`app/eval/metrics.py`, judge —
`settings.eval.judge_model`, намеренно другая модель, чем продакшен) +
`has_citation` (не-LLM проверка формата). Результат — `{out-name}_rows.csv` и
`{out-name}_summary.json` в `settings.eval.results_dir`.

## Прочие файлы в этой папке

- `retrieval_dataset.json` — golden dataset для retrieval-метрик (блок 5.4,
  `app/services/retrieval_eval.py`), отдельный от RAGAS golden dataset выше.
- `mini_benchmark.json` — ручной мини-бенчмарк релевантных/нерелевантных
  фрагментов, использовался до RAGAS-пайплайна.
- `golden_dataset_raw_test3.csv` — 3 строки из тестового прогона
  `--provider openai` (см. app/settings/eval.py), безопасно смёржить в
  финальный `golden_dataset_raw.csv` вместо перегенерации.
