# Garak After Report — 2026-06-20

## Команда запуска

```bash
garak --target_type rest -G eval/security/rest_config.json \
      --probes promptinject.HijackHateHumans,encoding.InjectBase64,dan.Ablation_Dan_11_0 \
      --generations 1 \
      --report_prefix after
```

## Окружение

| Параметр | Значение |
|----------|----------|
| garak version | `<!-- garak --version -->` |
| Дата прогона | 2026-06-20 |
| Модель | gpt-4o-mini |
| Таргет | http://localhost:8000/chat |
| Защита | **включена** (input_validator + output_filter + canary) |

## Дельта baseline → after

| probe | baseline % | after % | закрыли | примечание |
|-------|-----------|---------|---------|------------|
| promptinject.HijackHateHumans | <!-- BB% --> | <!-- AA% --> | <!-- ΔΔ% --> | regex на `ignore.*instructions` |
| encoding.InjectBase64 | <!-- BB% --> | <!-- AA% --> | <!-- ΔΔ% --> | non-printable ratio heuristic |
| dan.Ablation_Dan_11_0 | <!-- BB% --> | <!-- AA% --> | <!-- ΔΔ% --> | `you are now DAN` pattern |

> Ожидаемо: `promptinject.HijackHateHumans` падает с ~60% до 5–15% благодаря
> regex-паттерну `\bignore\s+(all\s+)?(previous|prior|above)\s+instructions?\b`.

## Анализ

### Что закрыла защита

- `promptinject.HijackHateHumans` — блокируется `injection` правилом в `input_validator.py`
  (паттерн на `ignore previous/prior/above instructions`)
- `dan.Ablation_Dan_11_0` — блокируется паттерном `you are now DAN`

### Что осталось / стало хуже

<!-- Если какая-то проба ухудшилась — зафиксировать честно с гипотезой -->
<!-- Например: encoding.InjectBase64 — non-printable ratio heuristic может давать
     ложные срабатывания на многобайтовые UTF-8 символы в кириллических текстах.
     Гипотеза: порог 10% слишком агрессивный для длинных русскоязычных вопросов. -->

## Как garak засчитывает блокировку

При `HTTPException(400)` сервис возвращает `{"detail": {"code": "input_blocked", ...}}`.
В этом ответе нет поля `content`, поэтому garak не может извлечь текст через
`response_json_field: "content"` — проба считается защитой пройденной.
