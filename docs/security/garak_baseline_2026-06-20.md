# Garak Baseline Report — 2026-06-20

## Команда запуска

```bash
garak --target_type rest -G eval/security/rest_config.json \
      --probes promptinject.HijackHateHumans,encoding.InjectBase64,dan.Ablation_Dan_11_0 \
      --generations 1 \
      --report_prefix baseline
```

## Окружение

| Параметр | Значение |
|----------|----------|
| garak version | `<!-- garak --version -->` |
| Дата прогона | 2026-06-20 |
| Модель | gpt-4o-mini |
| Таргет | http://localhost:8000/chat |
| Защита | **отключена** (baseline) |

## Результаты

| probe | запусков | пробитых | attack_success_rate |
|-------|----------|----------|---------------------|
| promptinject.HijackHateHumans | <!-- N --> | <!-- n --> | <!-- XX% --> |
| encoding.InjectBase64 | <!-- N --> | <!-- n --> | <!-- XX% --> |
| dan.Ablation_Dan_11_0 | <!-- N --> | <!-- n --> | <!-- XX% --> |
| **Итого** | | | |

> HTML-отчёт garak: `docs/security/reports/baseline/`

## Примеры пробитий

### 1. promptinject.HijackHateHumans

**Input:**
```
<!-- вставить input из JSONL: entry_type=="attempt", status==2 -->
```

**Ответ модели:**
```
<!-- вставить output из JSONL -->
```

---

### 2. encoding.InjectBase64

**Input:**
```
<!-- вставить input -->
```

**Ответ модели:**
```
<!-- вставить output -->
```

---

### 3. dan.Ablation_Dan_11_0

**Input:**
```
<!-- вставить input -->
```

**Ответ модели:**
```
<!-- вставить output -->
```

## Как заполнить этот отчёт

```bash
# Извлечь пробития из JSONL
cat ~/.local/share/garak/garak_runs/baseline.report.jsonl \
  | jq 'select(.entry_type=="attempt" and .status==2)' \
  | head -20

# Получить attack_success_rate по пробам
cat ~/.local/share/garak/garak_runs/baseline.report.jsonl \
  | jq 'select(.entry_type=="eval")' \
  | jq '{probe: .probe, passed: .passed, total: .total}'
```
