# Руководство по конфигурации системы

## Анализ параметров запуска

Пример команды с ограниченной оптимизацией:
```bash
python main_pipeline.py all --force --llm-clean --optimize --optimize-mode test --optimize-sample 5
```

Ограничения данной конфигурации:
- `--optimize-mode test`: минимальный перебор (5 комбинаций параметров)
- `--optimize-sample 5`: малая выборка для валидации
- Отсутствие `--min-usefulness`: использование порога по умолчанию (0.3)

---

## Рекомендуемые конфигурации

### Полная оптимизация

```bash
python main_pipeline.py all \
  --force \
  --llm-clean \
  --min-usefulness 0.5 \
  --optimize \
  --optimize-mode full \
  --optimize-sample 100
```

Параметры:
- `--optimize-mode full`: полный перебор гиперпараметров (1225 комбинаций)
- `--optimize-sample 100`: репрезентативная выборка для оптимизации
- `--min-usefulness 0.5`: повышенный порог фильтрации документов

---

### Быстрая оптимизация

```bash
python main_pipeline.py all \
  --force \
  --llm-clean \
  --min-usefulness 0.5 \
  --optimize \
  --optimize-mode quick \
  --optimize-sample 50
```

Параметры:
- `--optimize-mode quick`: средний перебор (54 комбинации параметров)
- `--optimize-sample 50`: сбалансированная выборка для оптимизации
- `--min-usefulness 0.5`: повышенный порог фильтрации

---

### Использование API режима

```bash
# Настройка переменных окружения
export LLM_MODE=api
export LLM_API_MODEL=tngtech/deepseek-r1t2-chimera:free
export OPENROUTER_API_KEY=sk-or-v1-...
export LLM_API_MAX_WORKERS=10

# Запуск с API режимом
python main_pipeline.py all \
  --force \
  --llm-clean \
  --min-usefulness 0.5 \
  --optimize \
  --optimize-mode quick \
  --optimize-sample 50
```

Преимущества API режима:
- Ускорение обработки за счет параллельных запросов
- Поддержка бесплатных моделей
- Снижение требований к локальным ресурсам

---

## Сравнение режимов оптимизации

| Режим   | Комбинаций | Размер выборки |
|---------|-----------|----------------|
| `test`  | 5         | 5              |
| `quick` | 54        | 50             |
| `full`  | 1225      | 100            |

---

## Дополнительные параметры настройки

### Выбор LLM модели

```bash
# Бесплатные модели:
export LLM_API_MODEL=openrouter/sherlock-think-alpha
export LLM_API_MODEL=tngtech/deepseek-r1t2-chimera:free

# Коммерческие модели:
export LLM_API_MODEL=openai/gpt-4o-mini
export LLM_API_MODEL=anthropic/claude-3-haiku
```

### Настройка фильтрации

```bash
# Стандартная фильтрация
--min-usefulness 0.5

# Повышенная фильтрация
--min-usefulness 0.6

# Строгая фильтрация
--min-usefulness 0.7
```

Примечание: Значения выше 0.7 могут привести к излишнему отсечению релевантных документов.

### Проверка конфигурации функций

Параметры в `src/config.py`:
- `ENABLE_QUERY_EXPANSION = True`
- `ENABLE_RRF = True`
- `ENABLE_CONTEXT_WINDOW = True`
- `ENABLE_METADATA_FILTER = True`
- `ENABLE_USEFULNESS_FILTER = True`
- `ENABLE_DYNAMIC_TOP_K = True`
- `RERANKER_TYPE = "cross_encoder"` или `"llm"`

### Конфигурация переранжирования

```bash
export RERANKER_TYPE=llm  # альтернатива cross_encoder
```

LLM переранжирование обеспечивает более точную оценку релевантности при увеличении времени обработки.

---

## Полная конфигурация

```bash
# Настройка окружения
export LLM_MODE=api
export LLM_API_MODEL=openrouter/sherlock-think-alpha
export OPENROUTER_API_KEY=sk-or-v1-...
export LLM_API_MAX_WORKERS=10
export RERANKER_TYPE=llm

# Проверка конфигурации
python main_pipeline.py check-env

# Запуск с полной оптимизацией
python main_pipeline.py all \
  --force \
  --llm-clean \
  --min-usefulness 0.5 \
  --optimize \
  --optimize-mode full \
  --optimize-sample 100
```

---

## Альтернативные конфигурации

**Базовая конфигурация:**
```bash
python main_pipeline.py all --force
```

**С LLM-предобработкой:**
```bash
python main_pipeline.py all --force --llm-clean
```

**С быстрой оптимизацией:**
```bash
python main_pipeline.py all --force --llm-clean --optimize --optimize-mode quick --optimize-sample 50
```

**Полная оптимизация:**
```bash
python main_pipeline.py all --force --llm-clean --min-usefulness 0.5 --optimize --optimize-mode full --optimize-sample 100
```

---

## Сбалансированная конфигурация

```bash
export LLM_MODE=api
export LLM_API_MODEL=tngtech/deepseek-r1t2-chimera:free
export OPENROUTER_API_KEY=sk-or-v1-...

python main_pipeline.py all \
  --force \
  --llm-clean \
  --min-usefulness 0.5 \
  --optimize \
  --optimize-mode quick \
  --optimize-sample 50
```

---

## Рекомендации по параметрам

1. `--optimize-mode test`: для быстрого тестирования функциональности
2. `--optimize-sample`: минимум 30-50 для репрезентативной оптимизации
3. `--min-usefulness 0.5`: сбалансированное соотношение качества и покрытия
4. API режим: рекомендуется для ускорения обработки
5. `--optimize-mode full`: для наиболее тщательной оптимизации параметров

