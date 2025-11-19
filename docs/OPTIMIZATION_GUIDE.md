# Руководство по оптимизации системы поиска

## Методы оптимизации гиперпараметров

### Grid Search

Систематический перебор комбинаций параметров для определения оптимальных значений:

```bash
# Режим быстрого поиска
python scripts/run_grid_search.py --mode quick --sample 50

# Режим полного перебора
python scripts/run_grid_search.py --mode full --sample 100

# Без LLM-оценки (сокращение времени выполнения)
python scripts/run_grid_search.py --mode quick --sample 50 --no-llm
```

Оптимизируемые параметры:
- `TOP_K_DENSE`: размер выборки векторного поиска
- `TOP_K_BM25`: размер выборки BM25
- `TOP_K_RERANK`: размер выборки для переранжирования
- `HYBRID_ALPHA`: весовой коэффициент гибридного поиска

Результаты сохраняются в:
```
outputs/grid_search_quick_YYYYMMDD_HHMMSS.csv
```

Файл содержит все тестируемые комбинации и соответствующие метрики.

Применение оптимальных значений в `src/config.py`:
```python
TOP_K_DENSE = 30
TOP_K_BM25 = 30
TOP_K_RERANK = 25
HYBRID_ALPHA = 0.6
```

---

### Query Expansion

Расширение запроса синонимами и альтернативными формулировками для повышения полноты поиска.

Конфигурация в `src/config.py`:
```python
ENABLE_QUERY_EXPANSION = True
QUERY_EXPANSION_METHOD = "synonyms"  # варианты: "llm", "hybrid"
```

Или через переменные окружения:
```bash
export ENABLE_QUERY_EXPANSION=true
python main_pipeline.py search
```

Доступные методы:

**synonyms**
- Использование предопределенного словаря синонимов
- Высокая скорость выполнения
- Подходит для доменно-специфичной терминологии

**llm**
- Генерация вариантов запроса с использованием LLM
- Увеличенное время выполнения
- Расширенное покрытие формулировок

**hybrid**
- Комбинация обоих методов
- Максимальное покрытие
- Увеличенное время обработки

---

### Комплексная оптимизация

Последовательность применения методов оптимизации:

**1. Предобработка документов**
   ```bash
   python main_pipeline.py build --force --llm-clean --min-usefulness 0.5
   ```

**2. Конфигурация embedding модели**
   ```python
   # В config.py:
   EMBEDDING_MODEL = "BAAI/bge-m3"
   ```

**3. Активация Query Expansion**
   ```python
   ENABLE_QUERY_EXPANSION = True
   QUERY_EXPANSION_METHOD = "synonyms"
   ```

**4. Оптимизация гиперпараметров**
   ```bash
   python scripts/run_grid_search.py --mode quick --sample 100
   # Применение оптимальных значений в config.py
   ```

**5. Конфигурация переранжирования**
   ```python
   LLM_MODEL_FILE = "Qwen3-32B-2507-Q8_0.gguf"
   ```

---

## Процесс настройки

### Базовая конфигурация

1. Проверка качества данных:
   ```bash
   python main_pipeline.py build --force --llm-clean --min-usefulness 0.5
   ```

2. Конфигурация embeddings:
   ```python
   EMBEDDING_MODEL = "BAAI/bge-m3"
   ```

3. Переиндексация:
   ```bash
   python main_pipeline.py build --force
   ```

### Оптимизация параметров

1. Настройка гиперпараметров:
   ```bash
   python scripts/run_grid_search.py --mode quick --sample 100
   ```

2. Активация Query Expansion:
   ```python
   ENABLE_QUERY_EXPANSION = True
   ```

3. Настройка переранжирования:
   ```python
   LLM_MODEL_FILE = "Qwen3-32B-2507-Q8_0.gguf"
   ```

### Расширенная оптимизация

1. Увеличение размеров выборок:
   ```python
   TOP_K_DENSE = 40
   TOP_K_BM25 = 40
   TOP_K_RERANK = 30
   ```

2. Использование гибридного расширения запросов:
   ```python
   QUERY_EXPANSION_METHOD = "hybrid"
   ```

---

## Рекомендации

**Последовательность шагов:**

1. Построение базовой конфигурации с LLM-предобработкой
   ```bash
   python main_pipeline.py build --force --llm-clean --min-usefulness 0.5
   python main_pipeline.py search
   ```

2. Активация Query Expansion и оптимизация
   ```bash
   export ENABLE_QUERY_EXPANSION=true
   python scripts/run_grid_search.py --mode quick --sample 100
   # Применение оптимальных параметров в config.py
   python main_pipeline.py search
   ```

3. Дополнительная настройка (опционально)
   ```bash
   python scripts/finetune_embeddings.py
   python main_pipeline.py build --force
   python main_pipeline.py search
   ```

**Практические рекомендации:**

- Grid Search следует начинать с малых выборок для оценки времени выполнения
- Query Expansion режим "synonyms" обеспечивает оптимальный баланс скорости и качества
- Мониторинг использования VRAM через `nvidia-smi`
- Анализ логов в `outputs/pipeline.log` (конфигурируется через `LOG_FILE`, `LOG_LEVEL`)


