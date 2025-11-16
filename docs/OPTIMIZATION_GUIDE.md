# Руководство по оптимизации RAG системы

## 🎯 Инструменты для повышения accuracy

### 1. Grid Search гиперпараметров

Автоматический поиск оптимальных параметров:

```bash
# Быстрый поиск (Weaviate-only)
python scripts/run_grid_search.py --mode quick --sample 50

# Полный поиск
python scripts/run_grid_search.py --mode full --sample 100

# Отключить LLM-оценку (быстрее)
python scripts/run_grid_search.py --mode quick --sample 50 --no-llm
```

**Оптимизируемые параметры:**
- `TOP_K_DENSE`: количество результатов векторного поиска
- `TOP_K_BM25`: количество результатов BM25
- `TOP_K_RERANK`: количество результатов после reranking
- `HYBRID_ALPHA`: баланс между dense (alpha) и BM25 (1-alpha)

**Результат:**
```
outputs/grid_search_quick_YYYYMMDD_HHMMSS.csv
```

Содержит все комбинации параметров и их метрики (avg_score, accuracy).

**Применение лучшей конфигурации:**

После grid search обнови `src/config.py`:
```python
TOP_K_DENSE = 30  # было 25
TOP_K_BM25 = 30   # было 25
TOP_K_RERANK = 25 # было 20
HYBRID_ALPHA = 0.6  # было 0.5
```

---

### 2. Query Expansion (расширение запроса)

Улучшает recall за счет синонимов и альтернативных формулировок.

**Включение:**

В `src/config.py`:
```python
ENABLE_QUERY_EXPANSION = True
QUERY_EXPANSION_METHOD = "synonyms"  # или "llm", "hybrid"
```

Или через переменную окружения:
```bash
export ENABLE_QUERY_EXPANSION=true
python main_pipeline.py search
```

**Методы:**

| Метод | Описание | Скорость | Качество |
|-------|----------|----------|----------|
| `synonyms` | Только словарь синонимов | ⚡ Быстро | Среднее |
| `llm` | LLM генерация вариантов | 🐢 Медленно | Высокое |
| `hybrid` | Оба метода | 🐌 Очень медленно | Максимум |

**Рекомендация:** Используй `synonyms` - быстро и эффективно для банковских терминов.

---

### 3. Комбинированная стратегия

Для максимального результата:

1. **Предочистка данных (LLM-clean, опционально)**
   ```bash
   # включи --llm-clean при build
   python main_pipeline.py build --force --llm-clean --min-usefulness 0.5
   ```

2. **Эмбеддинги** (BGE-M3, рекомендуется)
   ```python
   # В config.py:
   EMBEDDING_MODEL = "BAAI/bge-m3"
   ```

3. **Query Expansion** (synonyms)
   ```python
   ENABLE_QUERY_EXPANSION = True
   QUERY_EXPANSION_METHOD = "synonyms"
   ```

4. **Grid Search** оптимизация
   ```bash
   python scripts/run_grid_search.py --mode quick --sample 100
   # Применить лучшие параметры в config.py
   ```

5. **Сильный LLM reranker** (Qwen3-32B 8-bit)
   ```python
   LLM_MODEL_FILE = "Qwen3-32B-2507-Q8_0.gguf"
   ```

---

## 🔧 Быстрая диагностика

### Если accuracy < 0.5:

1. **Проверь данные (очистка):**
   - Включи `--llm-clean` и проверь порог `--min-usefulness`.

2. **Проверь embeddings:**
   ```python
   # В config.py:
   EMBEDDING_MODEL = "BAAI/bge-m3"
   ```

3. **Пересоздай индексацию (Weaviate):**
```bash
python main_pipeline.py build --force
```

### Если accuracy 0.5-0.7:

1. **Настрой гиперпараметры:**
   ```bash
   python scripts/run_grid_search.py --mode quick --sample 100
   ```

2. **Включи Query Expansion:**
   ```python
   ENABLE_QUERY_EXPANSION = True
   ```

3. **Используй сильный reranker:**
   ```python
   LLM_MODEL_FILE = "Qwen3-32B-2507-Q8_0.gguf"
   ```

### Если accuracy > 0.7 но нужно больше:

1. **Увеличь TOP_K:**
   ```python
   TOP_K_DENSE = 40
   TOP_K_BM25 = 40
   TOP_K_RERANK = 30
   ```

2. **Используй LLM для query expansion:**
   ```python
   QUERY_EXPANSION_METHOD = "hybrid"
   ```

---

## 🚀 Рекомендуемый workflow

### День 1: Baseline + Предочистка
```bash
# 1. Build (опционально с LLM-clean)
python main_pipeline.py build --force --llm-clean --min-usefulness 0.5

# 2. Inference
python main_pipeline.py search
```

### День 2: Query Expansion + Grid Search
```bash
# 1. Включить Query Expansion
export ENABLE_QUERY_EXPANSION=true

# 2. Grid Search
python scripts/run_grid_search.py --mode quick --sample 100

# 3. Применить лучшие параметры в config.py

# 4. Перезапустить inference
python main_pipeline.py search
```

### День 3: Fine-tuning (опционально)
```bash
# 1. Fine-tune embeddings (если необходимо)
python scripts/finetune_embeddings.py

# 2. Пересоздать индексы
python main_pipeline.py build --force

# 3. Финальный inference
python main_pipeline.py search
```

---

## 💡 Советы:

1. **Grid Search сначала на маленькой выборке** (50 вопросов), затем на полной
2. **Query Expansion = только synonyms** для скорости, llm если нужно максимум
3. **Следи за VRAM** через `nvidia-smi` (эмбеддинги/LLM-reranker)
4. **Логи** в `outputs/pipeline.log` (настраивается `LOG_FILE`, `LOG_LEVEL`)


