# RAG Пайплайн для поиска релевантных документов

Система информационного поиска на основе гибридного RAG подхода (Retrieval-Augmented Generation) для работы с корпусом документов банковской тематики.

## Описание

Проект представляет собой полнофункциональный пайплайн для семантического поиска релевантных документов по текстовым запросам. Система использует комбинацию современных методов обработки естественного языка, включая векторный поиск, лексический поиск и переранжирование на основе нейросетевых моделей.

## Основные возможности

### Методы поиска и обработки

**Индексация документов:**
- Векторное представление на базе BAAI/bge-m3 (multilingual embeddings)
- BM25 индекс для лексического поиска
- Опциональная предобработка документов с использованием LLM

**Поиск и ранжирование:**
- Гибридный поиск (Dense + Sparse)
- Reciprocal Rank Fusion для объединения результатов
- Cross-Encoder переранжирование
- Context Window расширение результатов
- Query Expansion для улучшения запросов
- Multi-hop Reasoning для сложных запросов

**Оптимизация:**
- Grid Search для подбора гиперпараметров
- Адаптивный выбор количества результатов
- Фильтрация по метаданным и оценке полезности
- Параллельная обработка запросов

---

## Архитектура системы

Система построена по двухэтапной архитектуре: offline-этап индексации документов и online-этап обработки запросов.

### Этап 1: Построение индекса (Offline)

**1.1. Загрузка и предобработка документов**
- Чтение корпуса документов из CSV формата
- Очистка текста: удаление HTML-тегов, нормализация пробелов
- Лексическая нормализация: приведение к нижнему регистру
- Применение словаря синонимов

**1.2. LLM-предобработка (опциональная)**
- Удаление нерелевантных элементов (навигация, футеры, метаинформация)
- Извлечение структурированных метаданных (продукты, действия, условия, темы)
- Оценка информативности документа
- Фильтрация по порогу полезности

**1.3. Сегментация (Chunking)**
- Разбиение документов на фрагменты фиксированного размера (200 токенов)
- Использование скользящего окна с перекрытием (50 токенов)
- Сохранение контекста между сегментами

**1.4. Индексация**
- Построение векторного индекса с использованием BAAI/bge-m3 embeddings
- Построение BM25 индекса для лексического поиска
- Хранение в Weaviate векторной базе данных

### Этап 2: Обработка запроса (Online)

**2.1. Предобработка запроса**
- Query Reformulation: переформулирование запроса с использованием LLM (опционально)
- Query Expansion: расширение запроса синонимами и связанными терминами

**2.2. Гибридный поиск**
- Dense retrieval: семантический поиск по векторным представлениям
- Sparse retrieval: BM25 поиск по ключевым словам
- Параллельное выполнение обоих методов

**2.3. Объединение результатов**
- Применение Reciprocal Rank Fusion (RRF) для комбинирования результатов
- Нормализация скоров из разных источников
- Формирование единого ранжированного списка кандидатов

**2.4. Переранжирование**
- Cross-Encoder оценка релевантности пар (запрос, документ)
- Использование модели cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
- Отбор топ-N наиболее релевантных результатов

**2.5. Расширение контекста**
- Context Window: добавление соседних сегментов для полноты информации
- Multi-hop Reasoning: итеративный поиск для многоступенчатых запросов
- Применение метаданных для фильтрации результатов

**2.6. Формирование ответа**
- Отбор финального набора документов (топ-5)
- Возврат идентификаторов релевантных документов

### Оптимизация гиперпараметров

Grid Search автоматически подбирает оптимальные значения:
- `TOP_K_DENSE`: количество результатов векторного поиска
- `TOP_K_BM25`: количество результатов BM25
- `TOP_K_RERANK`: количество документов для переранжирования
- `HYBRID_ALPHA`: вес векторного поиска в гибридной схеме

Поддерживаются режимы: test (5 комбинаций), quick (54 комбинации), full (1225 комбинаций)

---

## Структура проекта

```
АльфаRAg/
├── src/                           # Исходный код
│   ├── config.py                  # Конфигурация параметров
│   ├── preprocessing.py           # Предобработка текстов
│   ├── chunking.py                # Сегментация документов
│   ├── indexing.py                # Построение индексов
│   ├── retrieval.py               # Гибридный поиск и RAG pipeline
│   ├── llm_preprocessing.py       # LLM очистка документов
│   ├── grid_search_optimizer.py   # Grid search оптимизация
│   ├── cross_encoder_reranker.py  # Cross-Encoder переранжирование
│   ├── query_expansion.py         # Расширение запросов
│   ├── query_reformulation.py     # Переформулирование запросов
│   ├── reciprocal_rank_fusion.py  # Reciprocal Rank Fusion
│   ├── context_window.py          # Context window расширение
│   ├── multi_hop_reasoning.py     # Multi-hop reasoning
│   ├── streaming_builder.py       # Потоковая обработка документов
│   ├── document_cleaner.py        # Очистка документов
│   ├── openrouter_cleaner.py      # API интеграция для очистки
│   ├── llm_judge.py               # LLM оценка релевантности
│   ├── llm_evaluator.py           # Оценка качества поиска
│   └── logger.py                  # Логирование
├── scripts/                       # Вспомогательные скрипты
├── data/
│   └── processed/                 # Обработанные данные и индексы
├── models/                        # Модели и веса
├── outputs/                       # Результаты поиска
├── docs/                          # Документация
├── main_pipeline.py               # Главный исполняемый модуль
├── docker-compose.yml             # Docker конфигурация для Weaviate
├── requirements.txt               # Python зависимости
├── .env.example                   # Шаблон переменных окружения
├── websites.csv                   # Корпус документов
├── questions_clean.csv            # Набор запросов
└── README.md                      # Документация проекта
```

---

## Установка

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Настройка переменных окружения

Создайте файл `.env` в корне проекта:

```bash
# Скопируйте шаблон
cp .env.example .env

# Или создайте вручную с минимальными настройками:
cat > .env << 'EOF'
# LLM настройки (ОБЯЗАТЕЛЬНО!)
LLM_MODE=api
OPENROUTER_API_KEY=your-api-key-here

# Модель (бесплатная)
LLM_API_MODEL=tngtech/deepseek-r1t2-chimera:free

# Weaviate
USE_WEAVIATE=true
WEAVIATE_URL=http://localhost:8080

# Обработка
QUESTION_PROCESSING_WORKERS=20
LLM_API_MAX_WORKERS=10
EOF
```

**Получение API ключа:**
1. Перейдите на https://openrouter.ai/keys
2. Зарегистрируйтесь (бесплатно)
3. Создайте API ключ
4. Вставьте в `.env`: `OPENROUTER_API_KEY=sk-or-v1-...`

### 3. Запуск Weaviate

```bash
# Запуск через Docker
docker-compose up -d

# Проверка статуса
docker-compose ps
```

### 4. Проверка конфигурации

```bash
# Проверьте все настройки
python main_pipeline.py check-env
```

### 5. Опционально: Локальная LLM модель

Если хотите использовать локальную модель вместо API:

```bash
# 1. Скачайте модель в папку models/
# Рекомендуется: Qwen3-32B-IQ4_NL.gguf (~20 GB VRAM)

# 2. Установите llama-cpp-python с GPU поддержкой
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# 3. Настройте .env
export LLM_MODE=local
```

---

## Использование

### Базовый сценарий работы

```bash
# 1. Настройка переменных окружения
export LLM_MODE=api
export OPENROUTER_API_KEY=sk-or-v1-your-key

# 2. Запуск Weaviate
docker-compose up -d

# 3. Проверка конфигурации
python main_pipeline.py check-env

# 4. Построение индекса
python main_pipeline.py build

# 5. Обработка запросов
python main_pipeline.py search

# Результат: outputs/submission.csv
```

### Расширенные сценарии

**Построение индекса с LLM-предобработкой:**
```bash
python main_pipeline.py build --llm-clean --min-usefulness 0.5
```

Параметры:
- `--llm-clean`: активация LLM-предобработки документов
- `--min-usefulness`: порог фильтрации по полезности (0.0-1.0, по умолчанию 0.3)
- `--force`: принудительная переиндексация

**Поиск с оптимизацией гиперпараметров:**
```bash
python main_pipeline.py search --optimize --optimize-mode quick
```

Параметры:
- `--optimize`: активация Grid Search оптимизации
- `--optimize-mode`: режим оптимизации (test/quick/full)
- `--optimize-sample N`: размер выборки для оптимизации
- `--limit N`: ограничение количества обрабатываемых запросов

**Полный пайплайн:**
```bash
python main_pipeline.py all --llm-clean --optimize
```

Выполняет последовательно:
1. Построение индекса с опциональной LLM-предобработкой
2. Оптимизацию гиперпараметров (опционально)
3. Обработку всех запросов

---

## Справочник команд

### Команда: build

Построение индекса документов.

```bash
python main_pipeline.py build [опции]
```

Опции:
- `--force` - принудительное пересоздание индексов
- `--llm-clean` - активация LLM-предобработки документов
- `--min-usefulness FLOAT` - порог фильтрации по полезности (0.0-1.0)

### Команда: search

Обработка запросов.

```bash
python main_pipeline.py search [опции]
```

Опции:
- `--limit N` - обработка первых N запросов
- `--optimize` - активация Grid Search оптимизации
- `--optimize-mode MODE` - режим оптимизации (test/quick/full)
- `--optimize-sample N` - размер выборки для оптимизации

### Команда: all

Выполнение полного цикла (build + search).

```bash
python main_pipeline.py all [опции]
```

Поддерживает все опции команд build и search.

### Команда: check-env

Проверка конфигурации и переменных окружения.

```bash
python main_pipeline.py check-env
```

### Команда: evaluate

Оценка качества на эталонных примерах.

```bash
python main_pipeline.py evaluate
```

---

## Конфигурация

### Переменные окружения (.env файл)

Основные настройки задаются через файл `.env`:

```bash
# === LLM НАСТРОЙКИ (ОБЯЗАТЕЛЬНО!) ===
LLM_MODE=api                                          # "api" или "local"
OPENROUTER_API_KEY=sk-or-v1-...                      # API ключ (обязателен!)
LLM_API_MODEL=tngtech/deepseek-r1t2-chimera:free    # модель OpenRouter
LLM_API_MAX_WORKERS=10                               # параллельные API запросы
LLM_API_TIMEOUT=60                                   # таймаут запроса (сек)
LLM_API_MAX_TOKENS=32768                            # макс токенов

# === WEAVIATE (ВЕКТОРНАЯ БД) ===
USE_WEAVIATE=true                                    # использовать Weaviate
WEAVIATE_URL=http://localhost:8080                  # адрес Weaviate

# === ОБРАБОТКА ДАННЫХ ===
CSV_CHUNKSIZE=10                                     # размер chunk для CSV
LLM_PARALLEL_WORKERS=1                              # параллельные LLM воркеры
QUESTION_PROCESSING_WORKERS=20                       # параллельная обработка вопросов

# === RAG УЛУЧШЕНИЯ ===
ENABLE_RRF=true                                      # Reciprocal Rank Fusion
ENABLE_CONTEXT_WINDOW=true                           # Context Window
ENABLE_QUERY_EXPANSION=false                         # Query Expansion
ENABLE_METADATA_FILTER=true                          # фильтрация по метаданным
ENABLE_USEFULNESS_FILTER=true                        # фильтрация по полезности
ENABLE_DYNAMIC_TOP_K=true                           # адаптивный TOP_K

# === RERANKER ===
RERANKER_TYPE=cross_encoder                          # "cross_encoder", "llm", "none"

# === GRID SEARCH ===
GRID_SEARCH_MODE=test                                # "test", "quick", "full"
GRID_SEARCH_USE_LLM=true                            # использовать LLM для оценки

# === ЛОГИРОВАНИЕ ===
LOG_LEVEL=INFO                                       # DEBUG, INFO, WARNING, ERROR
LOG_FILE=pipeline.log                                # файл логов
```

### Параметры в src/config.py

Дополнительные настройки в `src/config.py`:

```python
# Чанкинг
CHUNK_SIZE = 200           # слов в чанке
CHUNK_OVERLAP = 50         # перекрытие

# Embedding модель
EMBEDDING_MODEL = "BAAI/bge-m3"  # BGE-M3 (лучшая для русского)

# Поиск (оптимизируются через --optimize)
TOP_K_DENSE = 20           # топ-K векторного поиска
TOP_K_BM25 = 20            # топ-K BM25
TOP_K_RERANK = 10          # топ-K после reranking
HYBRID_ALPHA = 0.5         # вес Dense (1-alpha для BM25)

# Cross-Encoder reranker
CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# LLM модель (для локального режима)
LLM_MODEL_FILE = "Qwen3-32B-IQ4_NL.gguf"  # 32B IQ4_NL (~20 GB VRAM)
```

### Проверка конфигурации

```bash
# Проверка всех настроек и переменных окружения
python main_pipeline.py check-env
```

---

## Результаты

Выходной файл `outputs/submission.csv`:

```csv
q_id,web_list
1,[45, 78, 12, 90, 34]
2,[23, 56, 89, 12, 45]
...
```

- `q_id` - ID вопроса
- `web_list` - список из 5 web_id (топ-5 документов)

---

## Метрика оценки

Система оценивается по метрике Recall@5:

```
Recall@5 = (количество релевантных документов в топ-5) / (общее количество релевантных документов)
```

## Технические характеристики

### Векторная база данных

Система использует Weaviate - векторную базу данных с поддержкой гибридного поиска.

Особенности:
- Встроенная поддержка Dense + BM25 поиска
- Персистентное хранение данных
- HTTP API для интеграции
- Масштабируемость для больших корпусов

Запуск:
```bash
docker-compose up -d
```

### Требования к ресурсам

Минимальная конфигурация:
- RAM: 16 GB
- GPU: 8 GB VRAM (опционально, для ускорения индексации)
- Диск: 10 GB свободного пространства

Рекомендуемая конфигурация:
- RAM: 32 GB+
- GPU: 16 GB+ VRAM (NVIDIA A100, RTX 4090 или аналог)
- Диск: 50 GB+ SSD

---

## Дополнительная документация

### Руководства
- [QUICKSTART.md](QUICKSTART.md) - Быстрый старт
- [CHEATSHEET.md](CHEATSHEET.md) - Справочник команд
- [FINAL_IMPROVEMENTS_SUMMARY.md](FINAL_IMPROVEMENTS_SUMMARY.md) - Сводка улучшений
- [NEW_IMPROVEMENTS_GUIDE.md](NEW_IMPROVEMENTS_GUIDE.md) - Подробное руководство

### Настройка
- [WEAVIATE_SETUP.md](WEAVIATE_SETUP.md) - Настройка Weaviate
- [QUICKSTART_WEAVIATE.md](QUICKSTART_WEAVIATE.md) - Интеграция с Weaviate
- [README_SERVER.md](README_SERVER.md) - Развертывание на сервере

### Разработка
- [docs/FUTURE_IMPROVEMENTS.md](docs/FUTURE_IMPROVEMENTS.md) - Планы развития
- [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) - Примеры использования

---

## Troubleshooting

### API ключ не работает:
```bash
# 1. Проверьте что ключ установлен
python main_pipeline.py check-env

# 2. Проверьте формат ключа
echo $OPENROUTER_API_KEY  # должен начинаться с sk-or-v1-

# 3. Получите новый ключ на https://openrouter.ai/keys

# 4. Установите в .env
echo "OPENROUTER_API_KEY=sk-or-v1-your-new-key" >> .env
```

### Переменные окружения не загружаются:
```bash
# Python не загружает .env автоматически!
# Используйте один из вариантов:

# Вариант 1: экспортируйте вручную
export $(cat .env | xargs)

# Вариант 2: используйте python-dotenv
pip install python-dotenv
# Добавьте в начало main_pipeline.py:
# from dotenv import load_dotenv
# load_dotenv()

# Вариант 3: установите напрямую
export LLM_MODE=api
export OPENROUTER_API_KEY=sk-or-v1-...
```

### Weaviate не запускается:
```bash
# Проверьте Docker
docker-compose ps

# Проверьте логи
docker-compose logs weaviate

# Перезапустите
docker-compose down
docker-compose up -d

# Проверьте подключение
curl http://localhost:8080/v1/.well-known/ready
```

### LLM модель не найдена (локальный режим):
```bash
# Скачайте модель в папку models/
# Рекомендуется: Qwen3-32B-IQ4_NL.gguf

# Или используйте API режим
export LLM_MODE=api
```

### CUDA out of memory:
```bash
# Уменьшите batch size в .env
export EMBEDDING_BATCH_SIZE=64

# Или в src/config.py:
# EMBEDDING_BATCH_SIZE = 64
```

### Grid search слишком долгий:
```bash
# Используйте test режим (5 комбинаций)
python main_pipeline.py search --optimize --optimize-mode test

# Или quick режим (54 комбинации)
python main_pipeline.py search --optimize --optimize-mode quick

# Уменьшите выборку
python main_pipeline.py search --optimize --optimize-sample 10
```

### Медленная обработка вопросов:
```bash
# Увеличьте количество воркеров в .env
export QUESTION_PROCESSING_WORKERS=30  # для API режима
export LLM_API_MAX_WORKERS=20

# Проверьте что используете API режим (быстрее)
export LLM_MODE=api
```

---

## История версий

### Версия 2.0
- Интеграция LLM-предобработки документов
- Grid Search оптимизация гиперпараметров
- Реализация 7 техник улучшения RAG:
  - Reciprocal Rank Fusion
  - Context Window расширение
  - Query Reformulation
  - Query Expansion
  - Cross-Encoder переранжирование
  - Multi-hop Reasoning
  - Метаданные и фильтрация
- Обновление документации

### Версия 1.0
- Гибридный поиск (Dense + BM25)
- Cross-Encoder переранжирование
- Поддержка Weaviate векторной базы данных
- Базовая предобработка текстов

---

## Лицензия и авторство

Проект разработан для задачи информационного поиска в корпусе банковских документов.
