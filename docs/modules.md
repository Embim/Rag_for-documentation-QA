# Обзор модулей проекта

Проект: Альфа RAG (Weaviate-only, streaming processing)

## Структура каталогов
- `main_pipeline.py` — CLI-оркестратор: build, search, all, evaluate.
- `src/config.py` — конфигурация путей, параметров индексации/поиска, логирования.
- `src/logger.py` — единая настройка логирования (файл `outputs/pipeline.log`).
- `src/preprocessing.py` — предобработка документов/вопросов.
- `src/streaming_builder.py` — потоковая обработка документов: load → clean (LLM) → chunk → batch index (Weaviate).
- `src/chunking.py` — разбиение на чанки (слова, parent-child при необходимости).
- `src/indexing.py` — WeaviateIndexer (векторный + BM25 в Weaviate) и BM25Indexer (локальный для совместимости). FAISS удалён.
- `src/retrieval.py` — гибридный ретривер (Weaviate hybrid) + rerankers (LLM/Transformer/Cross-Encoder).
- `src/grid_search_optimizer.py` — grid search оптимизация гиперпараметров.
- `scripts/run_grid_search.py` — запуск оптимизации параметров retriever.
- `scripts/test_weaviate.py` — вспомогательные проверки Weaviate.

## Ключевые компоненты

### main_pipeline.py
- `build_knowledge_base(force_rebuild, llm_clean, min_usefulness)` — строит базу: потоковая обработка и индексация в Weaviate.
- `process_questions(...)` — online обработка вопросов через `RAGPipeline`.
- CLI команды: `build`, `search`, `all`, `evaluate`.

### streaming_builder.py
- `StreamingDocumentProcessor` — потоковая обработка CSV чанками:
  - preprocess → опциональная LLM-clean → chunk
  - накопление чанков в батчи и индексация в Weaviate (сразу)
- `build_knowledge_base_streaming(...)` — удобный фасад.

### indexing.py
- `WeaviateIndexer` — создание эмбеддингов (SentenceTransformers), загрузка документов батчами в коллекцию Weaviate, гибридный/векторный поиск, `delete_all()`, `close()`.
- `BM25Indexer` — локальный BM25 (используется только при необходимости оффлайн, Weaviate использует встроенный BM25).

### retrieval.py
- `HybridRetriever` — Weaviate-only гибридный поиск (vector + BM25) с `alpha` балансом, query expansion, фильтры (usefulness, metadata).
- `LLMReranker` / `TransformerReranker` — переранжирование кандидатов.
- `RAGPipeline` — объединяет retriever + reranker + выбор документов.

### config.py
- Пути/директории, параметры Weaviate, embedding-модели, поиск/alpha, reranker, параметры логирования (`LOG_LEVEL`, `LOG_FILE`), автосоздание директорий.

### logger.py
- `setup_logging(level, log_file, enable_console)` — настраивает ротацию логов в `outputs/pipeline.log` и вывод в консоль без дублирования хендлеров.


