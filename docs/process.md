# Процесс работы пайплайна

Модель работы: Weaviate-only, потоковая обработка (streaming).

## High-level
1) Input: `websites.csv` (title, url, kind, text, web_id, ...).
2) Streaming builder читает CSV маленькими порциями (`csv_chunksize`).
3) Для каждого документа:
   - preprocess (очистка, нормализация);
   - опционально LLM-clean (фильтрация по полезности, добавление метаданных);
   - чанкинг текста (по словам).
4) Накопление чанков в батч `chunk_batch_size` → индексация в Weaviate (с генерацией эмбеддингов на лету).
5) Итог: сохраняется `outputs/pipeline.log` и `data/processed/chunks.pkl` (метаданные чанков).

## Детали потоковой обработки
- CSV читается блоками (по N документов) — не держим весь датафрейм в памяти.
- LLM-clean (если включён) отбрасывает нерелевантные документы и строит полезные метаданные (`entities`, `topics`).
- Чанки формируются c web_id, title, url, kind, word/char counts, chunk_index.
- Для Weaviate батчи отправляются сразу — эмбеддинги считаются локально (SentenceTransformers) и сохраняются вместе с объектами.

## Поиск (online)
1) Загружаются метаданные чанков `chunks.pkl`.
2) Подключение к Weaviate.
3) `HybridRetriever` делает гибридный поиск: vector + BM25 (вызов `collection.query.hybrid`).
4) Опционально: Query Expansion, фильтры, Reranker (LLM/Transformer/Cross-Encoder).
5) Возвращается список web_id топ-N документов.

## Логирование
- Инициализируется в `main()`: `setup_logging(level=LOG_LEVEL, log_file=LOG_FILE)`.
- Файл логов по умолчанию: `outputs/pipeline.log` (ротация, UTF-8).


