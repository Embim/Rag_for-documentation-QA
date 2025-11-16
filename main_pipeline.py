"""
Главный скрипт RAG пайплайна

Использование:
    python main_pipeline.py build           # Построить базу знаний
    python main_pipeline.py search          # Обработать вопросы
    python main_pipeline.py all             # Полный цикл (build + search)
    python main_pipeline.py evaluate        # Оценка на примерах
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.config import (
    WEBSITES_CSV,
    QUESTIONS_CSV,
    MODELS_DIR,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    USE_WEAVIATE,
    ENABLE_AGENT_RAG,
    LOG_LEVEL,
    LOG_FILE
)
from src.preprocessing import load_and_preprocess_documents, load_and_preprocess_questions
from src.chunking import create_chunks_from_documents
from src.indexing import build_indexes, EmbeddingIndexer, BM25Indexer, WeaviateIndexer
from src.retrieval import RAGPipeline
from src.llm_preprocessing import apply_llm_cleaning
from src.grid_search_optimizer import optimize_rag_params
from src.logger import setup_logging, get_logger, log_timing
import logging
import time

# Проверка доступности Weaviate
try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    if USE_WEAVIATE:
        # Логгер будет инициализирован в main()
        pass


def build_knowledge_base(force_rebuild: bool = False, llm_clean: bool = False,
                        min_usefulness: float = 0.3):
    """
    Построение базы знаний (offline этап)

    Args:
        force_rebuild: пересоздать индексы даже если они существуют
        llm_clean: использовать LLM для очистки документов (медленно, но качественно)
        min_usefulness: минимальный порог полезности для LLM фильтрации (0.0-1.0)

    Returns:
        (embedding_indexer, bm25_indexer, chunks_df)
    """
    logger = get_logger(__name__)
    logger.info("="*80)
    logger.info("ЭТАП 1: ПОСТРОЕНИЕ БАЗЫ ЗНАНИЙ (OFFLINE)")
    logger.info("="*80)

    chunks_path = PROCESSED_DIR / "chunks.pkl"
    bm25_path = MODELS_DIR / "bm25.pkl"

    # Определяем режим работы
    use_weaviate = USE_WEAVIATE and WEAVIATE_AVAILABLE

    if use_weaviate:
        logger.info("[РЕЖИМ] Используется Weaviate для векторного поиска")

        # Проверяем существуют ли чанки
        if not force_rebuild and chunks_path.exists():
            logger.info("Чанки уже существуют. Загружаем...")

            # Загрузка чанков
            chunks_df = pd.read_pickle(chunks_path)
            logger.info(f"Загружено {len(chunks_df)} чанков")

            # Подключаемся к Weaviate
            try:
                weaviate_indexer = WeaviateIndexer()
                # Сохраняем метаданные
                weaviate_indexer.chunk_metadata = chunks_df

                logger.info("✓ Подключено к Weaviate")
                logger.info("Weaviate содержит векторный индекс + BM25")
                logger.info("Для переиндексации используйте --force")

                # Для Weaviate BM25 не нужен (встроен в Weaviate)
                return weaviate_indexer, None, chunks_df

            except Exception as e:
                logger.warning(f"Не удалось подключиться к Weaviate: {e}")
                logger.info("Убедитесь что Weaviate запущен: docker-compose up -d")
                logger.info("Или установите USE_WEAVIATE=false для использования FAISS")
                raise

        # Строим индексы с нуля
        logger.info("Построение новых индексов...")

    else:
        logger.info("[РЕЖИМ] Используется FAISS для векторного поиска")

        faiss_path = MODELS_DIR / "faiss.index"

        # Проверяем существуют ли индексы (FAISS режим)
        if not force_rebuild and chunks_path.exists() and faiss_path.exists() and bm25_path.exists():
            logger.info("Индексы уже существуют. Загружаем...")

            # Загрузка чанков
            chunks_df = pd.read_pickle(chunks_path)
            logger.info(f"Загружено {len(chunks_df)} чанков")

            # Загрузка индексов
            embedding_indexer = EmbeddingIndexer()
            embedding_indexer.load_index(str(faiss_path))
            embedding_indexer.chunk_metadata = chunks_df

            bm25_indexer = BM25Indexer()
            bm25_indexer.load_index(str(bm25_path))

            return embedding_indexer, bm25_indexer, chunks_df

        logger.info("Построение новых индексов...")

    # === ОБЩАЯ ЧАСТЬ: Предобработка и чанкинг ===

    # 1. Загрузка и предобработка документов
    logger.info("1. Предобработка документов...")
    with log_timing(logger, "Предобработка документов"):
        documents_df = load_and_preprocess_documents(
            str(WEBSITES_CSV),
            apply_lemmatization=False  # Отключаем для скорости
        )
        logger.info(f"Загружено документов: {len(documents_df)}")

    # 1.5. LLM очистка (опционально)
    if llm_clean:
        logger.info("1.5. LLM-очистка документов (это может занять несколько часов)...")
        logger.info(f"Минимальный порог полезности: {min_usefulness}")

        try:
            with log_timing(logger, "LLM-очистка документов"):
                documents_df = apply_llm_cleaning(
                    documents_df,
                    min_usefulness=min_usefulness,
                    verbose=True
                )

            # Используем clean_text вместо text для дальнейшей обработки
            if 'clean_text' in documents_df.columns:
                documents_df['text'] = documents_df['clean_text']

            logger.info(f"✅ LLM-очистка завершена! Документов после фильтрации: {len(documents_df)}")

        except Exception as e:
            logger.error(f"ОШИБКА LLM-очистки: {e}")
            logger.info("Продолжаем с исходными документами без LLM обработки")

    # 2. Разбиение на чанки
    logger.info("2. Разбиение на чанки...")
    with log_timing(logger, "Чанкинг документов"):
        chunks_df = create_chunks_from_documents(documents_df, method='words')
    logger.info(f"Всего чанков: {len(chunks_df)}")

    # Сохранение чанков
    chunks_df.to_pickle(chunks_path)
    logger.info(f"Чанки сохранены: {chunks_path}")

    # 3. Построение векторного индекса
    if use_weaviate:
        logger.info("3. Построение Weaviate индекса (с встроенным BM25)...")

        try:
            with log_timing(logger, "Индексация в Weaviate"):
                weaviate_indexer = WeaviateIndexer()

                # Очищаем предыдущие данные если force_rebuild
                if force_rebuild:
                    logger.info("Очистка предыдущих данных в Weaviate...")
                    weaviate_indexer.delete_all()

                # Индексируем документы (Weaviate автоматически создаст BM25 индекс)
                weaviate_indexer.index_documents(chunks_df, show_progress=True)

            # Сохраняем метаданные
            weaviate_indexer.chunk_metadata = chunks_df

            logger.info("✓ Weaviate индекс построен успешно!")
            logger.info("Включает: векторный индекс + BM25 (гибридный поиск)")

            # Для Weaviate не нужен отдельный BM25
            return weaviate_indexer, None, chunks_df

        except Exception as e:
            logger.error(f"Ошибка при построении Weaviate индекса: {e}")
            logger.info("Убедитесь что Weaviate запущен: docker-compose up -d")
            raise

    else:
        logger.info("3. Построение BM25 индекса...")
        with log_timing(logger, "BM25 индексация"):
            bm25_indexer = BM25Indexer()
            texts = chunks_df['text'].tolist()
            bm25_indexer.build_index(texts)
            bm25_indexer.save_index(str(bm25_path))

        logger.info("4. Построение FAISS индекса...")
        with log_timing(logger, "FAISS индексация"):
            embedding_indexer = EmbeddingIndexer()
            embeddings = embedding_indexer.create_embeddings(texts)
            embedding_indexer.build_faiss_index(embeddings)
            embedding_indexer.chunk_metadata = chunks_df
            embedding_indexer.save_index(str(MODELS_DIR / "faiss.index"))

        logger.info("База знаний построена успешно!")
        return embedding_indexer, bm25_indexer, chunks_df


def process_questions(embedding_indexer, bm25_indexer,
                     questions_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Обработка вопросов (online этап)

    Args:
        embedding_indexer: векторный индексер
        bm25_indexer: BM25 индексер
        questions_df: DataFrame с вопросами (если None - загружаем из файла)

    Returns:
        DataFrame с результатами
    """
    logger = get_logger(__name__)
    logger.info("="*80)
    logger.info("ЭТАП 2: ОБРАБОТКА ВОПРОСОВ (ONLINE)")
    logger.info("="*80)

    # Загрузка вопросов если не переданы
    if questions_df is None:
        questions_df = load_and_preprocess_questions(
            str(QUESTIONS_CSV),
            apply_lemmatization=False
        )

    # Создание RAG пайплайна
    pipeline = RAGPipeline(embedding_indexer, bm25_indexer)

    # Обработка каждого вопроса
    results = []

    logger.info(f"Обработка {len(questions_df)} вопросов...")

    started_at = time.time()
    last_partial_save = time.time()
    save_every = 50  # каждые N вопросов сохраняем частичный файл
    partial_path = OUTPUTS_DIR / "submission_partial.csv"

    for idx, row in tqdm(questions_df.iterrows(), total=len(questions_df)):
        q_id = row['q_id']
        query = row['processed_query']

        try:
            # Поиск релевантных документов
            t0 = time.time()
            result = pipeline.search(query)
            dt = time.time() - t0

            # Формируем результат
            doc_ids = result['documents_id']

            # Дополняем до 5 документов если нужно
            while len(doc_ids) < 5:
                doc_ids.append(-1)  # заглушка

            results.append({
                'q_id': q_id,
                'web_list': str(doc_ids[:5])
            })

            if (idx + 1) % save_every == 0:
                # Сохраняем частичный результат
                pd.DataFrame(results).to_csv(partial_path, index=False)
                elapsed = time.time() - started_at
                per_q = elapsed / (idx + 1)
                eta = per_q * (len(questions_df) - (idx + 1))
                logger.info(f"Прогресс: {idx + 1}/{len(questions_df)} | {per_q:.2f}s/вопрос | ETA ~ {eta/60:.1f} мин | частичный файл: {partial_path}")

            # Логируем короткую метрику
            logger.debug(f"q_id={q_id} | кандидатов={result.get('num_candidates', 'NA')} | время={dt:.2f}s | docs={doc_ids[:5]}")

        except Exception as e:
            logger.error(f"Ошибка при обработке вопроса {q_id}: {e}")
            # Возвращаем пустой результат
            results.append({
                'q_id': q_id,
                'web_list': '[-1, -1, -1, -1, -1]'
            })

    results_df = pd.DataFrame(results)
    return results_df


def evaluate_on_examples(embedding_indexer, bm25_indexer):
    """
    Оценка качества на эталонных примерах

    Args:
        embedding_indexer: векторный индексер
        bm25_indexer: BM25 индексер

    Returns:
        средняя метрика
    """
    logger = get_logger(__name__)
    logger.info("="*80)
    logger.info("ОЦЕНКА НА ЭТАЛОННЫХ ПРИМЕРАХ")
    logger.info("="*80)

    from src.config import EXAMPLES_CSV

    examples_df = pd.read_csv(EXAMPLES_CSV)
    pipeline = RAGPipeline(embedding_indexer, bm25_indexer)

    # Извлекаем релевантные web_id из chunk'ов
    # (это требует дополнительной логики, упростим)

    logger.info(f"Загружено {len(examples_df)} примеров для валидации")
    logger.info("Детальная оценка на примерах будет реализована отдельно")

    # TODO: Реализовать метрику recall@5
    # Для этого нужно извлечь web_id из chunk'ов в examples

    return None


def cmd_build(args):
    """Команда: построить базу знаний"""
    logger = get_logger(__name__)
    logger.info("="*80)
    logger.info("РЕЖИМ: ПОСТРОЕНИЕ БАЗЫ ЗНАНИЙ")
    logger.info("="*80)

    if args.llm_clean:
        logger.info("[LLM-РЕЖИМ] Включена очистка документов через LLM")
        logger.info(f"[LLM-РЕЖИМ] Минимальный порог полезности: {args.min_usefulness}")
        logger.info("[LLM-РЕЖИМ] Это увеличит время обработки в 10-20 раз!")

    embedding_indexer, bm25_indexer, chunks_df = build_knowledge_base(
        force_rebuild=args.force,
        llm_clean=args.llm_clean,
        min_usefulness=args.min_usefulness
    )

    logger.info("="*80)
    logger.info("[OK] БАЗА ЗНАНИЙ ПОСТРОЕНА УСПЕШНО")
    logger.info("="*80)
    logger.info(f"Всего чанков: {len(chunks_df)}")

    if USE_WEAVIATE and WEAVIATE_AVAILABLE:
        logger.info("Векторный индекс: Weaviate (http://localhost:8080)")
        logger.info("BM25 индекс: встроен в Weaviate (гибридный поиск)")
    else:
        logger.info(f"Векторный индекс: {MODELS_DIR / 'faiss.index'}")
        logger.info(f"BM25 индекс: {MODELS_DIR / 'bm25.pkl'}")


def cmd_search(args):
    """Команда: обработать вопросы"""
    logger = get_logger(__name__)
    logger.info("="*80)
    logger.info("РЕЖИМ: ОБРАБОТКА ВОПРОСОВ")
    logger.info("="*80)

    # Загрузка существующих индексов
    logger.info("Загрузка базы знаний...")

    chunks_path = PROCESSED_DIR / "chunks.pkl"

    if not chunks_path.exists():
        logger.error("ОШИБКА: База знаний не найдена! Сначала выполните: python main_pipeline.py build")
        return

    # Определяем режим работы
    use_weaviate = USE_WEAVIATE and WEAVIATE_AVAILABLE

    # Загрузка чанков
    chunks_df = pd.read_pickle(chunks_path)
    logger.info(f"Загружено {len(chunks_df)} чанков")

    # Загрузка векторного индекса
    if use_weaviate:
        logger.info("Используется Weaviate (векторный поиск + BM25)")
        try:
            embedding_indexer = WeaviateIndexer()
            embedding_indexer.chunk_metadata = chunks_df
            bm25_indexer = None  # не нужен для Weaviate
            logger.info("✓ Подключено к Weaviate")
        except Exception as e:
            logger.error(f"Не удалось подключиться к Weaviate: {e}")
            logger.info("Убедитесь что Weaviate запущен: docker-compose up -d")
            return
    else:
        logger.info("Используется FAISS для векторного поиска")
        faiss_path = MODELS_DIR / "faiss.index"
        bm25_path = MODELS_DIR / "bm25.pkl"

        if not faiss_path.exists() or not bm25_path.exists():
            logger.error("ОШИБКА: FAISS или BM25 индекс не найден! Сначала выполните: python main_pipeline.py build")
            return

        # Загрузка BM25
        bm25_indexer = BM25Indexer()
        bm25_indexer.load_index(str(bm25_path))

        # Загрузка FAISS
        embedding_indexer = EmbeddingIndexer()
        embedding_indexer.load_index(str(faiss_path))
        embedding_indexer.chunk_metadata = chunks_df

    # Оптимизация параметров (опционально)
    if args.optimize:
        logger.info("="*80)
        logger.info("GRID SEARCH ОПТИМИЗАЦИЯ ПАРАМЕТРОВ")
        logger.info("="*80)

        # Загружаем вопросы для оптимизации
        optimize_questions_df = load_and_preprocess_questions(
            str(QUESTIONS_CSV),
            apply_lemmatization=False
        )

        # Создаем временный retriever для оптимизации
        from src.retrieval import HybridRetriever
        temp_retriever = HybridRetriever(embedding_indexer, bm25_indexer)

        # Запускаем grid search
        try:
            with log_timing(logger, "Grid Search"):
                best_params = optimize_rag_params(
                    retriever=temp_retriever,
                    questions_df=optimize_questions_df,
                    mode=args.optimize_mode,
                    sample_size=args.optimize_sample
                )
            logger.info("✅ Параметры оптимизированы! Продолжаем с лучшими параметрами...")

        except Exception as e:
            logger.warning(f"ОШИБКА оптимизации: {e}")
            logger.info("Продолжаем с текущими параметрами из config.py")

    # Обработка вопросов
    if args.limit:
        logger.info(f"Обработка первых {args.limit} вопросов (режим тестирования)")
        questions_df = load_and_preprocess_questions(
            str(QUESTIONS_CSV),
            apply_lemmatization=False
        ).head(args.limit)
    else:
        logger.info("Обработка всех вопросов")
        questions_df = None

    with log_timing(logger, "Обработка всех вопросов"):
        results_df = process_questions(embedding_indexer, bm25_indexer, questions_df)

    # Сохранение результатов
    output_path = OUTPUTS_DIR / "submission.csv"
    results_df.to_csv(output_path, index=False)

    logger.info("="*80)
    logger.info("[OK] ОБРАБОТКА ЗАВЕРШЕНА")
    logger.info("="*80)
    logger.info(f"Результаты: {output_path}")
    logger.info(f"Обработано вопросов: {len(results_df)}")


def cmd_all(args):
    """Команда: полный цикл (build + search)"""
    logger = get_logger(__name__)
    logger.info("="*80)
    logger.info("РЕЖИМ: ПОЛНЫЙ ЦИКЛ (BUILD + SEARCH)")
    logger.info("="*80)

    if hasattr(args, 'llm_clean') and args.llm_clean:
        logger.info("[LLM-РЕЖИМ] Включена очистка документов через LLM")

    # 1. Построение базы знаний
    logger.info("[1/2] Построение базы знаний...")
    with log_timing(logger, "Полный цикл: build"):
        embedding_indexer, bm25_indexer, chunks_df = build_knowledge_base(
            force_rebuild=args.force,
            llm_clean=getattr(args, 'llm_clean', False),
            min_usefulness=getattr(args, 'min_usefulness', 0.3)
        )

    # 2. Обработка вопросов
    logger.info("[2/2] Обработка вопросов...")

    if args.limit:
        logger.info(f"Обработка первых {args.limit} вопросов (режим тестирования)")
        questions_df = load_and_preprocess_questions(
            str(QUESTIONS_CSV),
            apply_lemmatization=False
        ).head(args.limit)
    else:
        questions_df = None

    with log_timing(logger, "Полный цикл: search"):
        results_df = process_questions(embedding_indexer, bm25_indexer, questions_df)

    # 3. Сохранение результатов
    output_path = OUTPUTS_DIR / "submission.csv"
    results_df.to_csv(output_path, index=False)

    logger.info("="*80)
    logger.info("[OK] ПОЛНЫЙ ЦИКЛ ЗАВЕРШЕН")
    logger.info("="*80)
    logger.info(f"Результаты: {output_path}")
    logger.info(f"Обработано вопросов: {len(results_df)}")


def cmd_evaluate(args):
    """Команда: оценка на примерах"""
    logger = get_logger(__name__)
    logger.info("="*80)
    logger.info("РЕЖИМ: ОЦЕНКА НА ПРИМЕРАХ")
    logger.info("="*80)

    # Загрузка индексов
    chunks_path = PROCESSED_DIR / "chunks.pkl"
    faiss_path = MODELS_DIR / "faiss.index"
    bm25_path = MODELS_DIR / "bm25.pkl"

    if not chunks_path.exists() or not faiss_path.exists() or not bm25_path.exists():
        logger.error("ОШИБКА: База знаний не найдена! Сначала выполните: python main_pipeline.py build")
        return

    chunks_df = pd.read_pickle(chunks_path)
    embedding_indexer = EmbeddingIndexer()
    embedding_indexer.load_index(str(faiss_path))
    embedding_indexer.chunk_metadata = chunks_df

    bm25_indexer = BM25Indexer()
    bm25_indexer.load_index(str(bm25_path))

    # Оценка
    evaluate_on_examples(embedding_indexer, bm25_indexer)


def main():
    """Главная функция с парсингом аргументов"""
    # Инициализация логирования (до парсинга, чтобы ловить ранние сообщения)
    setup_logging(level=LOG_LEVEL, log_file=LOG_FILE)
    logger = get_logger(__name__)

    parser = argparse.ArgumentParser(
        description="RAG пайплайн для поиска релевантных документов Альфа-Банка",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

BUILD (создание базы знаний):
  python main_pipeline.py build                           # Построить базу знаний
  python main_pipeline.py build --force                   # Пересоздать базу знаний
  python main_pipeline.py build --llm-clean               # С LLM очисткой документов
  python main_pipeline.py build --llm-clean --min-usefulness 0.5  # С фильтрацией

SEARCH (поиск ответов):
  python main_pipeline.py search                          # Обработать все вопросы
  python main_pipeline.py search --limit 10               # Тест на 10 вопросах
  python main_pipeline.py search --optimize               # С оптимизацией параметров (grid search)
  python main_pipeline.py search --optimize --optimize-mode full  # Полная оптимизация

ALL (полный цикл):
  python main_pipeline.py all                             # Build + Search
  python main_pipeline.py all --llm-clean --optimize      # С LLM очисткой и оптимизацией

EVALUATE:
  python main_pipeline.py evaluate                        # Оценка на примерах
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Команда для выполнения')

    # Команда: build
    parser_build = subparsers.add_parser(
        'build',
        help='Построить базу знаний (индексация документов)'
    )
    parser_build.add_argument(
        '--force',
        action='store_true',
        help='Пересоздать индексы даже если они существуют'
    )
    parser_build.add_argument(
        '--llm-clean',
        action='store_true',
        help='Использовать LLM для очистки документов (медленно, +качество)'
    )
    parser_build.add_argument(
        '--min-usefulness',
        type=float,
        default=0.3,
        help='Минимальный порог полезности для LLM фильтрации (0.0-1.0, по умолчанию 0.3)'
    )
    parser_build.set_defaults(func=cmd_build)

    # Команда: search
    parser_search = subparsers.add_parser(
        'search',
        help='Обработать вопросы (требует готовую базу знаний)'
    )
    parser_search.add_argument(
        '--limit',
        type=int,
        help='Обработать только первые N вопросов (для тестирования)'
    )
    parser_search.add_argument(
        '--optimize',
        action='store_true',
        help='Запустить grid search для оптимизации параметров перед поиском'
    )
    parser_search.add_argument(
        '--optimize-sample',
        type=int,
        default=50,
        help='Размер выборки для grid search (по умолчанию 50)'
    )
    parser_search.add_argument(
        '--optimize-mode',
        type=str,
        default='quick',
        choices=['quick', 'full'],
        help='Режим grid search: quick (быстрый) или full (полный)'
    )
    parser_search.set_defaults(func=cmd_search)

    # Команда: all
    parser_all = subparsers.add_parser(
        'all',
        help='Полный цикл: построить базу знаний и обработать вопросы'
    )
    parser_all.add_argument(
        '--force',
        action='store_true',
        help='Пересоздать индексы даже если они существуют'
    )
    parser_all.add_argument(
        '--llm-clean',
        action='store_true',
        help='Использовать LLM для очистки документов (медленно, +качество)'
    )
    parser_all.add_argument(
        '--min-usefulness',
        type=float,
        default=0.3,
        help='Минимальный порог полезности для LLM фильтрации (0.0-1.0, по умолчанию 0.3)'
    )
    parser_all.add_argument(
        '--limit',
        type=int,
        help='Обработать только первые N вопросов (для тестирования)'
    )
    parser_all.set_defaults(func=cmd_all)

    # Команда: evaluate
    parser_eval = subparsers.add_parser(
        'evaluate',
        help='Оценка качества на эталонных примерах'
    )
    parser_eval.set_defaults(func=cmd_evaluate)

    # Парсинг аргументов
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Вывод заголовка
    logger.info("="*80)
    logger.info("RAG ПАЙПЛАЙН ДЛЯ ПОИСКА РЕЛЕВАНТНЫХ ДОКУМЕНТОВ АЛЬФА-БАНКА")
    logger.info("="*80)

    if USE_WEAVIATE and WEAVIATE_AVAILABLE:
        logger.info("Используется Weaviate для векторного поиска")
    else:
        logger.info("Используется FAISS для векторного поиска")

    if USE_WEAVIATE and not WEAVIATE_AVAILABLE:
        logger.critical("USE_WEAVIATE=true, но weaviate-client не установлен!")

    # Выполнение команды
    args.func(args)

    logger.info("[OK] Готово!")


if __name__ == "__main__":
    main()
