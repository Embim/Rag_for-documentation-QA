"""
Скрипт для запуска Grid Search с LLM-based оценкой

Использует гибридную оценку:
- 30% косинусное расстояние (semantic similarity)
- 70% LLM метрики (Context Relevance, Precision, Sufficiency)

Использование:
    python scripts/run_grid_search.py --mode quick --sample 50
    python scripts/run_grid_search.py --mode full --sample 100 --no-llm
"""
import argparse
import pandas as pd
import sys
from pathlib import Path

# Добавляем корень проекта в путь
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    QUESTIONS_CSV,
    PROCESSED_DIR,
    MODELS_DIR,
    USE_WEAVIATE,
    GRID_SEARCH_SAMPLE_SIZE,
    GRID_SEARCH_MODE,
    GRID_SEARCH_USE_LLM
)
from src.preprocessing import load_and_preprocess_questions
from src.indexing import BM25Indexer, EmbeddingIndexer, WeaviateIndexer
from src.retrieval import HybridRetriever
from src.grid_search_optimizer import optimize_rag_params
from src.logger import setup_logging, get_logger, log_timing


def load_indexes():
    """
    Загружает индексы (BM25 + Embedding/Weaviate)
    """
    logger = get_logger(__name__)

    chunks_path = PROCESSED_DIR / "chunks.pkl"

    if not chunks_path.exists():
        logger.error("ОШИБКА: База знаний не найдена!")
        logger.error("Сначала выполните: python main_pipeline.py build")
        sys.exit(1)

    # Загрузка чанков
    chunks_df = pd.read_pickle(chunks_path)
    logger.info(f"Загружено {len(chunks_df)} чанков")

    # Определяем режим работы
    use_weaviate = USE_WEAVIATE

    if use_weaviate:
        logger.info("Используется Weaviate (векторный поиск + BM25)")
        try:
            embedding_indexer = WeaviateIndexer()
            embedding_indexer.chunk_metadata = chunks_df
            bm25_indexer = None
            logger.info("✓ Подключено к Weaviate")
        except Exception as e:
            logger.error(f"Не удалось подключиться к Weaviate: {e}")
            logger.info("Убедитесь что Weaviate запущен: docker-compose up -d")
            sys.exit(1)
    else:
        logger.info("Используется FAISS для векторного поиска")
        faiss_path = MODELS_DIR / "faiss.index"
        bm25_path = MODELS_DIR / "bm25.pkl"

        if not faiss_path.exists() or not bm25_path.exists():
            logger.error("ОШИБКА: FAISS или BM25 индекс не найден!")
            logger.error("Сначала выполните: python main_pipeline.py build")
            sys.exit(1)

        # Загрузка BM25
        bm25_indexer = BM25Indexer()
        bm25_indexer.load_index(str(bm25_path))

        # Загрузка FAISS
        embedding_indexer = EmbeddingIndexer()
        embedding_indexer.load_index(str(faiss_path))
        embedding_indexer.chunk_metadata = chunks_df

    return embedding_indexer, bm25_indexer


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description="Grid Search оптимизация RAG параметров с LLM-based оценкой",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Быстрый grid search на 50 вопросах
  python scripts/run_grid_search.py --mode quick --sample 50

  # Полный grid search на 100 вопросах
  python scripts/run_grid_search.py --mode full --sample 100

  # Без LLM оценки (только косинусное расстояние, быстро)
  python scripts/run_grid_search.py --mode quick --sample 30 --no-llm

Режимы:
  quick - быстрый (3x3x2x3 = 54 комбинации)
  full  - полный (7x7x5x5 = 1225 комбинаций, очень долго с LLM!)

Рекомендации:
  - Для первого запуска используйте --mode quick --sample 30
  - С LLM: ~10-30 сек на комбинацию (зависит от GPU)
  - Без LLM: ~1-5 сек на комбинацию
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        default=None,
        choices=['quick', 'full'],
        help=f'Режим grid search (по умолчанию из config: {GRID_SEARCH_MODE})'
    )

    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help=f'Размер выборки вопросов для оценки (по умолчанию из config: {GRID_SEARCH_SAMPLE_SIZE})'
    )

    parser.add_argument(
        '--no-llm',
        action='store_true',
        help=f'Отключить LLM оценку (по умолчанию из config: {"включена" if GRID_SEARCH_USE_LLM else "выключена"})'
    )

    args = parser.parse_args()

    # Используем дефолты из config если не указано
    mode = args.mode if args.mode is not None else GRID_SEARCH_MODE
    sample = args.sample if args.sample is not None else GRID_SEARCH_SAMPLE_SIZE
    use_llm = not args.no_llm  # если --no-llm передан, отключаем LLM

    # Если --no-llm НЕ передан, используем значение из config
    if not args.no_llm:
        use_llm = GRID_SEARCH_USE_LLM

    # Инициализация логирования
    setup_logging()
    logger = get_logger(__name__)

    logger.info("="*80)
    logger.info("GRID SEARCH ОПТИМИЗАЦИЯ RAG С LLM-BASED ОЦЕНКОЙ")
    logger.info("="*80)
    logger.info(f"Режим: {mode}")
    logger.info(f"Размер выборки: {sample} вопросов")
    logger.info(f"LLM оценка: {'ВЫКЛ (только cosine)' if not use_llm else 'ВКЛ (cosine 30% + LLM 70%)'}")

    # Показываем откуда взяты параметры
    logger.info("\nИсточники параметров:")
    logger.info(f"  mode: {'CLI аргумент' if args.mode else 'config.GRID_SEARCH_MODE'}")
    logger.info(f"  sample: {'CLI аргумент' if args.sample else 'config.GRID_SEARCH_SAMPLE_SIZE'}")
    logger.info(f"  use_llm: {'--no-llm флаг' if args.no_llm else 'config.GRID_SEARCH_USE_LLM'}")

    # 1. Загрузка индексов
    logger.info("\n[1/3] Загрузка индексов...")
    with log_timing(logger, "Загрузка индексов"):
        embedding_indexer, bm25_indexer = load_indexes()

    # 2. Загрузка вопросов
    logger.info("\n[2/3] Загрузка вопросов...")
    with log_timing(logger, "Загрузка вопросов"):
        questions_df = load_and_preprocess_questions(
            str(QUESTIONS_CSV),
            apply_lemmatization=False
        )
        logger.info(f"Загружено {len(questions_df)} вопросов")

    # 3. Создание retriever
    logger.info("\n[3/3] Создание retriever...")
    retriever = HybridRetriever(embedding_indexer, bm25_indexer)

    # 4. Запуск Grid Search
    logger.info("\n" + "="*80)
    logger.info("ЗАПУСК GRID SEARCH")
    logger.info("="*80)

    if use_llm:
        logger.info("⚠️  LLM оценка ВКЛЮЧЕНА - это будет медленно!")
        logger.info(f"⚠️  Примерное время: {sample} вопросов × комбинации × ~15 сек")
        logger.info("⚠️  Для быстрого теста используйте --no-llm")

    with log_timing(logger, "Grid Search"):
        best_params = optimize_rag_params(
            retriever=retriever,
            questions_df=questions_df,
            mode=mode,
            sample_size=sample,
            use_llm_eval=use_llm
        )

    logger.info("\n" + "="*80)
    logger.info("✅ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
    logger.info("="*80)
    logger.info("Лучшие параметры сохранены в src/config.py")
    logger.info(f"TOP_K_DENSE = {best_params['TOP_K_DENSE']}")
    logger.info(f"TOP_K_BM25 = {best_params['TOP_K_BM25']}")
    logger.info(f"TOP_K_RERANK = {best_params['TOP_K_RERANK']}")
    logger.info(f"HYBRID_ALPHA = {best_params['HYBRID_ALPHA']}")


if __name__ == "__main__":
    main()
