"""
Потоковая обработка документов для построения базы знаний

Вместо загрузки всех документов в память (batch processing):
    documents_df (все) → llm_clean (все) → chunk (все) → index (все)

Используем потоковую обработку (streaming):
    для каждого документа: load → clean → chunk → накопить → index батч

Преимущества:
- Меньше памяти (не держим весь DataFrame)
- Быстрее (индексируем по мере обработки для Weaviate)
- Прогресс виден сразу
"""
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional, Union
import gc

from src.preprocessing import TextPreprocessor
from src.chunking import DocumentChunker
from src.logger import get_logger, log_timing

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class StreamingDocumentProcessor:
    """
    Потоковая обработка документов: load → clean → chunk → accumulate

    Уменьшает использование памяти обрабатывая документы по одному
    вместо загрузки всего датасета в память.
    """

    def __init__(self,
                 llm_clean: bool = False,
                 min_usefulness: float = 0.3,
                 chunk_batch_size: int = 500,
                 csv_chunksize: int = 10):
        """
        Args:
            llm_clean: использовать LLM для очистки документов
            min_usefulness: минимальный порог полезности (0.0-1.0)
            chunk_batch_size: сколько чанков накапливать перед индексацией батча
            csv_chunksize: сколько строк CSV читать за раз
        """
        self.llm_clean = llm_clean
        self.min_usefulness = min_usefulness
        self.chunk_batch_size = chunk_batch_size
        self.csv_chunksize = csv_chunksize

        self.logger = get_logger(__name__)

        # Инициализация компонентов
        self.preprocessor = TextPreprocessor()
        self.chunker = DocumentChunker()

        # LLM cleaner (загружаем только если нужен)
        self.llm_cleaner = None
        if llm_clean:
            try:
                from src.llm_preprocessing import LLMDocumentCleaner
                self.llm_cleaner = LLMDocumentCleaner(verbose=True)
                self.llm_cleaner.load_model()
                self.logger.info("✓ LLM Document Cleaner загружен")
            except Exception as e:
                self.logger.warning(f"Не удалось загрузить LLM cleaner: {e}")
                self.logger.warning("Продолжаем без LLM очистки")
                self.llm_clean = False

    def process_document(self, doc_row: pd.Series) -> List[Dict]:
        """
        Обработка одного документа: preprocess → llm_clean → chunk

        Args:
            doc_row: строка из CSV (pandas Series)

        Returns:
            list of chunk dicts
        """
        # 1. Предобработка
        processed_text = self.preprocessor.preprocess_document(
            text=doc_row.get('text', ''),
            title=doc_row.get('title', ''),
            apply_lemmatization=False
        )

        if not processed_text or len(processed_text.strip()) < 10:
            # Слишком короткий текст - пропускаем
            return []

        # 2. LLM очистка (если включена)
        entities = ''
        topics = ''

        if self.llm_clean and self.llm_cleaner:
            try:
                cleaned_result = self.llm_cleaner.clean_document(processed_text)

                # Фильтруем по полезности
                usefulness = cleaned_result.get('usefulness_score', 1.0)
                is_useful = cleaned_result.get('is_useful', True)

                if is_useful and usefulness >= self.min_usefulness:
                    # Используем очищенный текст
                    processed_text = cleaned_result.get('clean_text', processed_text)

                    # Собираем метаданные
                    products = cleaned_result.get('products', [])
                    actions = cleaned_result.get('actions', [])
                    conditions = cleaned_result.get('conditions', [])

                    # Комбинируем entities
                    all_entities = products + actions + conditions
                    if all_entities:
                        entities = json.dumps(all_entities, ensure_ascii=False)

                    # Темы
                    topics_list = cleaned_result.get('topics', [])
                    if topics_list:
                        topics = json.dumps(topics_list, ensure_ascii=False)
                else:
                    # Документ бесполезен - пропускаем
                    self.logger.debug(f"Пропущен документ web_id={doc_row.get('web_id')} (usefulness={usefulness:.2f})")
                    return []

            except Exception as e:
                # Ошибка LLM - используем исходный текст
                self.logger.debug(f"Ошибка LLM для web_id={doc_row.get('web_id')}: {e}")
                pass

        # 3. Чанкинг
        chunks = self.chunker.chunk_by_words(
            text=processed_text,
            web_id=int(doc_row.get('web_id', 0)),
            title=str(doc_row.get('title', '')),
            url=str(doc_row.get('url', '')),
            kind=str(doc_row.get('kind', '')),
            entities=entities,
            topics=topics
        )

        return chunks

    def process_csv_streaming(self,
                             csv_path: str,
                             indexer = None,
                             for_weaviate: bool = False) -> Optional[pd.DataFrame]:
        """
        Потоковая обработка CSV:
        - Читает по csv_chunksize документов за раз
        - Обрабатывает каждый: preprocess → llm_clean → chunk
        - Накапливает чанки в батчи
        - Для Weaviate: индексирует батчи сразу и очищает память
        - Для FAISS: возвращает все чанки в конце

        Args:
            csv_path: путь к websites.csv
            indexer: WeaviateIndexer (для streaming индексации) или None
            for_weaviate: True если используем Weaviate (индексируем сразу)

        Returns:
            DataFrame со всеми чанками (для FAISS) или None (для Weaviate)
        """
        self.logger.info("="*80)
        self.logger.info("ПОТОКОВАЯ ОБРАБОТКА ДОКУМЕНТОВ")
        self.logger.info("="*80)
        self.logger.info(f"Режим: {'Weaviate (streaming index)' if for_weaviate else 'FAISS (accumulate all)'}")
        self.logger.info(f"LLM очистка: {'ВКЛ' if self.llm_clean else 'ВЫКЛ'}")
        if self.llm_clean:
            self.logger.info(f"Порог полезности: {self.min_usefulness}")
        self.logger.info(f"Размер батча чанков: {self.chunk_batch_size}")
        self.logger.info(f"Размер чанка CSV: {self.csv_chunksize} документов")
        self.logger.info("="*80)

        all_chunks = []
        chunk_batch = []

        total_docs_processed = 0
        total_docs_filtered = 0
        total_chunks_created = 0
        batches_indexed = 0

        # Подсчитаем общее количество документов для прогресса
        total_docs = sum(1 for _ in open(csv_path, encoding='utf-8')) - 1  # -1 для header
        self.logger.info(f"Всего документов в CSV: {total_docs}")

        # Читаем CSV по частям (streaming)
        csv_reader = pd.read_csv(csv_path, chunksize=self.csv_chunksize)

        for csv_chunk_idx, doc_chunk_df in enumerate(csv_reader):
            self.logger.info(f"\n[Батч CSV {csv_chunk_idx + 1}] Обработка {len(doc_chunk_df)} документов...")

            for idx, doc_row in doc_chunk_df.iterrows():
                # Обработка одного документа
                doc_chunks = self.process_document(doc_row)

                if doc_chunks:
                    chunk_batch.extend(doc_chunks)
                    total_chunks_created += len(doc_chunks)
                else:
                    total_docs_filtered += 1

                total_docs_processed += 1

                # Если накопили достаточно чанков для батча
                if len(chunk_batch) >= self.chunk_batch_size:
                    # Конвертируем в DataFrame
                    batch_df = pd.DataFrame(chunk_batch)

                    if for_weaviate and indexer is not None:
                        # Weaviate: индексируем сразу
                        self.logger.info(f"  → Индексация батча {batches_indexed + 1}: {len(batch_df)} чанков в Weaviate...")

                        with log_timing(self.logger, f"Индексация батча {batches_indexed + 1}"):
                            indexer.index_documents(batch_df, show_progress=False)

                        batches_indexed += 1

                        # Для Weaviate: сохраняем метаданные чанков (без эмбеддингов)
                        all_chunks.extend(chunk_batch)

                        # Очищаем батч из памяти
                        chunk_batch = []
                        del batch_df

                        # Чистим GPU память
                        gc.collect()
                        if TORCH_AVAILABLE and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        # FAISS: накапливаем все чанки
                        all_chunks.extend(chunk_batch)
                        chunk_batch = []

            # Прогресс
            self.logger.info(f"  Прогресс: {total_docs_processed}/{total_docs} документов | "
                           f"Чанков создано: {total_chunks_created} | "
                           f"Отфильтровано: {total_docs_filtered}")

        # Обработка остатка
        if chunk_batch:
            batch_df = pd.DataFrame(chunk_batch)

            if for_weaviate and indexer is not None:
                self.logger.info(f"  → Индексация финального батча: {len(batch_df)} чанков...")
                indexer.index_documents(batch_df, show_progress=False)
                batches_indexed += 1

                # Сохраняем метаданные
                all_chunks.extend(chunk_batch)
                del batch_df
            else:
                all_chunks.extend(chunk_batch)

        # Итоговая статистика
        self.logger.info("\n" + "="*80)
        self.logger.info("СТАТИСТИКА ОБРАБОТКИ")
        self.logger.info("="*80)
        self.logger.info(f"Документов обработано: {total_docs_processed}")
        self.logger.info(f"Документов отфильтровано: {total_docs_filtered} ({total_docs_filtered/max(total_docs_processed,1)*100:.1f}%)")
        self.logger.info(f"Чанков создано: {total_chunks_created}")
        self.logger.info(f"Среднее чанков/документ: {total_chunks_created/max(total_docs_processed-total_docs_filtered,1):.1f}")

        if for_weaviate:
            self.logger.info(f"Батчей проиндексировано в Weaviate: {batches_indexed}")

        self.logger.info("="*80)

        # Конвертируем в DataFrame (для обоих режимов)
        if all_chunks:
            chunks_df = pd.DataFrame(all_chunks)
            self.logger.info(f"DataFrame создан: {len(chunks_df)} строк")
            return chunks_df
        else:
            self.logger.warning("Ни одного чанка не создано!")
            return pd.DataFrame()


def build_knowledge_base_streaming(csv_path: str,
                                   indexer = None,
                                   for_weaviate: bool = False,
                                   llm_clean: bool = False,
                                   min_usefulness: float = 0.3,
                                   chunk_batch_size: int = 500,
                                   csv_chunksize: int = 10) -> Optional[pd.DataFrame]:
    """
    Удобная функция для построения базы знаний потоковым методом

    Args:
        csv_path: путь к websites.csv
        indexer: WeaviateIndexer или None
        for_weaviate: True если используем Weaviate
        llm_clean: использовать LLM очистку
        min_usefulness: минимальный порог полезности
        chunk_batch_size: размер батча для индексации
        csv_chunksize: сколько документов читать за раз

    Returns:
        DataFrame с чанками (для FAISS) или None (для Weaviate)
    """
    processor = StreamingDocumentProcessor(
        llm_clean=llm_clean,
        min_usefulness=min_usefulness,
        chunk_batch_size=chunk_batch_size,
        csv_chunksize=csv_chunksize
    )

    chunks_df = processor.process_csv_streaming(
        csv_path=csv_path,
        indexer=indexer,
        for_weaviate=for_weaviate
    )

    return chunks_df


if __name__ == "__main__":
    # Простой тест
    print("StreamingDocumentProcessor")
    print("Используйте через main_pipeline.py build")
