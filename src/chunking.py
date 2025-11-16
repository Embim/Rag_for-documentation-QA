"""
Разбиение документов на чанки (фрагменты)
"""
import pandas as pd
from typing import List, Dict
from razdel import sentenize

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE
from src.logger import get_logger, log_timing


class DocumentChunker:
    """Класс для разбиения документов на чанки"""

    def __init__(self, chunk_size: int = CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP,
                 min_chunk_size: int = MIN_CHUNK_SIZE):
        """
        Args:
            chunk_size: размер чанка в словах
            chunk_overlap: перекрытие между чанками в словах
            min_chunk_size: минимальный размер чанка в словах
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_by_words(self, text: str, web_id: int, title: str = "",
                      url: str = "", kind: str = "", entities: str = "", topics: str = "") -> List[Dict]:
        """
        Разбиение текста на чанки скользящим окном по словам

        Args:
            text: текст для разбиения
            web_id: идентификатор исходного документа
            title: заголовок документа
            url: URL документа
            kind: тип документа

        Returns:
            список чанков с метаданными
        """
        if not text:
            return []

        words = text.split()
        chunks = []
        chunk_idx = 0

        # Префикс с заголовком документа для лучшего контекста
        title_prefix = f"[{title}] " if title else ""

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]

            # Пропускаем слишком маленькие чанки (кроме последнего)
            if len(chunk_words) < self.min_chunk_size and i + self.chunk_size < len(words):
                continue

            chunk_text = ' '.join(chunk_words)

            # Добавляем заголовок в начало чанка
            full_chunk_text = title_prefix + chunk_text

            # Базовые метаданные чанка
            chunk_data = {
                'chunk_id': f"{web_id}_{chunk_idx}",
                'web_id': web_id,
                'title': title,
                'text': full_chunk_text,
                'original_text': chunk_text,  # без префикса
                'chunk_index': chunk_idx,
                'start_word': i,
                'end_word': min(i + self.chunk_size, len(words)),
                # Дополнительные метаданные
                'url': url,
                'kind': kind,
                'word_count': len(chunk_words),
                'char_count': len(chunk_text),
            }

            # Прокидываем LLM-метаданные если есть
            if entities:
                chunk_data['entities'] = entities
            if topics:
                chunk_data['topics'] = topics

            chunks.append(chunk_data)

            chunk_idx += 1

        return chunks

    def chunk_by_sentences(self, text: str, web_id: int, title: str = "",
                          url: str = "", kind: str = "", entities: str = "", topics: str = "", max_sentences: int = 10) -> List[Dict]:
        """
        Разбиение текста на чанки по предложениям

        Args:
            text: текст для разбиения
            web_id: идентификатор документа
            title: заголовок
            url: URL документа
            kind: тип документа
            max_sentences: максимальное количество предложений в чанке

        Returns:
            список чанков
        """
        if not text:
            return []

        # Используем Razdel для сегментации предложений
        sentences = list(sentenize(text))

        chunks = []
        chunk_idx = 0
        title_prefix = f"[{title}] " if title else ""

        for i in range(0, len(sentences), max_sentences):
            chunk_sentences = sentences[i:i + max_sentences]
            chunk_text = ' '.join([s.text for s in chunk_sentences])

            # Проверка минимального размера
            if len(chunk_text.split()) < self.min_chunk_size:
                continue

            full_chunk_text = title_prefix + chunk_text

            chunk_data = {
                'chunk_id': f"{web_id}_{chunk_idx}",
                'web_id': web_id,
                'title': title,
                'text': full_chunk_text,
                'original_text': chunk_text,
                'chunk_index': chunk_idx,
                # Дополнительные метаданные
                'url': url,
                'kind': kind,
                'word_count': len(chunk_text.split()),
                'char_count': len(chunk_text),
            }
            if entities:
                chunk_data['entities'] = entities
            if topics:
                chunk_data['topics'] = topics

            chunks.append(chunk_data)

            chunk_idx += 1

        return chunks

    def chunk_documents_dataframe(self, df: pd.DataFrame,
                                  method: str = 'words') -> pd.DataFrame:
        """
        Разбиение всех документов из DataFrame на чанки

        Args:
            df: DataFrame с колонками web_id, title, processed_text, url, kind
            method: метод чанкинга ('words' или 'sentences')

        Returns:
            DataFrame с чанками
        """
        logger = get_logger(__name__)
        all_chunks = []

        for idx, row in df.iterrows():
            if method == 'words':
                chunks = self.chunk_by_words(
                    row['processed_text'],
                    row['web_id'],
                    row.get('title', ''),
                    row.get('url', ''),
                    row.get('kind', ''),
                    row.get('entities', ''),
                    row.get('topics', '')
                )
            elif method == 'sentences':
                chunks = self.chunk_by_sentences(
                    row['processed_text'],
                    row['web_id'],
                    row.get('title', ''),
                    row.get('url', ''),
                    row.get('kind', ''),
                    row.get('entities', ''),
                    row.get('topics', '')
                )
            else:
                raise ValueError(f"Unknown chunking method: {method}")

            all_chunks.extend(chunks)

        chunks_df = pd.DataFrame(all_chunks)
        logger.info(f"Создано {len(chunks_df)} чанков из {len(df)} документов (method={method})")
        logger.info(f"Среднее чанков/док: {len(chunks_df) / max(len(df),1):.1f}")

        # Разбивка по типу документа (kind) и базовой статистике размеров
        if 'kind' in chunks_df.columns:
            kind_counts = chunks_df['kind'].fillna('').value_counts().to_dict()
            logger.info(f"Распределение по kind: {kind_counts}")
        if 'chunk_index' in chunks_df.columns:
            logger.debug(f"Макс индекс чанка: {chunks_df['chunk_index'].max()}")

        return chunks_df


def create_chunks_from_documents(documents_df: pd.DataFrame,
                                 chunk_size: int = CHUNK_SIZE,
                                 chunk_overlap: int = CHUNK_OVERLAP,
                                 method: str = 'words') -> pd.DataFrame:
    """
    Создание чанков из DataFrame с документами

    Args:
        documents_df: DataFrame с предобработанными документами
        chunk_size: размер чанка
        chunk_overlap: перекрытие
        method: метод чанкинга

    Returns:
        DataFrame с чанками
    """
    chunker = DocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    logger = get_logger(__name__)
    with log_timing(logger, f"Чанкинг документов (method={method}, size={chunk_size}, overlap={chunk_overlap})"):
        chunks_df = chunker.chunk_documents_dataframe(documents_df, method=method)

    return chunks_df


if __name__ == "__main__":
    # Тестирование чанкинга
    test_doc = {
        'web_id': 1,
        'title': 'Тестовый документ',
        'processed_text': ' '.join(['Слово'] * 500)  # 500 слов
    }

    df = pd.DataFrame([test_doc])
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)

    chunks = chunker.chunk_documents_dataframe(df, method='words')
    print(f"\nСоздано чанков: {len(chunks)}")
    print(f"\nПервый чанк:\n{chunks.iloc[0]['text'][:200]}...")
