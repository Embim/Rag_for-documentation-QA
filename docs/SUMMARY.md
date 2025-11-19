# Архитектурные улучшения пайплайна

## Модульная структура выполнения

### Разделение на независимые этапы

Система поддерживает три режима работы для оптимизации процесса разработки и эксплуатации.

#### Режим 1: Построение индекса
```bash
python main_pipeline.py build
```
Выполняет индексацию документов в Weaviate (векторный + BM25 индексы). Метаданные сегментов сохраняются локально в `data/processed/chunks.pkl`. Требует выполнения однократно или при обновлении корпуса документов.

#### Режим 2: Обработка запросов
```bash
python main_pipeline.py search
python main_pipeline.py search --limit 10
```
Загружает предварительно построенный индекс и выполняет поиск. Поддерживает многократное выполнение без повторной индексации.

#### Режим 3: Полный цикл
```bash
python main_pipeline.py all
python main_pipeline.py all --force --limit 50
```
Последовательное выполнение построения индекса и обработки запросов.

## Преимущества модульного подхода

### Исходная архитектура
```bash
python main_pipeline.py  # Полный цикл: индексация + поиск
```
- Требуется полный цикл при каждом запуске
- Невозможность итеративной разработки
- Отсутствие гибкости в тестировании

### Модульная архитектура
```bash
# Однократная индексация
python main_pipeline.py build

# Итеративная обработка запросов
python main_pipeline.py search --limit 10   # Тестовая выборка
python main_pipeline.py search --limit 100  # Валидация
python main_pipeline.py search              # Полный набор
```
- Независимость этапов индексации и поиска
- Возможность итеративной настройки
- Гибкое тестирование на подвыборках

## Сценарии использования

### Инициализация системы
```bash
python main_pipeline.py build
python main_pipeline.py search
```

### Разработка и настройка
```bash
# Индекс построен однократно
python main_pipeline.py search --limit 10
# Модификация параметров в config.py
python main_pipeline.py search --limit 10
# Итеративная настройка
```

### Обновление корпуса документов
```bash
python main_pipeline.py build --force
python main_pipeline.py search
```

### Эксплуатация
```bash
# Индекс актуален
python main_pipeline.py search
```

## Технические возможности

### Автоматическое использование GPU
```
[GPU] Используется: NVIDIA ...
      CUDA версия: 12.1
      Batch size: 128
```

### Поддержка Weaviate
Интеграция с векторной базой данных для гибридного поиска:
```bash
docker-compose up -d
python main_pipeline.py build
python main_pipeline.py search
```

### Дополнительная документация
- `docs/commands.md` - справочник команд
- `docs/process.md` - описание процессов
- `docs/modules.md` - архитектура модулей
- `docs/server_setup_venv.md` - настройка окружения

## Рабочие процессы

### Процесс разработки

```bash
# Инициализация
python main_pipeline.py build

# Итеративная настройка
# 1. Модификация config.py
# 2. Тестирование
python main_pipeline.py search --limit 10
# 3. Валидация результатов
# 4. Повторение цикла

# Финальная обработка
python main_pipeline.py search
```

### Быстрое тестирование

```bash
python main_pipeline.py search --limit 5
```

### Продакшн развертывание

```bash
# Проверка актуальности индекса
python main_pipeline.py build --force  # при необходимости

# Обработка полного набора
python main_pipeline.py search
```

## Дополнительные ресурсы

Подробная документация:
- `docs/project.md` - архитектура системы
- `docs/OPTIMIZATION_GUIDE.md` - руководство по оптимизации

## Итоги

Модульная архитектура обеспечивает:
- Независимость этапов индексации и поиска
- Возможность итеративной настройки
- Гибкое тестирование на подвыборках
- Поддержку GPU-ускорения
- Интеграцию с Weaviate

---

Базовое использование:
```bash
python main_pipeline.py build
python main_pipeline.py search --limit 10
```

