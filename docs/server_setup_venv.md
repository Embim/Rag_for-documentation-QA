## Настройка сервера (Python venv, Weaviate, CUDA)

Инструкция для **Ubuntu/Debian**. Для других Linux‑дистрибутивов команды аналогичны (названия пакетов и менеджер могут отличаться).

Рекомендуется выполнять шаги **последовательно сверху вниз** на чистой машине/сервере.

---

## 1) Установка системных зависимостей

Минимальный набор пакетов:

```bash
sudo apt update
sudo apt install -y \
  python3 python3-venv python3-pip \
  git \
  curl \
  docker.io docker-compose
```

- **python3 / venv / pip**: базовая Python‑среда.
- **git**: чтобы клонировать репозиторий.
- **docker / docker-compose**: для запуска Weaviate в контейнере.

> При необходимости добавьте `build-essential` и `cmake`, если планируется сборка из исходников:
> 
> ```bash
> sudo apt install -y build-essential cmake
> ```

---

## 2) Клонирование проекта

```bash
git clone <repo_url> alfa-rag
cd alfa-rag
```

- Замените `<repo_url>` на реальный URL Git‑репозитория (`git@...` или `https://...`).
- Папка проекта в примере называется `alfa-rag`, но можно использовать любое имя.

---

## 3) Python virtualenv и зависимости

Создаём и активируем виртуальное окружение, затем ставим зависимости.

```bash
# Создать виртуальное окружение в каталоге .venv
python3 -m venv .venv

# Активировать окружение (важно: делать это в каждом новом терминале)
source .venv/bin/activate

# (опционально) флаги для сборки CUDA-версии llama-cpp
export CMAKE_ARGS="-DLLAMA_CUBLAS=on"
export FORCE_CMAKE=1

# Обновляем pip и ставим базовые зависимости проекта
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.1) Установка PyTorch (CPU / CUDA)

Если у вас **есть GPU** и установлен CUDA‑драйвер, желательно поставить PyTorch с поддержкой CUDA.

- **Вариант 1: стандартный wheel с CUDA 12.1 (пример)**

```bash
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124
```

- **Вариант 2: другой CUDA wheel (пример для CUDA 12.4)**  
  (подставьте свою версию из документации PyTorch)

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 "torch>=2.6"
```

Если GPU нет или CUDA недоступен — можно установить CPU‑сборку (по умолчанию `pip install torch`), но инференс LLM будет заметно медленнее.

### 3.2) Библиотеки для LLM и векторного поиска

```bash
pip install \
  transformers accelerate bitsandbytes \
  sentence-transformers weaviate-client \
  pandas numpy tqdm
```

### 3.3) Установка `llama-cpp-python`

Если нужен **CUDA‑вариант** `llama-cpp-python`, используем дополнительный индекс:

```bash
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

Если пакет уже установлен, но требуется пересборка/переустановка, можно использовать:

```bash
python -m pip install --upgrade pip

python -m pip install \
  --force-reinstall \
  --no-cache-dir \
  --index-url https://pypi.org/simple \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
  "llama-cpp-python[server]"
```

Убедитесь, что перед этим активировано окружение: `source .venv/bin/activate`.

---

## 4) Docker и Weaviate

### 4.1) Установка Docker (если не поставили через apt)

Если Docker ещё не установлен или нужна последняя версия:

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
  -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

После установки можно проверить:

```bash
docker --version
docker-compose --version
```

Также имеет смысл добавить текущего пользователя в группу `docker`, чтобы не использовать `sudo` каждый раз:

```bash
sudo usermod -aG docker "$USER"
newgrp docker
```

### 4.2) Подъём Weaviate через docker-compose

В корне проекта должен лежать корректный `docker-compose.yml` с конфигурацией Weaviate.

```bash
cd /path/to/alfa-rag        # корень проекта
docker-compose up -d        # или sudo docker-compose up -d при необходимости
```

Проверьте доступность UI/endpoint Weaviate в браузере:

- `http://localhost:8080`

Если порт занят или сервер удалённый, убедитесь, что открыты нужные порты в firewall / security group.

---

## 5) Установка `huggingface_hub` и shim для `huggingface-cli`

Некоторые скрипты (например, `scripts/download_models.py`) ожидают наличие бинари `huggingface-cli`.  
Вместо глобальной установки можно сделать небольшой shim внутри venv.

### 5.1) Устанавливаем CLI из `huggingface_hub`

```bash
pip install "huggingface_hub[cli]"
```

### 5.2) Создаём shim `huggingface-cli` в `.venv/bin`

```bash
cat > .venv/bin/huggingface-cli << 'EOF'
#!/usr/bin/env bash
# Shim для совместимости с scripts/download_models.py

# Скрипт download_models.py вызывает:
#   huggingface-cli --version
# Нам важно просто вернуть код 0, чтобы проверка прошла.
if [[ "$1" == "--version" ]]; then
  echo "huggingface-cli shim (using huggingface_hub.commands.huggingface_cli)"
  exit 0
fi

# Все остальные команды пробрасываем в реальный CLI
python -m huggingface_hub.commands.huggingface_cli "$@"
EOF

chmod +x .venv/bin/huggingface-cli
```

После этого внутри активированного venv команда `huggingface-cli` будет доступна и корректно работать для остальных команд.

---

## 6) Переменные окружения (опционально)

Некоторые параметры можно передавать через переменные окружения:

```bash
export USE_WEAVIATE=true          # использовать Weaviate как векторное хранилище
export LOG_LEVEL=INFO             # уровень логирования (DEBUG/INFO/WARNING/ERROR)
export LOG_FILE=pipeline.log      # файл логов (будет создан в ./outputs)
```

Эти переменные можно добавить в `~/.bashrc` или отдельный `.env`/скрипт, который вы выполняете перед запуском.

---

## 7) Запуск пайплайна

# API режим (автоматически для reranking)
export LLM_MODE=api
export LLM_API_MODEL=tngtech/deepseek-r1t2-chimera:free
export OPENROUTER_API_KEY=sk-or-v1-...  # опционально

```bash
cd /path/to/alfa-rag
source .venv/bin/activate

# Перестроить базу знаний (быстро, без LLM)
python main_pipeline.py build --force

# Поиск ответов (пример: ограничить до 20 результатов)
python main_pipeline.py search --limit 20
```

Если используется Weaviate, убедитесь, что контейнер поднят (`docker-compose ps`) и переменная `USE_WEAVIATE=true` установлена.

---

## 8) Логи

- **Файл логов**: `outputs/pipeline.log` (ротация, кодировка UTF‑8).
- Для более подробной отладки установите:

```bash
export LOG_LEVEL=DEBUG
```

и затем запустите пайплайн снова.

---

## 9) Обновление проекта

```bash
cd /path/to/alfa-rag
git pull                      # подтянуть последние изменения

source .venv/bin/activate
pip install -r requirements.txt   # обновить зависимости, если они поменялись
```

При крупном обновлении/смене версий иногда полезно пересоздать venv (удалить `.venv` и повторить шаг 3).
