# Настройка сервера (venv, Weaviate)

Инструкция для Ubuntu/Debian (аналогично для других систем).

## 1) Зависимости ОС
```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git docker.io docker-compose
```

## 2) Клонирование проекта
```bash
git clone <repo_url> alfa-rag
cd alfa-rag
```

## 3) Python virtualenv
```bash
python3 -m venv .venv
source .venv/bin/activate
export CMAKE_ARGS="-DLLAMA_CUBLAS=on"
export FORCE_CMAKE=1

pip install --upgrade pip
pip install -r requirements.txt

# Если нужен CUDA билд torch, установите подходящий колёсник:
# пример для CUDA 12.4:
# pip install --index-url https://download.pytorch.org/whl/cu124 "torch>=2.6"
# Python пакеты
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate bitsandbytes sentence-transformers weaviate-client pandas numpy tqdm
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# Docker (для Weaviate)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

```
python -m pip install --upgrade pip

python -m pip install \
  --force-reinstall \
  --no-cache-dir \
  --index-url https://pypi.org/simple \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
  "llama-cpp-python[server]"

## 4) Weaviate
Поднимите Weaviate локально (пример docker-compose.yml должен быть в корне или добавьте свой):
```bash
sudo docker-compose up -d
```
Проверьте: http://localhost:8080
pip install "huggingface_hub[cli]"

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

## 5) Переменные окружения (опционально)
```bash
export USE_WEAVIATE=true
export LOG_LEVEL=INFO
export LOG_FILE=pipeline.log   # будет создан в ./outputs
```

## 6) Запуск
```bash
# Build базы знаний (быстро, без LLM)
python main_pipeline.py build --force

# Поиск ответов
python main_pipeline.py search --limit 20
```

## 7) Логи
- Файл логов: `outputs/pipeline.log` (ротация, UTF-8).
- Для подробной отладки: `export LOG_LEVEL=DEBUG` перед запуском.

## 8) Обновление
```bash
git pull
source .venv/bin/activate
pip install -r requirements.txt
```


