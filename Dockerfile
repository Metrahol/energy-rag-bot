FROM python:3.11-slim

WORKDIR /app

# Системные зависимости для сборки и работы с PDF/C++ (chromadb)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем проект
COPY . .

# Переменные окружения по умолчанию
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    HF_HUB_OFFLINE=1

# Если индекс уже собран в storage/, бот сразу запускается.
# Если нет — при старте сначала выполнится ingest.
CMD ["sh", "-c", "if [ ! -f storage/chunks.jsonl ]; then python -m app.ingest; fi && python -m app.main"]
