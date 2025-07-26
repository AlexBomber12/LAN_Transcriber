FROM python:3.12-slim

# Рабочая директория в контейнере
WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip install --upgrade pip setuptools wheel

# Копируем весь исходный код проекта
COPY . .

# Обеспечиваем видимость модулей внутри /app
ENV PYTHONPATH=/app

# Переменная для версии (можно переопределить через ENV)
ENV TRANSCRIBER_VERSION=0.2.0

# Запуск основного файла
CMD ["python", "web_transcribe.py"]
