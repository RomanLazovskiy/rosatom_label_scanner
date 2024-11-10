# Использование официального образа Python 3.12 slim
FROM python:3.12-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Обновление пакетов и установка необходимых зависимостей
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Добавление Poetry в PATH, чтобы его можно было использовать
ENV PATH="/root/.local/bin:$PATH"
# Установка PYTHONPATH для корректной работы с импортами в проекте
ENV PYTHONPATH="/app:/app/scripts:/app/routes"

# Настройка Poetry для использования глобального окружения, без создания виртуальныхenv
RUN poetry config virtualenvs.create false

# Копирование файлов pyproject.toml и poetry.lock для установки зависимостей
COPY pyproject.toml poetry.lock ./

# Установка зависимостей с помощью Poetry (без установки самого проекта)
RUN poetry install --no-root --only main

# Копирование всех файлов проекта в контейнер
COPY . .

# Команда для запуска основного скрипта при старте контейнера
CMD ["python", "main.py"]
