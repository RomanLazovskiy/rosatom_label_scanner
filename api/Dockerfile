# Использование официального образа Python 3.12 slim
FROM python:3.12-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Обновление пакетов и установка необходимых зависимостей
RUN apt-get update && apt-get install -y \
    git \  # Установка git для работы с репозиториями
    curl \  # Установка curl для скачивания скриптов и файлов
    libgl1-mesa-glx \  # Установка библиотеки для графических операций
    libglib2.0-0 \  # Установка библиотеки для графических операций
    && curl -sSL https://install.python-poetry.org | python3 - \  # Установка Poetry для управления зависимостями
    && apt-get clean \  # Очистка кэша apt для уменьшения размера контейнера
    && rm -rf /var/lib/apt/lists/*  # Удаление временных файлов

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
