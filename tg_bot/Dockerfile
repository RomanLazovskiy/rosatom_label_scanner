# Использование официального образа Python 3.12 slim
FROM python:3.12-slim

# Установка переменной окружения PYTHONPATH для указания пути к приложению
ENV PYTHONPATH="/app"
# Установка переменной окружения PATH для включения Poetry в систему
ENV PATH="/root/.local/bin:$PATH"

# Обновление пакетов, установка curl и Poetry, очистка кэша
RUN apt-get update && apt-get install -y curl \  # Установка curl для скачивания данных
    && curl -sSL https://install.python-poetry.org | python3 - \  # Установка Poetry для управления зависимостями
    && apt-get clean \  # Очистка кэша apt для уменьшения размера контейнера
    && rm -rf /var/lib/apt/lists/*  # Удаление временных файлов, чтобы уменьшить размер контейнера

# Устанавливаем рабочую директорию в контейнере
WORKDIR /app

# Копирование файлов pyproject.toml и poetry.lock для установки зависимостей
COPY pyproject.toml poetry.lock ./

# Настройка Poetry для использования глобального окружения, без создания виртуальных env
RUN poetry config virtualenvs.create false && poetry install --no-root

# Копирование всех файлов проекта в контейнер
COPY . /app

# Копирование файла .env в контейнер
COPY .env /app/.env

# Команда для запуска основного скрипта при старте контейнера
CMD ["python", "main.py"]
