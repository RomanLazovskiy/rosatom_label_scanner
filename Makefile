# Переменные для API
API_IMAGE_NAME = rosatom-label-scanner-api
API_CONTAINER_NAME = api-container
API_PORT = 8001

# Переменные для TG Bot
BOT_IMAGE_NAME = rosatom-label-scanner-tg-bot
BOT_CONTAINER_NAME = tg-bot-container


# Скачивание файлов с Google Диска в директорию api/models и api/data
download-data:
	@echo "Создание директорий ./api/data и ./api/models..."
	@mkdir -p ./api/data
	@mkdir -p ./api/models
	@echo "Скачивание 1.JPG в ./api/data..."
	@gdown --id * -O ./api/data/1.JPG
	@echo "Скачивание database.xlsx в ./api/data..."
	@gdown --id * -O ./api/data/database.xlsx
	@echo "Скачивание models.zip в ./api/models..."
	@gdown --id * -O ./api/models/models.zip
	@echo "Распаковка models.zip в ./api/models..."
	@unzip -o ./api/models/models.zip -d ./api/models/
	@echo "Удаление архива models.zip..."
	@rm ./api/models/models.zip
	@echo "Все файлы успешно загружены и распакованы."

# Запуск всех сервисов с помощью docker-compose
start-all-project:
	@echo "Скачивание данных..."
	@$(MAKE) download-data
	@echo "Запуск всех сервисов с помощью docker-compose..."
	docker-compose up -d --build

# Остановка всех сервисов
stop-all-project:
	@echo "Остановка всех сервисов с помощью docker-compose..."
	docker-compose down

# Сборка Docker-образа для API
build-api:
	@echo "Сборка Docker-образа $(API_IMAGE_NAME)..."
	docker build -t $(API_IMAGE_NAME) -f ./api/Dockerfile ./api

# Запуск Docker-контейнера для API
run-api:
	@echo "Остановка и удаление старого контейнера (если существует)..."
	@docker stop $(API_CONTAINER_NAME) || true
	@docker rm $(API_CONTAINER_NAME) || true
	@echo "Запуск нового контейнера $(API_CONTAINER_NAME)..."
	docker run -d --name $(API_CONTAINER_NAME) -p $(API_PORT):$(API_PORT) $(API_IMAGE_NAME)

# Перезапуск Docker-контейнера для API
restart-api: stop-api run-api

# Остановка Docker-контейнера для API
stop-api:
	@echo "Остановка Docker-контейнера $(API_CONTAINER_NAME)..."
	@docker stop $(API_CONTAINER_NAME) || true
	@docker rm $(API_CONTAINER_NAME) || true

# Удаление Docker-образа для API
clean-api:
	@echo "Удаление Docker-образа $(API_IMAGE_NAME)..."
	@docker rmi $(API_IMAGE_NAME) || true

# Логи контейнера для API
logs-api:
	@docker logs -f $(API_CONTAINER_NAME)

# Вход в контейнер API для отладки
shell-api:
	@docker exec -it $(API_CONTAINER_NAME) /bin/bash

# Сборка Docker-образа для TG Bot
build-bot:
	@echo "Сборка Docker-образа $(BOT_IMAGE_NAME)..."
	docker build -t $(BOT_IMAGE_NAME) -f ./tg_bot/Dockerfile ./tg_bot

# Запуск Docker-контейнера для TG Bot
run-bot:
	@echo "Остановка и удаление старого контейнера (если существует)..."
	@docker stop $(BOT_CONTAINER_NAME) || true
	@docker rm $(BOT_CONTAINER_NAME) || true
	@echo "Запуск нового контейнера $(BOT_CONTAINER_NAME)..."
	docker run -d --name $(BOT_CONTAINER_NAME) $(BOT_IMAGE_NAME)

# Перезапуск Docker-контейнера для TG Bot
restart-bot: stop-bot run-bot

# Остановка Docker-контейнера для TG Bot
stop-bot:
	@echo "Остановка Docker-контейнера $(BOT_CONTAINER_NAME)..."
	@docker stop $(BOT_CONTAINER_NAME) || true
	@docker rm $(BOT_CONTAINER_NAME) || true

# Удаление Docker-образа для TG Bot
clean-bot:
	@echo "Удаление Docker-образа $(BOT_IMAGE_NAME)..."
	@docker rmi $(BOT_IMAGE_NAME) || true

# Логи контейнера для TG Bot
logs-bot:
	@docker logs -f $(BOT_CONTAINER_NAME)

# Вход в контейнер TG Bot для отладки
shell-bot:
	@docker exec -it $(BOT_CONTAINER_NAME) /bin/bash

# Запуск TG Bot без Docker (локальный запуск)
run-bot-local:
	@echo "Запуск бота локально..."
	@poetry run python ./tg_bot/main.py


# Список доступных команд
help:
	@echo "Доступные команды:"
	@echo "  download-data     - Скачивание файлов с Google Диска для API"
	@echo "  start-all-project  - Скачивание данных и запуск всех сервисов"
	@echo "  stop-all-project   - Остановка всех сервисов"
	@echo "  build-api         - Сборка Docker-образа для API"
	@echo "  run-api           - Запуск Docker-контейнера для API"
	@echo "  restart-api       - Перезапуск Docker-контейнера для API"
	@echo "  stop-api          - Остановка и удаление Docker-контейнера для API"
	@echo "  clean-api         - Удаление Docker-образа для API"
	@echo "  logs-api          - Просмотр логов контейнера для API"
	@echo "  shell-api         - Вход в контейнер для API"
	@echo ""
	@echo "  build-bot         - Сборка Docker-образа для TG Bot"
	@echo "  run-bot           - Запуск Docker-контейнера для TG Bot"
	@echo "  run-bot-local     - Запуск TG Bot локально"
	@echo "  restart-bot       - Перезапуск Docker-контейнера для TG Bot"
	@echo "  stop-bot          - Остановка и удаление Docker-контейнера для TG Bot"
	@echo "  clean-bot         - Удаление Docker-образа для TG Bot"
	@echo "  logs-bot          - Просмотр логов контейнера для TG Bot"
	@echo "  shell-bot         - Вход в контейнер для TG Bot"