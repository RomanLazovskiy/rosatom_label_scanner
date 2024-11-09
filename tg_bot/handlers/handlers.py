import aiohttp
import logging
import zipfile
import os
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ContextTypes
from dotenv import load_dotenv


TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["Start", "Info", "Help"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
    greeting = (
        "Добрый день! Это автоматический сканнер маркировки для компании РосАтом от команды Уральские мандарины. "
        "Я помогу вам в распознавании и классификации изображений или архивов с маркировками. "
        "Отправьте изображение или zip-архив, и я верну вам тестовую метку для каждого файла."
    )
    await update.message.reply_text(greeting, reply_markup=reply_markup)


async def info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    info_message = (
        "Я автоматический сканнер маркировки для компании РосАтом от команды Уральские мандарины. "
        "Моя задача — помочь вам в распознавании и классификации маркировок. "
        "Вы можете отправить изображение или zip-архив с маркировками, и я верну тестовую метку для каждого файла."
    )
    await update.message.reply_text(info_message)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_message = (
        "Список доступных команд:\n"
        "/start — Начать работу с ботом\n"
        "/info — Информация о боте и его возможностях\n"
        "/help — Показать список команд"
    )
    await update.message.reply_text(help_message)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    text = update.message.text.lower()
    logging.info(f"Получено сообщение от {user.username} (ID: {user.id}): {text}")

    if text == "start":
        await start(update, context)
    elif text == "info":
        await info(update, context)
    elif text == "help":
        await help_command(update, context)
    else:
        await update.message.reply_text("Неизвестная команда. Попробуйте /start, /info или /help.")


async def process_and_send_results(file_path, update):
    load_dotenv()
    url = os.getenv("API_URL", "http://api-container:8001/api/predict")
    if not url:
        logging.info("Ошибка: переменная окружения API_URL не установлена")
        exit(1)
    async with aiohttp.ClientSession() as session:
        with open(file_path, 'rb') as file:
            form_data = aiohttp.FormData()
            form_data.add_field("file", file, filename=os.path.basename(file_path))

            async with session.post(url, data=form_data) as response:
                if response.status == 200:
                    result = await response.json()
                    for prediction in result:
                        await update.message.reply_photo(
                            photo=open(file_path, 'rb'),
                            caption=f"Метка: {prediction['best_match']}"
                        )
                else:
                    await update.message.reply_text("Ошибка при обработке изображения на сервере.")


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    logging.info(f"Получено изображение от {user.username} (ID: {user.id})")

    if update.message.photo:
        photo = update.message.photo[-1]
        file = await photo.get_file()
        file_path = os.path.join(TEMP_DIR, f"{user.id}_photo.jpg")
        await file.download_to_drive(file_path)

        await process_and_send_results(file_path, update)
        os.remove(file_path)

    elif update.message.document and update.message.document.mime_type == "application/zip":
        logging.info("Начинается обработка zip-архива.")
        await update.message.reply_text("Начинаю обработку фото из архива")

        document = await update.message.document.get_file()
        zip_path = os.path.join(TEMP_DIR, f"{user.id}_archive.zip")
        await document.download_to_drive(zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(TEMP_DIR)
            logging.info(f"Извлечено содержимое zip-архива в папку {TEMP_DIR}")

            for filename in os.listdir(TEMP_DIR):
                file_path = os.path.join(TEMP_DIR, filename)
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    await process_and_send_results(file_path, update)
                    os.remove(file_path)

        os.remove(zip_path)
