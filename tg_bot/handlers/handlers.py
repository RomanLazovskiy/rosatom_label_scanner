import aiohttp
import logging
import zipfile
import os
import pandas as pd
from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from dotenv import load_dotenv

TEMP_DIR = "temp_files"
BAD_IMAGES_FILE = "data/bad_images.csv"
os.makedirs(TEMP_DIR, exist_ok=True)

def add_bad_image(filename, label):
    df = pd.DataFrame([{"filename": filename, "label": label}])
    try:
        existing_df = pd.read_csv(BAD_IMAGES_FILE)
        df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        pass  # Если файла еще нет, создается новый DataFrame
    df.to_csv(BAD_IMAGES_FILE, index=False)

async def send_bad_images(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        with open(BAD_IMAGES_FILE, "rb") as file:
            await update.message.reply_document(file, filename="bad_images.csv")
    except FileNotFoundError:
        await update.message.reply_text("Файл с неверно распознанными изображениями пуст или не создан.")

async def send_image_with_buttons(file_path, caption, label, update):
    keyboard = [
        [InlineKeyboardButton("✅ Верно", callback_data=f"correct|{label}")],
        [InlineKeyboardButton("❌ Неверно", callback_data=f"incorrect|{os.path.basename(file_path)}|{label}")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_photo(photo=open(file_path, "rb"), caption=caption, reply_markup=reply_markup)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    action, *data = query.data.split("|")
    label = data[-1]

    if action == "incorrect":
        filename = data[0]
        add_bad_image(filename, label)
        updated_caption = f"Метка: ❌ {label} (обозначено как 'Неверно')"
    elif action == "correct":
        updated_caption = f"Метка: ✅ {label} (обозначено как 'Верно')"

    await query.edit_message_caption(caption=updated_caption)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["Start", "Info", "Help", "Bad Images"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
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
        "/help — Показать список команд\n"
        "/bad_images — Показать список неверно классифицированных изображений"
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
    elif text == "bad images":
        await send_bad_images(update, context)
    else:
        await update.message.reply_text("Неизвестная команда. Попробуйте Start, Info, Help или Bad Images.")

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
                        label = prediction['best_match']
                        caption = f"Метка: {label}"
                        await send_image_with_buttons(file_path, caption, label, update)
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
