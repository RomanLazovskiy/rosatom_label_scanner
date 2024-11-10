import aiohttp
import logging
import zipfile
import os
import pandas as pd
from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from telegram.helpers import escape_markdown
from dotenv import load_dotenv

TEMP_DIR = "temp_files"  # Папка для временных файлов
BAD_IMAGES_FILE = "data/bad_images.csv"  # Путь к файлу с неверно классифицированными изображениями
os.makedirs(TEMP_DIR, exist_ok=True)  # Создание папки для временных файлов, если она не существует


# Функция для добавления неверно классифицированных изображений в CSV
def add_bad_image(filename, label):
    """
    Функция для добавления информации о неверно классифицированном изображении в файл CSV.

    Параметры
    ----------
    filename : str
        Имя файла изображения, которое было классифицировано неверно.
    label : str
        Метка, с которой было классифицировано изображение.
    """
    df = pd.DataFrame([{"filename": filename, "label": label}])
    try:
        existing_df = pd.read_csv(BAD_IMAGES_FILE)
        df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        pass  # Если файла еще нет, создается новый DataFrame
    df.to_csv(BAD_IMAGES_FILE, index=False)


# Отправка файла с неверно распознанными изображениями
async def send_bad_images(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Асинхронная функция для отправки файла с неверно распознанными изображениями.

    Параметры
    ----------
    update : Update
        Объект, содержащий информацию о полученном сообщении.
    context : ContextTypes.DEFAULT_TYPE
        Контекст, содержащий данные о текущем состоянии обработки запроса.
    """
    try:
        with open(BAD_IMAGES_FILE, "rb") as file:
            await update.message.reply_document(file, filename="bad_images.csv")
    except FileNotFoundError:
        await update.message.reply_text("Файл с неверно распознанными изображениями пуст или не создан.")


# Отправка изображения с кнопками "Верно" и "Неверно"
async def send_image_with_buttons(file_path, caption, data, update, context):
    """
    Функция для отправки изображения с кнопками "Верно" и "Неверно".

    Параметры
    ----------
    file_path : str
        Путь к изображению, которое будет отправлено.
    caption : str
        Подпись, которая будет отображаться под изображением.
    data : dict
        Данные, которые будут использоваться для формирования метки.
    update : Update
        Объект, содержащий информацию о полученном сообщении.
    context : ContextTypes.DEFAULT_TYPE
        Контекст, содержащий данные о текущем состоянии обработки запроса.
    """
    # Сохраняем данные в context.user_data
    user_id = update.message.from_user.id
    context.user_data[user_id] = data  # Сохраняем все данные по ID пользователя

    # Создаем кнопки с коротким callback_data
    callback_correct = "correct"
    callback_incorrect = f"incorrect|{os.path.basename(file_path)}"

    keyboard = [
        [InlineKeyboardButton("✅ Верно", callback_data=callback_correct)],
        [InlineKeyboardButton("❌ Неверно", callback_data=callback_incorrect)]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_photo(photo=open(file_path, "rb"), caption=caption, reply_markup=reply_markup, parse_mode="Markdown")


# Обработчик нажатий кнопок
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Функция для обработки нажатий кнопок "Верно" и "Неверно".

    Параметры
    ----------
    update : Update
        Объект, содержащий информацию о полученном сообщении.
    context : ContextTypes.DEFAULT_TYPE
        Контекст, содержащий данные о текущем состоянии обработки запроса.
    """
    query = update.callback_query
    await query.answer()

    action, filename = query.data.split("|", 1) if "|" in query.data else (query.data, None)
    user_id = query.from_user.id
    data = context.user_data.get(user_id, {})  # Получаем сохраненные данные

    # Формирование обновленного caption с полной информацией
    updated_caption = (
        f"*Метка:* {escape_markdown(data.get('ДетальАртикул', 'N/A'))}\n"
        f"*Порядковый номер:* {escape_markdown(str(data.get('ПорядковыйНомер', 'N/A')))}\n"
        f"*Наименование детали:* {escape_markdown(data.get('ДетальНаименование', 'N/A'))}\n"
        f"*Номер заказа:* {escape_markdown(data.get('ЗаказНомер', 'N/A'))}\n"
        f"*Станция/Блок:* {escape_markdown(data.get('СтанцияБлок', 'N/A'))}\n"
    )

    if action == "incorrect" and filename:
        add_bad_image(filename, data.get('ДетальАртикул', 'N/A'))
        updated_caption += "*Результат:* ❌ (обозначено как 'Неверно')"
    elif action == "correct":
        updated_caption += "*Результат:* ✅ (обозначено как 'Верно')"

    await query.edit_message_caption(caption=updated_caption, parse_mode="Markdown")


# Обработчик команды start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обработчик команды "/start", отправляющий приветственное сообщение.

    Параметры
    ----------
    update : Update
        Объект, содержащий информацию о полученном сообщении.
    context : ContextTypes.DEFAULT_TYPE
        Контекст, содержащий данные о текущем состоянии обработки запроса.
    """
    keyboard = [["Start", "Info", "Help", "Bad Images"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    greeting = (
        "Добрый день! Это автоматический сканнер маркировки для компании РосАтом от команды Уральские мандарины. "
        "Я помогу вам в распознавании и классификации изображений или архивов с маркировками. "
        "Отправьте изображение или zip-архив, и я верну вам тестовую метку для каждого файла."
    )
    await update.message.reply_text(greeting, reply_markup=reply_markup)


# Обработчик команды info
async def info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обработчик команды "/info", отправляющий информацию о боте.

    Параметры
    ----------
    update : Update
        Объект, содержащий информацию о полученном сообщении.
    context : ContextTypes.DEFAULT_TYPE
        Контекст, содержащий данные о текущем состоянии обработки запроса.
    """
    info_message = (
        "Я автоматический сканнер маркировки для компании РосАтом от команды Уральские мандарины. "
        "Моя задача — помочь вам в распознавании и классификации маркировок. "
        "Вы можете отправить изображение или zip-архив с маркировками, и я верну тестовую метку для каждого файла."
    )
    await update.message.reply_text(info_message)


# Обработчик команды help
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обработчик команды "/help", отправляющий список команд.

    Параметры
    ----------
    update : Update
        Объект, содержащий информацию о полученном сообщении.
    context : ContextTypes.DEFAULT_TYPE
        Контекст, содержащий данные о текущем состоянии обработки запроса.
    """
    help_message = (
        "Список доступных команд:\n"
        "/start — Начать работу с ботом\n"
        "/info — Информация о боте и его возможностях\n"
        "/help — Показать список команд\n"
        "/bad_images — Показать список неверно классифицированных изображений"
    )
    await update.message.reply_text(help_message)


# Обработка входящих сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обработчик входящих сообщений и команд.

    Параметры
    ----------
    update : Update
        Объект, содержащий информацию о полученном сообщении.
    context : ContextTypes.DEFAULT_TYPE
        Контекст, содержащий данные о текущем состоянии обработки запроса.
    """
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


# Функция для обработки изображений и отправки результатов
async def process_and_send_results(file_path, update, context):
    """
    Функция для обработки изображения и отправки результатов классификации.

    Параметры
    ----------
    file_path : str
        Путь к файлу изображения для отправки.
    update : Update
        Объект, содержащий информацию о полученном сообщении.
    context : ContextTypes.DEFAULT_TYPE
        Контекст, содержащий данные о текущем состоянии обработки запроса.
    """
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
                        best_match_info = prediction['best_match']

                        # Формирование сообщения с основной меткой и дополнительной информацией
                        caption = (
                            f"*Метка:* {escape_markdown(best_match_info['ДетальАртикул'])}\n"
                            f"*Порядковый номер:* {escape_markdown(str(best_match_info.get('ПорядковыйНомер', 'N/A')))}\n"
                            f"*Наименование детали:* {escape_markdown(best_match_info.get('ДетальНаименование', 'N/A'))}\n"
                            f"*Номер заказа:* {escape_markdown(best_match_info.get('ЗаказНомер', 'N/A'))}\n"
                            f"*Станция/Блок:* {escape_markdown(best_match_info.get('СтанцияБлок', 'N/A'))}\n"
                            f"*Распознанный текст:* {escape_markdown(prediction['extracted_text'])}"
                        )

                        # Сохраняем все данные в context.user_data
                        user_id = update.message.from_user.id
                        context.user_data[user_id] = best_match_info

                        # Отправка изображения с кнопками
                        await send_image_with_buttons(file_path, caption, best_match_info, update, context)
                else:
                    await update.message.reply_text("Ошибка при обработке изображения на сервере.")


# Обработчик для изображений
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обработчик для входящих изображений и архивов. Принимает изображения и архивы, извлекает изображения из архива и обрабатывает их.

    Параметры
    ----------
    update : Update
        Объект, содержащий информацию о полученном сообщении.
    context : ContextTypes.DEFAULT_TYPE
        Контекст, содержащий данные о текущем состоянии обработки запроса.
    """
    user = update.effective_user
    logging.info(f"Получено изображение от {user.username} (ID: {user.id})")

    # Если сообщение содержит фото
    if update.message.photo:
        photo = update.message.photo[-1]  # Получаем самое большое фото
        file = await photo.get_file()  # Получаем файл изображения
        file_path = os.path.join(TEMP_DIR, f"{user.id}_photo.jpg")  # Путь для сохранения изображения
        await file.download_to_drive(file_path)  # Загружаем фото в папку

        await process_and_send_results(file_path, update, context)  # Обрабатываем изображение и отправляем результат
        os.remove(file_path)  # Удаляем изображение после обработки

    # Если сообщение содержит zip-архив
    elif update.message.document and update.message.document.mime_type == "application/zip":
        logging.info("Начинается обработка zip-архива.")
        await update.message.reply_text("Начинаю обработку фото из архива")

        # Загружаем архив
        document = await update.message.document.get_file()
        zip_path = os.path.join(TEMP_DIR, f"{user.id}_archive.zip")
        await document.download_to_drive(zip_path)

        # Извлекаем файлы из архива
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(TEMP_DIR)
            logging.info(f"Извлечено содержимое zip-архива в папку {TEMP_DIR}")

            # Обрабатываем каждое изображение из архива
            for filename in os.listdir(TEMP_DIR):
                file_path = os.path.join(TEMP_DIR, filename)
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    await process_and_send_results(file_path, update, context)
                    os.remove(file_path)

        os.remove(zip_path)  # Удаляем архив после обработки
