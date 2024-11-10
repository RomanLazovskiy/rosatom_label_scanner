import os
import logging
from dotenv import load_dotenv
from telegram import Update, BotCommand
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters
import asyncio
import nest_asyncio

from handlers.handlers import start, info, help_command, handle_image, handle_message, send_bad_images, button_handler

nest_asyncio.apply()

async def main():
    load_dotenv()
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    if not TOKEN:
        print("Ошибка: переменная окружения TELEGRAM_BOT_TOKEN не установлена")
        exit(1)

    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    application = ApplicationBuilder().token(TOKEN).build()

    commands = [
        BotCommand("start", "Начать работу с ботом"),
        BotCommand("info", "Информация о боте и его возможностях"),
        BotCommand("help", "Показать список команд"),
        BotCommand("bad_images", "Показать список неверно классифицированных изображений")
    ]
    await application.bot.set_my_commands(commands)

    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('info', info))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(CommandHandler('bad_images', send_bad_images))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO | filters.Document.MimeType("application/zip"), handle_image))

    application.add_handler(CallbackQueryHandler(button_handler))

    logger.info("Запуск Polling...")
    await application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == '__main__':
    asyncio.run(main())
