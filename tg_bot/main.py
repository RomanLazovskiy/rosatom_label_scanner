import os
import logging
from dotenv import load_dotenv
from telegram import Update, BotCommand
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
import asyncio
import nest_asyncio

from handlers.handlers import start, info, help_command, handle_image, handle_message

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
        BotCommand("help", "Показать список команд")
    ]
    await application.bot.set_my_commands(commands)

    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('info', info))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO | filters.Document.MimeType("application/zip"), handle_image))

    logger.info("Запуск Polling...")
    await application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == '__main__':
    asyncio.run(main())
