import torch
import textdistance
import logging


def device_detect():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# Функция для поиска наилучшего совпадения текста в базе данных
def find_best_match(text, database):
    if not isinstance(text, str):
        logging.info("Warning: text is not a string:", text)
        return '', 0  # Возвращаем пустой результат, если text не строка

    best_ratio = 0
    best_match = ''
    for entry in database:
        if not isinstance(entry, str):
            logging.info("Warning: entry is not a string:", entry)
            continue  # Пропускаем entry, если это не строка

        ratio = textdistance.levenshtein.normalized_similarity(text, entry)
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = entry
    return best_match, best_ratio
