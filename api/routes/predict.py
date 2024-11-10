from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
import os
import shutil
import zipfile
import pandas as pd
from typing import List, Dict, Any
import torch

from scripts.inference import get_inference_class

router = APIRouter()

# Изменение модели `PredictionResponse`
class PredictionResponse(BaseModel):
    filename: str
    best_match: Dict[str, Any]  # Изменено для хранения словаря с данными
    extracted_text: str

# Пути к файлам базы данных, модели и ротационной модели
db_path = 'data/database.xlsx'
yolo_model_path = 'models/best.pt'
rotation_model_path = 'models/clip_rotation_classifier.pth'
efficient_ocr_model_path = 'microsoft/trocr-large-stage1'

# Проверка наличия файлов базы данных и моделей
if not os.path.exists(db_path):
    raise FileNotFoundError(f"Database file not found at path: {db_path}")
if not os.path.exists(yolo_model_path):
    raise FileNotFoundError(f"YOLO model file not found at path: {yolo_model_path}")
if not os.path.exists(rotation_model_path):
    raise FileNotFoundError(f"Rotation model file not found at path: {rotation_model_path}")

# Загрузка базы данных
db = pd.read_excel(db_path)

# Инициализация класса для инференса
recognitor = get_inference_class(
    yolo_model_path=yolo_model_path,
    db_excel_path=db_path,
    path_to_rotate_model=rotation_model_path,
    path_to_efficient_ocr_model=efficient_ocr_model_path,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

@router.post("/predict", response_model=List[PredictionResponse])
async def predict(file: UploadFile = File(...)):
    temp_dir = "temp_upload"
    os.makedirs(temp_dir, exist_ok=True)
    predictions = []

    try:
        # Сохранение загруженного файла
        file_location = os.path.join(temp_dir, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Распаковка zip-архива или проверка на поддерживаемый формат изображения
        if file.filename.lower().endswith(".zip"):
            with zipfile.ZipFile(file_location, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            os.remove(file_location)
        elif file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
            pass
        else:
            raise HTTPException(status_code=415, detail="Unsupported file type. Please upload an image or a zip file containing images.")

        # Получение списка изображений
        images_list = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not images_list:
            raise HTTPException(status_code=400, detail="No images found in the uploaded file.")

        # Проведение инференса
        results = recognitor.recognition(images_list, postprocessing_image=True)
        print(f"Тип results: {type(results)}, Пример данных results: {results[:5]}")

        # Формирование ответа с результатами
        for img_path, ((best_match, best_info), extracted_text) in zip(images_list, results):
            filename = os.path.basename(img_path)
            # best_info переводится в словарь для добавления полной информации
            best_match_data = {
                "ДетальАртикул": best_info['ДетальАртикул'],
                "ПорядковыйНомер": best_info['ПорядковыйНомер'],
                "ДетальНаименование": best_info['ДетальНаименование'],
                "ЗаказНомер": best_info['ЗаказНомер'],
                "СтанцияБлок": best_info['СтанцияБлок']
            }
            predictions.append(PredictionResponse(
                filename=filename,
                best_match=best_match_data,  # Полные данные `best_match`
                extracted_text=extracted_text
            ))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Очистка временной директории
        shutil.rmtree(temp_dir, ignore_errors=True)

    return predictions
