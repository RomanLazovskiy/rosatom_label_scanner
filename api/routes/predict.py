from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
import os
import shutil
import zipfile
import pandas as pd
from typing import List

from scripts.inference import MarkingOCR

router = APIRouter()

class PredictionResponse(BaseModel):
    filename: str
    best_match: str
    extracted_text: str

db_path = 'data/database.xlsx'
model_path = 'models/best.pt'

if not os.path.exists(db_path):
    raise FileNotFoundError(f"Database file not found at path: {db_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at path: {model_path}")

db = pd.read_excel(db_path)['ДетальАртикул'].drop_duplicates().tolist()

recognitor = MarkingOCR(yolo_model_path=model_path, db=db)

@router.post("/predict", response_model=List[PredictionResponse])
async def predict(file: UploadFile = File(...)):
    temp_dir = "temp_upload"
    os.makedirs(temp_dir, exist_ok=True)
    predictions = []

    try:
        file_location = os.path.join(temp_dir, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if file.filename.lower().endswith(".zip"):
            with zipfile.ZipFile(file_location, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            os.remove(file_location)
        elif file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
            pass
        else:
            raise HTTPException(status_code=415, detail="Unsupported file type. Please upload an image or a zip file containing images.")

        images_list = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not images_list:
            raise HTTPException(status_code=400, detail="No images found in the uploaded file.")

        results = recognitor.recognition(images_list, postprocessing_image=True)
        print(f"Тип results: {type(results)}, Пример данных results: {results[:5]}")

        for img_path, (best_match, extracted_text) in zip(images_list, results):
            filename = os.path.basename(img_path)
            predictions.append(PredictionResponse(
                filename=filename,
                best_match=best_match,
                extracted_text=extracted_text
            ))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return predictions
