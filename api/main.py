import logging
from fastapi import FastAPI
import uvicorn
from routes.predict import router as predict_router

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()
app.include_router(predict_router, prefix="/api")

@app.get("/")
def index():
    return {"text": "Детектор маркировки от команды Уральские Мандарины."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
