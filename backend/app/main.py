from fastapi import FastAPI
from backend.app.routes import chat 
from fastapi.responses import FileResponse
import os

app = FastAPI(title="RAG Chatbot Backend")
UPLOAD_FOLDER = "static/uploads"

@app.get("/files/{filename}")
async def get_file(filename: str):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    return FileResponse(file_path)

# Подключаем маршруты
app.include_router(chat.router)

@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running"}
