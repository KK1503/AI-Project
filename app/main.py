from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.chatbot import get_answer
import os

app = FastAPI()

# Serve the frontend static files from app/static
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


class Query(BaseModel):
    question: str


@app.get("/")
def home():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "FAQ Chatbot API Running"}


@app.post("/chat")
def chat(query: Query):
    answer = get_answer(query.question)
    return {"question": query.question, "answer": answer}