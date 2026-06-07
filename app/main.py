from fastapi import FastAPI
from pydantic import BaseModel
from app.chatbot import get_answer

app = FastAPI()


class Query(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "FAQ Chatbot API Running"}


@app.post("/chat")
def chat(query: Query):
    answer = get_answer(query.question)
    return {"question": query.question, "answer": answer}