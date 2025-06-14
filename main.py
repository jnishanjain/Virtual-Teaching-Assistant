# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
import base64
from app.qa_engine import answer_question

app = FastAPI()

class Query(BaseModel):
    question: str
    image: str = None

@app.post("/api/")
async def get_answer(query: Query):
    response = answer_question(query.question, query.image)
    return response
