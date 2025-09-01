from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "data"
DOCS, CHUNKS = [], []

for fname in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, fname)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                DOCS.append((fname, text))
                CHUNKS.append(text)

VECTORIZER = TfidfVectorizer(stop_words="english").fit(CHUNKS)
VECTORS = VECTORIZER.transform(CHUNKS)

app = FastAPI(title="LLM Demo v3")

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: List[str]

@app.post("/ai/answer", response_model=AskResponse)
def ai_answer(req: AskRequest):
    q_vec = VECTORIZER.transform([req.question])
    sims = cosine_similarity(q_vec, VECTORS)
    idx = sims.argmax()
    best_doc = DOCS[idx]

    return AskResponse(
        answer=best_doc[1],
        sources=[best_doc[0]]
    )
