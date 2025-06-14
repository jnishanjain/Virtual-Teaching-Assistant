#!/usr/bin/env python3
import os, json, time, faiss, numpy as np
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DISCOURSE_JSON = "data/discourse_data.json"
PAGES_JSON     = "data/course_content.json"

EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL      = "google/flan-t5-small"
TOP_K          = 5
# ────────────────────────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str

class Link(BaseModel):
    url: str
    text: str

class QuestionResponse(BaseModel):
    answer: str
    links: List[Link]

app = FastAPI()

# ─── Load & index documents ─────────────────────────────────────────────────────

def load_docs() -> List[Dict]:
    docs = []
    for fn, base_url in ((DISCOURSE_JSON, None), (PAGES_JSON, "https://tds.s-anand.net/#/")):
        with open(fn, encoding="utf-8") as f:
            for rec in json.load(f):
                url = rec["url"] if base_url is None else base_url + rec["page"]
                title = rec.get("title", rec.get("page", url))
                docs.append({"text": rec["text"], "meta": {"url": url, "title": title}})
    return docs

# Initialize embedding model
embedder = SentenceTransformer(EMBED_MODEL)

# Build FAISS index
DOCS = load_docs()
texts = [d["text"] for d in DOCS]
embs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
dim  = embs.shape[1]
INDEX = faiss.IndexFlatIP(dim)
INDEX.add(embs)
print(f"🔍 Indexed {len(DOCS)} docs with {EMBED_MODEL}")

# Initialize generation pipeline
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
model     = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
gen_pipe  = pipeline(
    "text2text-generation",
    model=model, tokenizer=tokenizer,
    device_map="auto" if model.device.type!="cpu" else None,
)

# ─── Retrieval + QA ────────────────────────────────────────────────────────────

def retrieve(question: str, k=TOP_K):
    q_emb = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    D, I = INDEX.search(q_emb, k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        hits.append({"text": DOCS[idx]["text"], "meta": DOCS[idx]["meta"], "score": float(score)})
    return hits

@app.post("/api/", response_model=QuestionResponse)
def answer(req: QuestionRequest):
    # 1) Retrieve top‐K
    hits = retrieve(req.question)

    # 2) Build prompt for T5
    context = "\n\n".join(
        f"[{h['meta']['title']}]({h['meta']['url']})\n{h['text']}"
        for h in hits
    )
    prompt = (
        "Answer the question using the context below. Be concise.\n\n"
        f"{context}\n\nQUESTION: {req.question}\nANSWER:"
    )

    # 3) Generate
    out = gen_pipe(
        prompt,
        max_length=150,
        do_sample=False,
    )
    answer_text = out[0]["generated_text"].strip()

    # 4) Return with citations
    links = [Link(url=h["meta"]["url"], text=h["meta"]["title"]) for h in hits]
    return QuestionResponse(answer=answer_text, links=links)
