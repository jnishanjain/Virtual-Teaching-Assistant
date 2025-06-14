#!/usr/bin/env python3
import os, json, faiss, numpy as np
from typing import List, Dict
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, PreTrainedTokenizerBase
from functools import lru_cache

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DISCOURSE_JSON = "data/discourse_data.json"
PAGES_JSON     = "data/course_content.json"

EMBED_MODEL    = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
GEN_MODEL = "google/flan-t5-small"
TOP_K          = 1

# Setup cache directory
cache_dir = os.environ.get("TRANSFORMERS_CACHE", "/tmp/.cache")
os.makedirs(cache_dir, exist_ok=True)
for var in ("TRANSFORMERS_CACHE", "XDG_CACHE_HOME", "HF_HOME"):
    os.environ[var] = cache_dir

# ─── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "🟢 OK"}

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <head><title>Virtual TA</title></head>
        <body>
            <h1>Welcome to Virtual TA 🤖</h1>
            <p>Use the <code>/api/</code> endpoint to POST your questions.</p>
        </body>
    </html>
    """
@app.middleware("http")
async def log_request(request: Request, call_next):
    body = await request.body()
    print("➡️ BODY RECEIVED:", body.decode("utf-8"))
    return await call_next(request)

# ─── Models & Data ─────────────────────────────────────────────────────────────

def load_docs() -> List[Dict]:
    docs = []
    for fn, base_url in ((DISCOURSE_JSON, None), (PAGES_JSON, "https://tds.s-anand.net/#/")):
        if not os.path.exists(fn):
            continue
        with open(fn, encoding="utf-8") as f:
            for rec in json.load(f):
                url = rec["url"] if base_url is None else base_url + rec["page"]
                title = rec.get("title", rec.get("page", url))
                docs.append({"text": rec["text"], "meta": {"url": url, "title": title}})
    return docs

def truncate_prompt(prompt: str, tokenizer: PreTrainedTokenizerBase, max_tokens: int = 256) -> str:
    tokens = tokenizer(prompt, truncation=True, max_length=max_tokens, return_tensors="pt")
    return tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)

print("📦 Loading embedding model:", EMBED_MODEL)
embedder = SentenceTransformer(EMBED_MODEL)

DOCS = load_docs()
if not DOCS:
    raise RuntimeError("❌ No documents loaded. Check your JSON files in the data/ folder.")

texts = [d["text"] for d in DOCS]
embs  = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
dim   = embs.shape[1]
INDEX = faiss.IndexFlatIP(dim)
INDEX.add(embs)
print(f"🔍 Indexed {len(DOCS)} docs with {EMBED_MODEL}")

print("🧠 Loading generation model:", GEN_MODEL)
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
model     = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
gen_pipe  = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)

# ─── API Schemas ───────────────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str

class Link(BaseModel):
    url: str
    text: str

class QuestionResponse(BaseModel):
    answer: str
    links: List[Link]

# ─── QA Logic ──────────────────────────────────────────────────────────────────
def retrieve(question: str, k=TOP_K):
    q_emb = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    D, I = INDEX.search(q_emb, k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        hits.append({"text": DOCS[idx]["text"], "meta": DOCS[idx]["meta"], "score": float(score)})
    return hits

@lru_cache(maxsize=128)
def generate_answer(prompt: str) -> str:
    out = gen_pipe(prompt, max_new_tokens=128, do_sample=False)
    return out[0]["generated_text"].strip()

@app.post("/api/", response_model=QuestionResponse)
def answer(req: QuestionRequest):
    hits = retrieve(req.question)

    if not hits:
        return QuestionResponse(answer="No relevant content found.", links=[])

    context = f"[{hits[0]['meta']['title']}]({hits[0]['meta']['url']})\n{hits[0]['text']}"
    prompt = (
        "Answer the question using the context below. Be concise.\n\n"
        f"{context}\n\nQUESTION: {req.question}\nANSWER:"
    )
    prompt = truncate_prompt(prompt, tokenizer)

    try:
        answer_text = generate_answer(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

    links = [Link(url=hits[0]["meta"]["url"], text=hits[0]["meta"]["title"])]
    return QuestionResponse(answer=answer_text, links=links)