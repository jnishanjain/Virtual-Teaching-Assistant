from sentence_transformers import SentenceTransformer
import faiss
import json

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("data/index.json", "r") as f:
    index_data = json.load(f)
    texts = [item["text"] for item in index_data]
    urls = [item["url"] for item in index_data]
    embeddings = model.encode(texts, convert_to_tensor=False)

index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(embeddings)


def search_docs(query, top_k=3):
    query_embedding = model.encode([query])[0]
    D, I = index.search([query_embedding], top_k)
    return [{"text": texts[i], "url": urls[i]} for i in I[0]]
