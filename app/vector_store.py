import os
import faiss
import numpy as np
from app.embeddings import generate_embedding
from PyPDF2 import PdfReader
import re
import math
import openai

# Path to the PDF file (update if needed)
PDF_PATH = r"C:\Users\shail\Downloads\FAQ.pdf"


def load_pdf_pages(pdf_path=PDF_PATH):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({"page": i + 1, "text": text})

    return pages


def chunk_text(text, chunk_size=500, overlap=100):
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


# Load PDF and create chunks (passages)
pages = load_pdf_pages()
passages = []
for p in pages:
    chunks = chunk_text(p["text"], chunk_size=800, overlap=150)
    for i, c in enumerate(chunks):
        passages.append({"page": p["page"], "chunk_index": i, "text": c})


# Generate embeddings for passages
if len(passages) == 0:
    raise RuntimeError("No text extracted from PDF to index.")

doc_embeddings = np.array([generate_embedding(p["text"]) for p in passages]).astype("float32")

# Create FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)


def search_faq(user_query, top_k=1, snippet_max=1000):
    """Return the best-matching passage (snippet) and metadata.

    Args:
        user_query (str): the user's question
        top_k (int): number of results to fetch from FAISS
        snippet_max (int): max characters to return from passage

    Returns:
        dict: {"text": snippet, "page": int, "chunk_index": int, "distance": float}
    """
    query_embedding = np.array([generate_embedding(user_query)]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    best_idx = int(indices[0][0])
    best_distance = float(distances[0][0])

    passage = passages[best_idx]
    snippet = passage["text"]
    if len(snippet) > snippet_max:
        snippet = snippet[:snippet_max].rsplit(" ", 1)[0] + "..."

    return {"text": snippet, "page": passage["page"], "chunk_index": passage["chunk_index"], "distance": best_distance}


def search_top_k(user_query, top_k=3):
    """Return top_k matching passages with distances and indices."""
    query_embedding = np.array([generate_embedding(user_query)]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for d, idx in zip(distances[0], indices[0]):
        if int(idx) < 0:
            continue
        p = passages[int(idx)]
        results.append({"text": p["text"], "page": p["page"], "chunk_index": p["chunk_index"], "distance": float(d), "index": int(idx)})
    return results


def _cosine(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def rag_answer(user_query, top_k=3):
    """Retrieve top-k passages and pick the best sentence as concise answer using embedding similarity."""
    results = search_top_k(user_query, top_k=top_k)
    if not results:
        return None

    query_emb = generate_embedding(user_query)

    candidates = []
    sentence_split_re = re.compile(r"(?<=[\.\?!\n])\s+")

    for r in results:
        sentences = [s.strip() for s in sentence_split_re.split(r["text"]) if s.strip()]
        if not sentences:
            sentences = [r["text"]]

        for s in sentences:
            try:
                s_emb = generate_embedding(s)
            except Exception:
                continue
            sim = _cosine(query_emb, s_emb)
            candidates.append({"sentence": s, "score": sim, "page": r["page"], "chunk_index": r["chunk_index"]})

    # sort candidates by score desc
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Prefer non-question, reasonably long sentences as answer
    for c in candidates:
        s = c["sentence"]
        if not s.endswith("?") and len(s) > 20:
            return {"answer": s, "page": c["page"], "chunk_index": c["chunk_index"], "score": c["score"]}

    # fallback to top candidate (even if it's a question)
    if candidates:
        top = candidates[0]
        return {"answer": top["sentence"], "page": top["page"], "chunk_index": top["chunk_index"], "score": top["score"]}

    # final fallback to passage snippet
    top = results[0]
    snippet = top["text"]
    if len(snippet) > 500:
        snippet = snippet[:500].rsplit(" ", 1)[0] + "..."
    return {"answer": snippet, "page": top["page"], "chunk_index": top["chunk_index"], "score": None}


def rag_with_llm(user_query, top_k=3, model_name="gpt-3.5-turbo", max_tokens=150, temperature=0.2):
    """Use an LLM to synthesize a concise answer from top-k passages.

    If `OPENAI_API_KEY` is not set, falls back to `rag_answer`.
    Returns dict with `answer` and `sources` (list of source metadata).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return rag_answer(user_query, top_k=top_k)

    results = search_top_k(user_query, top_k=top_k)
    if not results:
        return None

    context_blocks = []
    sources = []
    for i, r in enumerate(results):
        label = f"Source {i+1} (page {r['page']}, chunk {r['chunk_index']})"
        context_blocks.append(f"{label}:\n{r['text']}")
        sources.append({"label": label, "page": r['page'], "chunk_index": r['chunk_index']})

    prompt = (
        "You are a helpful assistant. Use only the information in the provided sources to answer the question. "
        "If the information is not present, say 'I don't know'. Provide a concise answer (1-3 sentences).\n\n"
        f"Question: {user_query}\n\nSources:\n" + "\n\n".join(context_blocks)
    )

    openai.api_key = api_key
    try:
        resp = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        answer_text = resp["choices"][0]["message"]["content"].strip()
    except Exception:
        return rag_answer(user_query, top_k=top_k)

    return {"answer": answer_text, "sources": sources}