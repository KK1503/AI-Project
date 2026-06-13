import json
import os
import re
import urllib.request
import numpy as np
import chromadb
from chromadb.config import Settings
from app.embeddings import generate_embedding
from PyPDF2 import PdfReader

# Path to the PDF file in the project root.
PDF_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "OCI FAQ.pdf"))
CHROMA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app_chroma"))


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


if len(passages) == 0:
    raise RuntimeError("No text extracted from PDF to index.")


def _create_chroma_client():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    settings = Settings(
        persist_directory=CHROMA_DIR,
        is_persistent=True,
    )
    return chromadb.Client(settings=settings)


client = _create_chroma_client()
collection = client.get_or_create_collection(name="faq")


def _build_chroma_collection():
    if hasattr(collection, "count") and collection.count() > 0:
        return
    if len(passages) == 0:
        raise RuntimeError("No text extracted from PDF to index.")

    ids = [f"doc-{i}" for i in range(len(passages))]
    documents = [p["text"] for p in passages]
    embeddings = [generate_embedding(p["text"]).tolist() for p in passages]
    metadatas = [{"page": p["page"], "chunk_index": p["chunk_index"]} for p in passages]

    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)


_build_chroma_collection()


def _query_chroma(user_query, top_k=1):
    query_embedding = generate_embedding(user_query).tolist()
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return result


def search_faq(user_query, top_k=1, snippet_max=1000):
    """Return the best-matching passage (snippet) and metadata."""
    result = _query_chroma(user_query, top_k=top_k)
    if not result or not result.get("documents"):
        return None

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result.get("distances", [[None]])[0]

    if len(documents) == 0:
        return None

    passage_text = documents[0]
    snippet = passage_text
    if len(snippet) > snippet_max:
        snippet = snippet[:snippet_max].rsplit(" ", 1)[0] + "..."

    metadata = metadatas[0] if metadatas else {}
    return {
        "text": snippet,
        "page": metadata.get("page"),
        "chunk_index": metadata.get("chunk_index"),
        "distance": float(distances[0]) if distances and distances[0] is not None else None,
    }


def search_top_k(user_query, top_k=3):
    """Return top_k matching passages with distances and indices."""
    result = _query_chroma(user_query, top_k=top_k)
    if not result or not result.get("documents"):
        return []

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result.get("distances", [[None] * len(documents)])[0]

    results = []
    for i, doc_text in enumerate(documents):
        metadata = metadatas[i] if i < len(metadatas) else {}
        distance = float(distances[i]) if distances and i < len(distances) and distances[i] is not None else None
        results.append({
            "text": doc_text,
            "page": metadata.get("page"),
            "chunk_index": metadata.get("chunk_index"),
            "distance": distance,
            "index": i,
        })
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


def _call_ollama(payload):
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    request = urllib.request.Request(
        f"{ollama_host}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def ollama_rewrite_answer(user_question, answer, model_name=None, temperature=0.2):
    """Rewrite a static FAQ answer with the Ollama model to sound professional."""
    model_name = model_name or os.getenv("OLLAMA_MODEL", "gemma2:2b")
    prompt = (
        "You are a professional assistant. Rewrite the following answer so it is polished, concise, and professional. "
        "Do not add new information beyond the original answer. Keep the meaning the same.\n\n"
        f"Question: {user_question}\n"
        f"Original Answer: {answer}\n\n"
        "Professional Answer:"
    )

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }

    try:
        resp = _call_ollama(payload)
        return resp.get("response", "").strip()
    except Exception:
        return None


def rag_with_ollama(user_query, top_k=3, model_name=None, temperature=0.2):
    """Use Ollama to synthesize a concise answer from top-k passages."""
    model_name = model_name or os.getenv("OLLAMA_MODEL", "gemma2:2b")
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

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }

    try:
        resp = _call_ollama(payload)
        answer_text = resp.get("response", "").strip()
    except Exception:
        return rag_answer(user_query, top_k=top_k)

    if not answer_text:
        return rag_answer(user_query, top_k=top_k)

    return {"answer": answer_text, "sources": sources}


def rag_with_llm(user_query, top_k=3, model_name=None, temperature=0.2):
    """Backward-compatible wrapper for the Ollama RAG answer."""
    return rag_with_ollama(user_query, top_k=top_k, model_name=model_name, temperature=temperature)
