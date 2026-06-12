import os
import faiss
import numpy as np
from app.embeddings import generate_embedding
from PyPDF2 import PdfReader

# Path to the PDF file (update if needed)
PDF_PATH = r"C:\Users\shail\Downloads\FAQ.pdf"


def load_pdf_documents(pdf_path=PDF_PATH):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append(text)

    return pages


# Load documents (one entry per PDF page)
documents = load_pdf_documents()

# Generate embeddings for documents
doc_embeddings = np.array([generate_embedding(d) for d in documents]).astype("float32")

# Create FAISS index
if doc_embeddings.size == 0:
    raise RuntimeError("No embeddings generated from PDF documents.")

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)


def search_faq(user_query, top_k=1):
    query_embedding = np.array([generate_embedding(user_query)]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    best_idx = indices[0][0]
    best_distance = float(distances[0][0])

    return {"text": documents[best_idx], "distance": best_distance}