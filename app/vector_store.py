import json
import faiss
import numpy as np
from app.embeddings import generate_embedding

# Load FAQ data
with open("app/faq_data.json", "r") as file:
    faq_data = json.load(file)

# Extract questions
questions = [item["question"] for item in faq_data]

# Generate embeddings
question_embeddings = np.array(
    [generate_embedding(question) for question in questions]
).astype("float32")

# Create FAISS index
dimension = question_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(question_embeddings)


def search_faq(user_query, top_k=1):
    query_embedding = np.array(
        [generate_embedding(user_query)]
    ).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    best_match_index = indices[0][0]

    return faq_data[best_match_index]