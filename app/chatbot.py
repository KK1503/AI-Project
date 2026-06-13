import json
import os
import re
from difflib import SequenceMatcher


FAQ_DATA_PATH = os.path.join(os.path.dirname(__file__), "faq_data.json")


def _normalize_question(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _load_faq_data():
    with open(FAQ_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


FAQ_DATA = _load_faq_data()


def search_json_faq(user_question, min_score=0.86):
    """Return a direct FAQ answer when the question matches a known FAQ."""
    normalized_query = _normalize_question(user_question)

    for item in FAQ_DATA:
        if normalized_query == _normalize_question(item["question"]):
            return _format_faq_answer(item["question"], item["answer"])

    best_score = 0.0
    best_answer = None
    for item in FAQ_DATA:
        score = SequenceMatcher(None, normalized_query, _normalize_question(item["question"])).ratio()
        if score > best_score:
            best_score = score
            best_answer = item["answer"]

    if best_score >= min_score:
        return _format_faq_answer(user_question, best_answer)

    return None


def _format_faq_answer(question, answer):
    """Make direct FAQ answers read like polished assistant responses."""
    if ";" in answer and "\n" not in answer:
        items = [item.strip().rstrip(".") for item in answer.split(";") if item.strip()]
        if len(items) >= 3:
            intro = "The main OCI service categories are:" if "service categories" in _normalize_question(question) else "Here are the key points:"
            return intro + "\n\n" + "\n".join(f"- {item}" for item in items)

    if answer.startswith("A "):
        return answer

    return answer


def get_answer(user_question):
    faq_answer = search_json_faq(user_question)
    if faq_answer:
        from app.vector_store import ollama_rewrite_answer

        polished = ollama_rewrite_answer(user_question, faq_answer, temperature=0.2)
        return polished if polished else faq_answer

    from app.vector_store import rag_answer, search_faq, rag_with_ollama

    # Prefer Ollama-synthesized RAG answer when the question is not in JSON FAQ.
    llm_res = rag_with_ollama(user_question, top_k=3)
    if llm_res and "answer" in llm_res:
        return llm_res["answer"]

    # Use RAG-style concise answer (local selection)
    rag = rag_answer(user_question, top_k=3)
    if rag and "answer" in rag:
        # return concise answer text
        return rag["answer"]

    # fallback to older search
    result = search_faq(user_question)
    if isinstance(result, dict):
        if "answer" in result:
            return result["answer"]
        if "text" in result:
            # return only first 400 chars so it's not overwhelming
            t = result["text"]
            return t if len(t) <= 400 else t[:400].rsplit(" ", 1)[0] + "..."

    return "Sorry, I could not find an answer to your question."
