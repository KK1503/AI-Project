from app.vector_store import rag_answer, search_faq, rag_with_llm
import os


def get_answer(user_question):
    # Prefer LLM-synthesized RAG answer when API key is available
    if os.getenv("OPENAI_API_KEY"):
        llm_res = rag_with_llm(user_question, top_k=3)
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