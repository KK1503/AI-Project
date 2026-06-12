from app.vector_store import search_faq


def get_answer(user_question):
    result = search_faq(user_question)

    # Support legacy dict with 'answer' and new loader returning text
    if isinstance(result, dict):
        if "answer" in result:
            return result["answer"]
        if "text" in result:
            return result["text"]

    return "Sorry, I could not find an answer to your question."