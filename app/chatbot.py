from app.vector_store import search_faq


def get_answer(user_question):
    result = search_faq(user_question)

    if result:
        return result["answer"]

    return "Sorry, I could not find an answer to your question."