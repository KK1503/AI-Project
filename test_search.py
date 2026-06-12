from app.vector_store import search_faq

print("Running search test...")
try:
    res = search_faq("how do I install the app")
    print("Result:\n", res)
except Exception as e:
    print("Error during test:", e)
