from app.oracledbconnection import load_oracle_documents, oracle_configured


print("Checking Oracle configuration...")

print("Detected env vars:")
for name in [
    "ORACLE_USER",
    "ORACLE_USERNAME",
    "DB_USER",
    "ADMIN",
    "ORACLE_PASSWORD",
    "DB_PASSWORD",
    "PASSWORD",
    "ORACLE_DSN",
    "DB_DSN",
    "CONNECTION_STRING",
    "DESCRIPTION",
    "ORACLE_HOST",
    "ORACLE_PORT",
    "ORACLE_SERVICE_NAME",
    "ORACLE_SERVICE",
    "ORACLE_SID",
    "ORACLE_TNS_ADMIN",
    "TNS_ADMIN",
]:
    value = __import__("os").getenv(name)
    if value:
        print(f"  {name}={value}")

if not oracle_configured():
    print("Missing one or more required Oracle env vars.")
    raise SystemExit(1)

try:
    documents = load_oracle_documents()
except Exception as exc:
    print("Oracle connection/query failed:")
    print(exc)
    raise SystemExit(1)

print(f"Loaded {len(documents)} Oracle documents.")

for document in documents[:3]:
    preview = document["text"][:120].replace("\n", " ")
    print(f"- DOC_ID={document['id']}: {preview}")