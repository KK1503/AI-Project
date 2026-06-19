import os


def _get_env_var(*names):
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _get_tns_admin():
    return _get_env_var(
        "ORACLE_TNS_ADMIN",
        "TNS_ADMIN",
        "ORACLE_WALLET_DIR",
        "WALLET_DIR",
        "WALLET_LOCATION",
    )


def oracle_configured():
    user = _get_env_var("ORACLE_USER", "ORACLE_USERNAME", "DB_USER", "ADMIN")
    password = _get_env_var("ORACLE_PASSWORD", "DB_PASSWORD", "PASSWORD")
    if not user or not password:
        return False

    if _get_env_var("ORACLE_DSN", "DB_DSN", "CONNECTION_STRING", "DESCRIPTION"):
        return True

    host = _get_env_var("ORACLE_HOST")
    port = _get_env_var("ORACLE_PORT")
    service = _get_env_var("ORACLE_SERVICE_NAME", "ORACLE_SERVICE", "ORACLE_SID")
    return bool(host and port and service)


def _get_oracle_dsn():
    dsn = _get_env_var("ORACLE_DSN", "DB_DSN", "CONNECTION_STRING")
    description = _get_env_var("DESCRIPTION")
    if dsn and dsn.strip().startswith("("):
        return dsn
    if description:
        return description
    if dsn:
        return dsn

    host = os.getenv("ORACLE_HOST")
    port = os.getenv("ORACLE_PORT")
    service = os.getenv("ORACLE_SERVICE_NAME") or os.getenv("ORACLE_SERVICE") or os.getenv("ORACLE_SID")
    if host and port and service:
        return f"{host}:{port}/{service}"

    return None


def load_oracle_documents():
    """Load rows from Oracle DOCUMENTS table as RAG documents.

    Required env vars:
      ORACLE_USER or ORACLE_USERNAME or DB_USER or ADMIN
      ORACLE_PASSWORD or DB_PASSWORD or PASSWORD

    Optional DSN env vars:
      ORACLE_DSN or DB_DSN or CONNECTION_STRING or DESCRIPTION

    Optional host/port/service env vars:
      ORACLE_HOST, ORACLE_PORT, ORACLE_SERVICE_NAME, ORACLE_SERVICE, ORACLE_SID

    Optional Oracle client config env vars:
      ORACLE_TNS_ADMIN or TNS_ADMIN

    Optional env vars:
      ORACLE_SQL

    Expected default table:
      documents(DOC_ID, CHUNK_TEXT, EMBEDIING)
    """
    if not oracle_configured():
        return []

    tns_admin = _get_tns_admin()
    if tns_admin:
        os.environ["TNS_ADMIN"] = tns_admin

    dsn = _get_oracle_dsn()
    if not dsn:
        raise RuntimeError(
            "Oracle DSN is not configured. Set ORACLE_DSN/DB_DSN/CONNECTION_STRING/DESCRIPTION or "
            "provide ORACLE_HOST/ORACLE_PORT/ORACLE_SERVICE_NAME (or ORACLE_SERVICE/ORACLE_SID)."
        )

    if dsn.strip().startswith("("):
        print("Using full DESCRIPTION string for Oracle connection.")

    try:
        import oracledb
    except ImportError as exc:
        raise RuntimeError(
            "Oracle support needs the 'oracledb' package. Run: pip install -r requirements.txt"
        ) from exc

    try:
        if tns_admin and hasattr(oracledb, "init_oracle_client"):
            try:
                oracledb.init_oracle_client(config_dir=tns_admin)
            except Exception:
                # If Oracle client libs are not installed or init fails, continue and rely on thin mode.
                pass
    except Exception:
        pass

    sql = os.getenv(
        "ORACLE_SQL",
        """
        SELECT doc_id, chunk_text, embediing
        FROM documents
        WHERE chunk_text IS NOT NULL
        ORDER BY doc_id
        """,
    )

    documents = []
    try:
        connection = oracledb.connect(
            user=_get_env_var("ORACLE_USER", "ORACLE_USERNAME", "DB_USER", "ADMIN"),
            password=_get_env_var("ORACLE_PASSWORD", "DB_PASSWORD", "PASSWORD"),
            dsn=dsn,
        )
    except Exception as exc:
        msg = str(exc)
        if "DPY-4027" in msg or "no configuration directory" in msg.lower():
            raise RuntimeError(
                "Oracle client configuration error (DPY-4027): no configuration directory specified. "
                "Set ORACLE_TNS_ADMIN/TNS_ADMIN/ORACLE_WALLET_DIR/WALLET_DIR/WALLET_LOCATION to the directory "
                "containing your Oracle wallet/tnsnames.ora/sqlnet.ora, or set ORACLE_DSN/DB_DSN/CONNECTION_STRING/"
                "DESCRIPTION to a full connection string. "
                f"Original error: {msg}"
            ) from exc
        raise

    try:
        with connection.cursor() as cursor:
            cursor.execute(sql)
            for row in cursor:
                doc_id = row[0]
                chunk_text = "" if row[1] is None else str(row[1]).strip()
                stored_embedding = row[2] if len(row) > 2 else None

                if chunk_text:
                    documents.append(
                        {
                            "source": "oracle",
                            "id": str(doc_id),
                            "text": chunk_text,
                            "stored_embedding": stored_embedding,
                        }
                    )
    finally:
        connection.close()

    return documents