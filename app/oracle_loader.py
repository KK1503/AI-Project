"""Compatibility shim: expose the older `app.oracle_loader` API.

This module delegates to `app.oracledbconnection` so existing imports
like `from app.oracle_loader import load_oracle_documents` continue
to work.
"""
try:
    from app.oracledbconnection import load_oracle_documents, oracle_configured
except Exception:
    # Fallback to relative import when package semantics differ
    from .oracledbconnection import load_oracle_documents, oracle_configured

__all__ = ["load_oracle_documents", "oracle_configured"]
