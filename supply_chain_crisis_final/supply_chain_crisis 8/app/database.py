from contextlib import contextmanager
import sqlite3
from .config import DB_PATH

@contextmanager
def get_connection():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"SQLite database not found at {DB_PATH}. Run scripts/build_database.py first.")
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    try: yield conn
    finally: conn.close()

def rows(sql: str, params=()):
    with get_connection() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]

def one(sql: str, params=()):
    with get_connection() as conn:
        row = conn.execute(sql, params).fetchone()
        return dict(row) if row else None
