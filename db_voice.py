"""
Voice metadata in DB (SQLite or PostgreSQL). Optional: set DATABASE_URL to use DB for metadata.
.pt files remain in local or S3; this layer only stores voice_id, name, consent_scope, created_at.
"""
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Optional

from config import DATABASE_URL

# SQLite or psycopg2 connection
_conn: Any = None


def _is_sqlite() -> bool:
    return bool(DATABASE_URL and DATABASE_URL.startswith("sqlite"))


def _is_postgres() -> bool:
    return bool(
        DATABASE_URL
        and (DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith("postgres://"))
    )


def _sqlite_path() -> str:
    """Return path for sqlite3.connect (strip sqlite:///)."""
    u = DATABASE_URL
    if u.startswith("sqlite:///"):
        return u[10:]
    if u == "sqlite://" or u.startswith("sqlite:"):
        return os.path.join(os.path.dirname(__file__), "voice_metadata.db")
    return u


def _get_conn():
    global _conn
    if _conn is not None:
        return _conn
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")
    if _is_sqlite():
        path = _sqlite_path()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(path)
        _conn.row_factory = sqlite3.Row
        _init_schema_sqlite(_conn)
    elif _is_postgres():
        import psycopg2
        _conn = psycopg2.connect(DATABASE_URL)
        _init_schema_pg(_conn)
    else:
        raise ValueError("DATABASE_URL must be sqlite://... or postgresql://...")
    return _conn


def _init_schema_sqlite(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS voices (
            voice_id TEXT PRIMARY KEY,
            name TEXT NOT NULL DEFAULT '',
            consent_scope TEXT NOT NULL DEFAULT 'tts',
            created_at REAL NOT NULL,
            owner_id TEXT
        )
    """)
    conn.commit()
    try:
        conn.execute("ALTER TABLE voices ADD COLUMN owner_id TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists


def _init_schema_pg(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS voices (
                voice_id TEXT PRIMARY KEY,
                name TEXT NOT NULL DEFAULT '',
                consent_scope TEXT NOT NULL DEFAULT 'tts',
                created_at DOUBLE PRECISION NOT NULL,
                owner_id TEXT
            )
        """)
        cur.execute("""
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'voices' AND column_name = 'owner_id'
        """)
        if cur.fetchone() is None:
            cur.execute("ALTER TABLE voices ADD COLUMN owner_id TEXT")
    conn.commit()


def db_insert_voice(
    voice_id: str,
    name: str,
    consent_scope: str,
    created_at: float,
    owner_id: Optional[str] = None,
) -> None:
    conn = _get_conn()
    name = name or ""
    consent_scope = consent_scope or "tts"
    if _is_sqlite():
        conn.execute(
            "INSERT OR REPLACE INTO voices (voice_id, name, consent_scope, created_at, owner_id) VALUES (?, ?, ?, ?, ?)",
            (voice_id, name, consent_scope, created_at, owner_id),
        )
        conn.commit()
    else:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO voices (voice_id, name, consent_scope, created_at, owner_id)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (voice_id) DO UPDATE SET name = EXCLUDED.name, consent_scope = EXCLUDED.consent_scope, created_at = EXCLUDED.created_at, owner_id = EXCLUDED.owner_id
                """,
                (voice_id, name, consent_scope, created_at, owner_id),
            )
        conn.commit()


def db_get_voice(voice_id: str, owner_id: Optional[str] = None) -> Optional[dict]:
    conn = _get_conn()
    if _is_sqlite():
        row = conn.execute(
            "SELECT voice_id, name, consent_scope, created_at, owner_id FROM voices WHERE voice_id = ?",
            (voice_id,),
        ).fetchone()
    else:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT voice_id, name, consent_scope, created_at, owner_id FROM voices WHERE voice_id = %s",
                (voice_id,),
            )
            row = cur.fetchone()
    if row is None:
        return None
    row_owner = row[4] if len(row) > 4 else None
    if owner_id is not None and row_owner is not None and row_owner != owner_id:
        return None
    return {"voice_id": row[0], "name": row[1] or "", "consent_scope": row[2] or "tts", "created_at": row[3]}


def db_list_voices(owner_id: Optional[str] = None) -> list[dict]:
    conn = _get_conn()
    if owner_id is None:
        q = "SELECT voice_id, name, consent_scope, created_at FROM voices ORDER BY created_at DESC"
        args = ()
    else:
        q = "SELECT voice_id, name, consent_scope, created_at FROM voices WHERE owner_id = ? ORDER BY created_at DESC"
        args = (owner_id,)
    if _is_sqlite():
        rows = conn.execute(q, args).fetchall()
    else:
        q_pg = q.replace("?", "%s")
        with conn.cursor() as cur:
            cur.execute(q_pg, args)
            rows = cur.fetchall()
    return [{"voice_id": r[0], "name": r[1] or "", "consent_scope": r[2] or "tts", "created_at": r[3]} for r in rows]


def db_update_voice(voice_id: str, name: Optional[str] = None, owner_id: Optional[str] = None) -> bool:
    if name is None:
        return True
    conn = _get_conn()
    name_val = (name or "").strip()
    if owner_id is not None:
        if _is_sqlite():
            cur = conn.execute(
                "UPDATE voices SET name = ? WHERE voice_id = ? AND (owner_id IS NULL OR owner_id = ?)",
                (name_val, voice_id, owner_id),
            )
            conn.commit()
            return cur.rowcount > 0
        else:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE voices SET name = %s WHERE voice_id = %s AND (owner_id IS NULL OR owner_id = %s)",
                    (name_val, voice_id, owner_id),
                )
                conn.commit()
                return cur.rowcount > 0
    if _is_sqlite():
        cur = conn.execute("UPDATE voices SET name = ? WHERE voice_id = ?", (name_val, voice_id))
        conn.commit()
        return cur.rowcount > 0
    else:
        with conn.cursor() as cur:
            cur.execute("UPDATE voices SET name = %s WHERE voice_id = %s", (name_val, voice_id))
            conn.commit()
            return cur.rowcount > 0


def db_delete_voice(voice_id: str, owner_id: Optional[str] = None) -> bool:
    conn = _get_conn()
    if owner_id is not None:
        if _is_sqlite():
            cur = conn.execute(
                "DELETE FROM voices WHERE voice_id = ? AND (owner_id IS NULL OR owner_id = ?)",
                (voice_id, owner_id),
            )
            conn.commit()
            return cur.rowcount > 0
        else:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM voices WHERE voice_id = %s AND (owner_id IS NULL OR owner_id = %s)",
                    (voice_id, owner_id),
                )
                conn.commit()
                return cur.rowcount > 0
    if _is_sqlite():
        cur = conn.execute("DELETE FROM voices WHERE voice_id = ?", (voice_id,))
        conn.commit()
        return cur.rowcount > 0
    else:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM voices WHERE voice_id = %s", (voice_id,))
            conn.commit()
            return cur.rowcount > 0


def use_db() -> bool:
    """True if DATABASE_URL is set and we should use DB for metadata."""
    return bool(DATABASE_URL)
