
import sqlite3
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import os
import uuid
from init_llm import *


from common import *

# ---------- DB Helpers ----------

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS candidates (
            id TEXT PRIMARY KEY,
            name TEXT,
            profession TEXT,
            years_experience REAL,
            filepath TEXT,
            created_at TEXT,
            notes TEXT
        )
        """
    )
    con.commit()
    con.close()


def db_execute(query: str, params: Tuple = ()):  # simple helper
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(query, params)
    con.commit()
    con.close()


def db_query(query: str, params: Tuple = ()) -> List[Tuple]:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(query, params)
    rows = cur.fetchall()
    con.close()
    return rows




def load_candidates() -> List[Dict[str, Any]]:
    rows = db_query("SELECT id, name, profession, years_experience, filepath, created_at, notes FROM candidates ORDER BY created_at DESC")
    out = []
    for r in rows:
        out.append({
            "id": r[0],
            "name": r[1],
            "profession": r[2],
            "years_experience": r[3],
            "filepath": r[4],
            "created_at": r[5],
            "notes": r[6],
        })
    return out


def add_or_update_candidate(candidate_id: Optional[str], name: str|None, profession: str|None, years: Optional[float], filepath: str, notes: str) -> str:
    now = datetime.utcnow().isoformat()
    if candidate_id is None:
        candidate_id = uuid.uuid4().hex
        db_execute(
            "INSERT INTO candidates (id, name, profession, years_experience, filepath, created_at, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (candidate_id, name, profession, years if years is not None else None, filepath, now, notes),
        )
    else:
        db_execute(
            "UPDATE candidates SET name=?, profession=?, years_experience=?, filepath=?, notes=? WHERE id=?",
            (name, profession, years if years is not None else None, filepath, notes, candidate_id),
        )
    return candidate_id
