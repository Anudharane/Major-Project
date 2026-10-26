"""Tiny SQLite store for prediction history. No personal/medical information is stored:
only time, file name, a small thumbnail, the prediction and its confidence."""
import sqlite3
from datetime import datetime
from typing import List

import config


def _conn():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(config.HISTORY_DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("""CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, filename TEXT, thumbnail TEXT,
        prediction TEXT, confidence REAL, screening_result TEXT)""")
    return c


def add(filename: str, thumbnail_b64: str, prediction: str, confidence: float, screening_result: str):
    with _conn() as c:
        c.execute("INSERT INTO history (created_at, filename, thumbnail, prediction, confidence, screening_result) "
                  "VALUES (?,?,?,?,?,?)",
                  (datetime.now().strftime("%d-%m-%Y %H:%M"), filename[:120], thumbnail_b64, prediction,
                   round(confidence, 1), screening_result))


def list_all(limit: int = config.HISTORY_LIMIT) -> List[dict]:
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,))]


def clear():
    with _conn() as c:
        c.execute("DELETE FROM history")
