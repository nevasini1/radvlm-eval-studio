"""SQLite helpers for the local audit log and metadata."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from radvlm_eval import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    study_id TEXT NOT NULL,
    draft_before TEXT,
    draft_after TEXT,
    reviewer_action TEXT,
    metrics_before_json TEXT,
    metrics_after_json TEXT,
    notes TEXT
);
"""


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    db_path = Path(db_path) if db_path else config.DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()
