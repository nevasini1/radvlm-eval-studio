"""Append-only audit log persisted in SQLite."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from radvlm_eval.storage.sqlite import get_connection, init_db

VALID_ACTIONS = {"save_edit", "mark_reviewed", "mark_escalation", "export"}


def write_audit_entry(
    study_id: str,
    draft_before: str,
    draft_after: str,
    reviewer_action: str,
    metrics_before: Optional[Dict[str, Any]] = None,
    metrics_after: Optional[Dict[str, Any]] = None,
    notes: str = "",
    db_path: Optional[Path] = None,
    timestamp: Optional[str] = None,
) -> int:
    """Insert an audit entry; returns the new row id."""
    init_db(db_path)
    ts = timestamp or _dt.datetime.now().isoformat(timespec="seconds")
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            """
            INSERT INTO audit_log
                (timestamp, study_id, draft_before, draft_after, reviewer_action,
                 metrics_before_json, metrics_after_json, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                study_id,
                draft_before,
                draft_after,
                reviewer_action,
                json.dumps(metrics_before or {}),
                json.dumps(metrics_after or {}),
                notes,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def list_audit_entries(
    study_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        if study_id:
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE study_id = ? ORDER BY id DESC",
                (study_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY id DESC"
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["metrics_before"] = json.loads(d.pop("metrics_before_json") or "{}")
            d["metrics_after"] = json.loads(d.pop("metrics_after_json") or "{}")
            out.append(d)
        return out
    finally:
        conn.close()
