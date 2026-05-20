"""
storage.py — SQLite-backed persistent storage for receipt data.

All database access uses parameterized queries to prevent SQL injection.
Receipt images are stored as BLOBs to avoid filesystem path-traversal risks.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Database path — stored in a ``data/`` directory next to the app
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent / "data"
_DB_PATH = _DATA_DIR / "receipts.db"


def _get_connection() -> sqlite3.Connection:
    """Return a connection to the SQLite database with row_factory set."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create the database tables if they don't already exist."""
    conn = _get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS receipts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                merchant    TEXT,
                date        TEXT,
                subtotal    REAL,
                tax         REAL,
                tip         REAL,
                grand_total REAL,
                item_count  INTEGER,
                image_blob  BLOB,
                image_name  TEXT,
                created_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS line_items (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                receipt_id  INTEGER NOT NULL,
                item        TEXT,
                qty         INTEGER DEFAULT 1,
                unit_price  REAL,
                total_price REAL,
                tag         TEXT,
                FOREIGN KEY (receipt_id) REFERENCES receipts(id)
            );
        """)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

def save_receipt(
    receipt_data: dict,
    image_bytes: Optional[bytes] = None,
    image_name: Optional[str] = None,
) -> int:
    """Persist a receipt and its line items.

    Parameters
    ----------
    receipt_data:
        A dict matching the ``ReceiptData`` schema (as produced by
        ``ReceiptData.model_dump()``).
    image_bytes:
        Raw bytes of the original receipt image (stored as a BLOB).
    image_name:
        The original filename of the uploaded image.

    Returns
    -------
    The auto-generated ``receipt_id``.
    """
    fin = receipt_data.get("financials", {})
    conn = _get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO receipts
                (merchant, date, subtotal, tax, tip, grand_total,
                 item_count, image_blob, image_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt_data.get("merchant_name"),
                receipt_data.get("date"),
                fin.get("subtotal"),
                fin.get("tax", 0.0),
                fin.get("tip", 0.0),
                fin.get("grand_total", 0.0),
                receipt_data.get(
                    "total_items_counted",
                    len(receipt_data.get("line_items", [])),
                ),
                image_bytes,
                image_name,
            ),
        )
        receipt_id = cursor.lastrowid

        for li in receipt_data.get("line_items", []):
            conn.execute(
                """
                INSERT INTO line_items
                    (receipt_id, item, qty, unit_price, total_price, tag)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt_id,
                    li.get("item"),
                    li.get("qty", 1),
                    li.get("unit_price"),
                    li.get("total_price"),
                    li.get("tag"),
                ),
            )

        conn.commit()
        return receipt_id
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------

def get_all_receipts() -> list[dict]:
    """Return metadata for every saved receipt (no image blobs).

    Returns a list of dicts ordered by ``created_at`` descending.
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, merchant, date, subtotal, tax, tip, grand_total,
                   item_count, image_name, created_at
            FROM receipts
            ORDER BY created_at DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_receipt_detail(receipt_id: int) -> Optional[dict]:
    """Return full receipt data including line items and image bytes.

    Parameters
    ----------
    receipt_id:
        The primary key of the receipt to retrieve.

    Returns
    -------
    A dict with keys ``receipt`` (row data + ``image_blob``) and
    ``line_items`` (list of dicts), or ``None`` if not found.
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM receipts WHERE id = ?", (receipt_id,)
        ).fetchone()
        if row is None:
            return None

        items = conn.execute(
            """
            SELECT item, qty, unit_price, total_price, tag
            FROM line_items
            WHERE receipt_id = ?
            ORDER BY id
            """,
            (receipt_id,),
        ).fetchall()

        return {
            "receipt": dict(row),
            "line_items": [dict(i) for i in items],
        }
    finally:
        conn.close()


def get_all_receipts_with_items() -> list[dict]:
    """Return all receipts with their line items (for chat context injection).

    Returns a lightweight list (no image blobs) suitable for serialising
    into an LLM system prompt.
    """
    conn = _get_connection()
    try:
        receipts = conn.execute(
            """
            SELECT id, merchant, date, subtotal, tax, tip, grand_total,
                   item_count, created_at
            FROM receipts
            ORDER BY created_at DESC
            """
        ).fetchall()

        result = []
        for r in receipts:
            items = conn.execute(
                """
                SELECT item, qty, unit_price, total_price, tag
                FROM line_items
                WHERE receipt_id = ?
                ORDER BY id
                """,
                (r["id"],),
            ).fetchall()

            result.append({
                "merchant": r["merchant"],
                "date": r["date"],
                "grand_total": r["grand_total"],
                "tax": r["tax"],
                "tip": r["tip"],
                "subtotal": r["subtotal"],
                "line_items": [dict(i) for i in items],
            })

        return result
    finally:
        conn.close()
