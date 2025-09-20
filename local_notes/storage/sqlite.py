import os
import sqlite3
from typing import Optional, Dict, Any, List


SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id TEXT PRIMARY KEY,
    title TEXT,
    folder TEXT,
    modified_ts INTEGER,
    modified_raw TEXT,
    hash TEXT,
    body_md TEXT,
    last_indexed_ts INTEGER
);
CREATE INDEX IF NOT EXISTS idx_notes_modified ON notes(modified_ts);
"""


class NotesDB:
    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._init()

    def _init(self):
        with self.conn:
            self.conn.executescript(SCHEMA)

    def upsert_note(self, rec: Dict[str, Any]) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO notes (id, title, folder, modified_ts, modified_raw, hash, body_md, last_indexed_ts)
                VALUES (:id, :title, :folder, :modified_ts, :modified_raw, :hash, :body_md, COALESCE(:last_indexed_ts, strftime('%s','now')))
                ON CONFLICT(id) DO UPDATE SET
                  title=excluded.title,
                  folder=excluded.folder,
                  modified_ts=excluded.modified_ts,
                  modified_raw=excluded.modified_raw,
                  hash=excluded.hash,
                  body_md=excluded.body_md,
                  last_indexed_ts=excluded.last_indexed_ts
                """,
                rec,
            )

    def get(self, note_id: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute("SELECT * FROM notes WHERE id=?", (note_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def list_all_ids(self) -> List[str]:
        cur = self.conn.execute("SELECT id FROM notes")
        return [r[0] for r in cur.fetchall()]

    def close(self):
        self.conn.close()
