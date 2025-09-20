import os
import sqlite3
import threading
from typing import List, Dict, Any, Optional, Tuple

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conv_id TEXT NOT NULL,
  role TEXT NOT NULL, -- 'user' | 'assistant'
  content TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  citations TEXT,
  FOREIGN KEY(conv_id) REFERENCES conversations(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conv_id, id);
"""

class ConversationDB:
  def __init__(self, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Allow usage from multiple threads (FastAPI worker threads) and guard with a lock
    self.conn = sqlite3.connect(path, check_same_thread=False)
    self.conn.row_factory = sqlite3.Row
    self.lock = threading.Lock()
    with self.conn:
      self.conn.executescript(SCHEMA)
      # Ensure 'citations' column exists even on older DBs
      cur = self.conn.execute("PRAGMA table_info(messages)")
      cols = [r[1] for r in cur.fetchall()]
      if 'citations' not in cols:
        self.conn.execute("ALTER TABLE messages ADD COLUMN citations TEXT")

  def create_conversation(self, conv_id: str, title: str, ts: int) -> None:
    with self.lock, self.conn:
      self.conn.execute(
        "INSERT OR REPLACE INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (conv_id, title, ts, ts),
      )

  def update_title(self, conv_id: str, title: str, ts: int) -> None:
    with self.lock, self.conn:
      self.conn.execute(
        "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
        (title, ts, conv_id),
      )

  def list_conversations(self) -> List[Dict[str, Any]]:
    with self.lock:
      cur = self.conn.execute("SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC")
      return [dict(r) for r in cur.fetchall()]

  def get_conversation(self, conv_id: str) -> Optional[Dict[str, Any]]:
    with self.lock:
      cur = self.conn.execute("SELECT id, title, created_at, updated_at FROM conversations WHERE id=?", (conv_id,))
      r = cur.fetchone()
      return dict(r) if r else None

  def add_message(self, conv_id: str, role: str, content: str, ts: int, citations_json: Optional[str] = None) -> int:
    with self.lock, self.conn:
      cur = self.conn.execute(
        "INSERT INTO messages (conv_id, role, content, created_at, citations) VALUES (?, ?, ?, ?, ?)",
        (conv_id, role, content, ts, citations_json),
      )
      self.conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (ts, conv_id))
      return int(cur.lastrowid)

  def get_messages(self, conv_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    with self.lock:
      cur = self.conn.execute(
        "SELECT id, role, content, created_at, citations FROM messages WHERE conv_id=? ORDER BY id DESC LIMIT ?",
        (conv_id, limit),
      )
      rows = [dict(r) for r in cur.fetchall()]
    rows.reverse()
    return rows

  def delete_conversation(self, conv_id: str) -> None:
    with self.lock, self.conn:
      self.conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))

  def insert_message(self, conv_id: str, role: str, content: str, ts: int) -> None:
    """Insert a message without updating updated_at (for imports)."""
    with self.lock, self.conn:
      self.conn.execute(
        "INSERT INTO messages (conv_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (conv_id, role, content, ts),
      )
  def close(self):
    self.conn.close()
