import os
import sqlite3
import pickle
import hashlib
from typing import List, Sequence, Optional

from langchain_core.embeddings import Embeddings


class SqliteEmbeddingCache:
    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._init()

    def _init(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                  model TEXT NOT NULL,
                  hash TEXT NOT NULL,
                  vec  BLOB NOT NULL,
                  PRIMARY KEY (model, hash)
                )
                """
            )

    @staticmethod
    def content_hash(text: str) -> str:
        m = hashlib.md5()
        m.update(text.encode("utf-8"))
        return m.hexdigest()

    def get_many(self, model: str, hashes: List[str]) -> dict:
        if not hashes:
            return {}
        qmarks = ",".join(["?"] * len(hashes))
        rows = self.conn.execute(
            f"SELECT hash, vec FROM embeddings WHERE model=? AND hash IN ({qmarks})",
            [model, *hashes],
        ).fetchall()
        out = {}
        for h, blob in rows:
            out[h] = pickle.loads(blob)
        return out

    def set_many(self, model: str, items: List[tuple[str, Sequence[float]]]) -> None:
        if not items:
            return
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO embeddings (model, hash, vec) VALUES (?, ?, ?)",
                [(model, h, pickle.dumps(list(vec))) for h, vec in items],
            )


class CachedEmbeddings(Embeddings):
    """Wrap another Embeddings to provide SQLite-backed caching by content hash.

    Cache key: (model_name, md5(text))
    """

    def __init__(self, inner: Embeddings, cache_path: str = "./data/emb_cache.sqlite") -> None:
        self.inner = inner
        self.cache = SqliteEmbeddingCache(cache_path)
        # Attempt to infer model name
        self.model_name = getattr(inner, "model_name", getattr(inner, "model", "unknown-model"))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        hashes = [self.cache.content_hash(t or "") for t in texts]
        cached = self.cache.get_many(self.model_name, hashes)
        missing_idx = [i for i, h in enumerate(hashes) if h not in cached]
        if missing_idx:
            missing_texts = [texts[i] for i in missing_idx]
            computed = self.inner.embed_documents(missing_texts)
            self.cache.set_many(
                self.model_name,
                [(hashes[i], vec) for i, vec in zip(missing_idx, computed)],
            )
            # Update cached map
            for i, vec in zip(missing_idx, computed):
                cached[hashes[i]] = vec
        # Return in original order
        return [cached[h] for h in hashes]

    def embed_query(self, text: str) -> List[float]:
        h = self.cache.content_hash(text or "")
        got = self.cache.get_many(self.model_name, [h])
        if h in got:
            return got[h]
        vec = self.inner.embed_query(text)
        self.cache.set_many(self.model_name, [(h, vec)])
        return vec
