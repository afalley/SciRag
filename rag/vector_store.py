# Python
"""
A tiny SQLite-backed vector store.
- Stores documents as chunked text with metadata.
- Stores embeddings as JSON text (list of floats).
- Performs naive cosine similarity in Python with numpy.

Why simple? fewer dependencies, easy to inspect and learn from.
You can later swap to FAISS or Qdrant without changing higher-level code too much.
"""

import json
import os
import sqlite3
from typing import List, Tuple, Dict, Any
import numpy as np


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,          -- source document identifier (file path or custom ID)
    chunk_id INTEGER NOT NULL,     -- index of the chunk in that document
    text TEXT NOT NULL,            -- chunk text
    meta_json TEXT NOT NULL,       -- metadata (JSON)
    embedding_json TEXT NOT NULL   -- embedding (JSON) for this chunk
);
CREATE INDEX IF NOT EXISTS idx_doc_id ON documents (doc_id);
"""


class VectorStore:
    def __init__(self, db_path: str):
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA)

    def add_many(
        self,
        doc_id: str,
        chunks: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        """
        Adds many chunks for a given document in one transaction.
        """
        assert len(chunks) == len(metadatas) == len(embeddings)
        rows = []
        for i, (text, meta, emb) in enumerate(zip(chunks, metadatas, embeddings)):
            rows.append(
                (
                    doc_id,
                    i,
                    text,
                    json.dumps(meta, ensure_ascii=False),
                    json.dumps(emb),
                )
            )

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT INTO documents (doc_id, chunk_id, text, meta_json, embedding_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )

    def all(self) -> List[Tuple[int, str, int, str, Dict[str, Any], List[float]]]:
        """
        Loads all rows, parsing JSON fields.
        Returns tuples: (id, doc_id, chunk_id, text, meta, embedding)
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("SELECT * FROM documents")
            results = []
            for row in cur.fetchall():
                results.append(
                    (
                        row["id"],
                        row["doc_id"],
                        row["chunk_id"],
                        row["text"],
                        json.loads(row["meta_json"]),
                        json.loads(row["embedding_json"]),
                    )
                )
            return results

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Naive in-Python cosine similarity search over all vectors.
        Suitable for small-to-medium corpora. For large corpora, replace with FAISS/Qdrant.
        Returns a list of {id, doc_id, chunk_id, text, meta, score}.
        """
        rows = self.all()
        if not rows:
            return []

        # Convert to arrays for vectorized cosine similarity.
        matrix = np.array([r[5] for r in rows], dtype=np.float32)  # embeddings
        query = np.array(query_embedding, dtype=np.float32)

        # Cosine similarity: (A · B) / (||A|| * ||B||)
        denom = (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query) + 1e-12)
        sims = matrix @ query / denom

        # Get top_k indices
        idxs = np.argsort(-sims)[:top_k]
        out = []
        for idx in idxs:
            row = rows[int(idx)]
            out.append(
                {
                    "id": row[0],
                    "doc_id": row[1],
                    "chunk_id": row[2],
                    "text": row[3],
                    "meta": row[4],
                    "score": float(sims[int(idx)]),
                }
            )
        return out