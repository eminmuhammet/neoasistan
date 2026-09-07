from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

# Deliberately not embedding-based: a real vector index needs a model
# (sentence-transformers et al. pull in a multi-hundred-MB ML stack this
# app otherwise has no use for) plus its own storage and upkeep. TF-IDF
# over whole documents is pure Python/stdlib, works fully offline, and
# answers the actual use case this was asked for -- "which of my files
# mentions X" -- without that cost. It won't catch synonyms the way real
# embeddings would; that tradeoff is deliberate, not an oversight.

_TR_FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")
_TOKEN_RE = re.compile(r"[a-zçğıöşü0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    lowered = text.replace("I", "ı").replace("İ", "i").lower()
    folded = lowered.translate(_TR_FOLD)
    return _TOKEN_RE.findall(folded)


@dataclass
class IndexedDocument:
    id: int
    path: str
    name: str
    indexed_at: str


@dataclass
class SearchHit:
    path: str
    name: str
    score: float
    snippet: str


def _make_snippet(content: str, query_terms: set[str], width: int = 220) -> str:
    """Centers the snippet on the first place a query term actually
    appears, rather than always returning the start of the document --
    the start is rarely where the relevant sentence is."""
    lowered = content.lower()
    best_at = None
    for term in query_terms:
        idx = lowered.find(term)
        if idx != -1 and (best_at is None or idx < best_at):
            best_at = idx
    if best_at is None:
        best_at = 0
    start = max(0, best_at - width // 2)
    end = min(len(content), start + width)
    snippet = content[start:end].strip()
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(content) else ""
    return f"{prefix}{snippet}{suffix}"


class DocumentIndex:
    """Local, offline full-text search over documents NEO has been asked
    to index (see neo/tools/document_search.py's IndexFolderTool /
    SearchDocumentsTool). One file per NEO install, same pattern as every
    other *_store.py in this codebase.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS indexed_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    indexed_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def upsert(self, path: str, name: str, content: str) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO indexed_documents (path, name, content, indexed_at)
                VALUES (?, ?, ?, datetime('now'))
                ON CONFLICT(path) DO UPDATE SET
                    name = excluded.name, content = excluded.content,
                    indexed_at = excluded.indexed_at
                """,
                (path, name, content),
            )
            conn.commit()

    def remove(self, path: str) -> bool:
        with closing(self._connect()) as conn:
            cursor = conn.execute("DELETE FROM indexed_documents WHERE path = ?", (path,))
            conn.commit()
            return cursor.rowcount > 0

    def list_documents(self) -> list[IndexedDocument]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT id, path, name, indexed_at FROM indexed_documents ORDER BY name"
            ).fetchall()
        return [IndexedDocument(r["id"], r["path"], r["name"], r["indexed_at"]) for r in rows]

    def count(self) -> int:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM indexed_documents").fetchone()
        return int(row["n"])

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        query_terms = set(_tokenize(query))
        if not query_terms:
            return []

        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT path, name, content FROM indexed_documents").fetchall()
        if not rows:
            return []

        docs = [(r["path"], r["name"], r["content"], _tokenize(r["content"])) for r in rows]
        num_docs = len(docs)

        doc_freq: Counter[str] = Counter()
        for _, _, _, tokens in docs:
            doc_freq.update(set(tokens) & query_terms)

        idf = {
            term: math.log((num_docs + 1) / (1 + doc_freq.get(term, 0))) + 1.0
            for term in query_terms
        }

        scored: list[SearchHit] = []
        for path, name, content, tokens in docs:
            if not tokens:
                continue
            term_counts = Counter(tokens)
            score = sum(term_counts.get(term, 0) * idf[term] for term in query_terms)
            if score <= 0:
                continue
            # Penalize raw length so a huge file that happens to contain the
            # term once doesn't outrank a short, focused one that's mostly
            # about it.
            normalized = score / math.sqrt(len(tokens))
            scored.append(SearchHit(path, name, normalized, _make_snippet(content, query_terms)))

        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]
