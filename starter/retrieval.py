"""Local-file dense, TF-IDF, and RRF retrieval over converted Day 6 chunks."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .baseline import TfidfIndex
from .day06_starter.src.embeddings import MyEmbedder


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON array or JSONL file of objects."""
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{path}: expected a JSON array or JSONL objects")
    return value


class Day6Retriever:
    """Search one local converted chunk file with three ranking methods.

    Dense vectors are embedded from the file on initialization. The caller must
    configure an embedding provider and use the same model used for queries.
    """

    def __init__(self, method: str, chunks_path: str | Path = "starter/data/converted/chunks.json",
                 strategy: str = "B_paragraph", *, embedder: Any | None = None,
                 chunks: list[dict[str, Any]] | None = None,
                 document_vectors: list[list[float]] | None = None) -> None:
        if method not in {"dense", "tfidf", "hybrid"}:
            raise ValueError("method must be dense, tfidf, or hybrid")
        env_path = os.getenv("DAY6_ENV_FILE", ".env")
        load_dotenv(env_path)
        self.method = method
        self.strategy = strategy
        self.chunks = chunks if chunks is not None else read_records(chunks_path)
        if not self.chunks:
            raise ValueError(f"No chunks loaded from {chunks_path}")
        self.index = TfidfIndex(self.chunks)
        self.embedder = embedder
        self.vectors: list[list[float]] = []
        if method in {"dense", "hybrid"}:
            if self.embedder is None:
                model = os.getenv("EMBEDDING_MODEL", "").strip()
                if not model:
                    raise ValueError(f"Set EMBEDDING_MODEL in {env_path}")
                self.embedder = MyEmbedder(model)
            if document_vectors is not None:
                self.vectors = document_vectors
            else:
                for start in range(0, len(self.chunks), 64):
                    batch = self.chunks[start:start + 64]
                    self.vectors.extend(self.embedder.embed_documents([c["text"] for c in batch]))
            if len(self.vectors) != len(self.chunks) or any(
                    len(v) != self.embedder.dimensions for v in self.vectors):
                raise ValueError("Document embedding count/dimension does not match chunks/model")

    @staticmethod
    def _matches_filters(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
        metadata = chunk.get("metadata") or {}
        for key, value in (filters or {}).items():
            actual = chunk.get(key, metadata.get(key))
            if str(actual) != str(value):
                return False
        return True

    def _eligible(self, filters: dict[str, Any] | None) -> list[int]:
        return [i for i, c in enumerate(self.chunks) if self._matches_filters(c, filters)]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    def _dense(self, query: str, top_k: int, filters: dict[str, Any] | None):
        assert self.embedder is not None
        vector = self.embedder.embed_query(query)
        rows = [{**self.chunks[i], "score": self._cosine(vector, self.vectors[i]), "strategy": "dense"}
                for i in self._eligible(filters)]
        return sorted(rows, key=lambda r: (-r["score"], r["id"]))[:top_k]

    def _lexical(self, query: str, top_k: int, filters: dict[str, Any] | None):
        rows = self.index.search(query, top_k=len(self.chunks))
        return [row for row in rows if self._matches_filters(row, filters)][:top_k]

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 3) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.method == "dense":
            return self._dense(query, top_k, filters)
        if self.method == "tfidf":
            return self._lexical(query, top_k, filters)
        depth = max(20, top_k)
        dense = self._dense(query, depth, filters)
        lexical = self._lexical(query, depth, filters)
        combined: dict[str, dict[str, Any]] = {}
        for name, ranking in (("dense", dense), ("tfidf", lexical)):
            for rank, row in enumerate(ranking, start=1):
                item = combined.setdefault(row["id"], {**row, "score": 0.0, "rrf_ranks": {}})
                item["score"] += 1.0 / (60 + rank)
                item["rrf_ranks"][name] = rank
        return sorted(combined.values(), key=lambda r: (-r["score"], r["id"]))[:top_k]
