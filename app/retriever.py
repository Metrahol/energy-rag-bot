"""Гибридный поиск: векторный (ChromaDB) + лексический (BM25) через RRF."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import chromadb
from rank_bm25 import BM25Okapi

from app.config import settings
from app.embeddings import Embedder, tokenize
from app.ingest import CHUNKS_FILE, COLLECTION_NAME

RRF_K = 60


@dataclass
class Hit:
    id: str
    text: str
    doc_short: str
    page: int
    section: str
    kind: str
    score: float

    @property
    def citation(self) -> str:
        return f"{self.doc_short}, с. {self.page}"


class Retriever:

    def __init__(self):
        if not CHUNKS_FILE.exists():
            raise RuntimeError(
                f"Индекс не найден в {CHUNKS_FILE}. Сначала запустите: python -m app.ingest"
            )

        self.records: dict[str, dict] = {}
        order: list[str] = []

        with CHUNKS_FILE.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                self.records[r["id"]] = r
                order.append(r["id"])

        self.bm25_ids = order

        # Индексируем связку раздел + текст для лучшей точности BM25
        corpus = [
            tokenize(f"{self.records[i]['section']} {self.records[i]['text']}")
            for i in order
        ]
        self.bm25 = BM25Okapi(corpus)

        client = chromadb.PersistentClient(
            path=str(settings.storage_dir / "chroma")
        )
        self.col = client.get_collection(COLLECTION_NAME)

        indexed_model = (self.col.metadata or {}).get("embed_model")
        if indexed_model and indexed_model != settings.embed_model:
            raise RuntimeError(
                f"Индекс создан моделью '{indexed_model}', а в настройках '{settings.embed_model}'. "
                "Требуется переиндексация: python -m app.ingest --reparse"
            )

        self.embedder = Embedder()

    def _vector_ranks(self, queries: list[str]) -> list[list[str]]:
        vecs = self.embedder.embed_queries(queries)
        # include=[] экономит память и время: текст мы берем из self.records
        res = self.col.query(
            query_embeddings=vecs,
            n_results=settings.candidates_k,
            include=[],
        )
        return res["ids"]

    def _bm25_ranks(self, query: str) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        top_indices = sorted(
            range(len(scores)), key=scores.__getitem__, reverse=True
        )[: settings.candidates_k]
        return [self.bm25_ids[i] for i in top_indices if scores[i] > 0]

    def search(
        self, queries: list[str] | str, top_k: int | None = None
    ) -> list[Hit]:
        """Гибридный поиск по одному или нескольким запросам (RU/EN, синонимы)."""
        if isinstance(queries, str):
            queries = [queries]

        clean_queries = [
            q for q in dict.fromkeys(q.strip() for q in queries) if q
        ]
        if not clean_queries:
            return []

        # Собираем ранги от всех запросов (векторные + лексические)
        rankings = self._vector_ranks(clean_queries) + [
            self._bm25_ranks(q) for q in clean_queries
        ]

        # Reciprocal Rank Fusion (слияние рангов)
        fused_scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, chunk_id in enumerate(ranking):
                fused_scores[chunk_id] = fused_scores.get(
                    chunk_id, 0.0
                ) + 1.0 / (RRF_K + rank + 1)

        limit = top_k or settings.top_k
        best_chunks = sorted(
            fused_scores.items(), key=lambda x: x[1], reverse=True
        )[:limit]

        hits: list[Hit] = []
        for chunk_id, score in best_chunks:
            r = self.records[chunk_id]
            hits.append(
                Hit(
                    id=chunk_id,
                    text=r["text"],
                    doc_short=r["doc_short"],
                    page=r["page"],
                    section=r["section"],
                    kind=r["kind"],
                    score=score,
                )
            )
        return hits


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    query_args = sys.argv[1:] or [
        "прогноз потребления электроэнергии ЕЭС России до 2030"
    ]
    retriever = Retriever()
    for h in retriever.search(query_args):
        print(f"\n=== [{h.score:.4f}] {h.citation} | {h.kind} | {h.section[:80]}")
        print(h.text[:600])