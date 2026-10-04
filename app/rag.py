"""Полный RAG-пайплайн: переписать запрос → гибридный поиск → ответ эксперта + источники.

Используется и ботом, и скриптом оценки (tests/run_eval.py).
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

from app import llm
from app.retriever import Hit, Retriever

log = logging.getLogger("rag")
CITE_RE = re.compile(r"\[(\d{1,2})\]")


@dataclass
class RagResult:
    answer: str
    sources: list[str]          # уникальные «Документ, с. N», реально процитированные в ответе
    queries: dict
    hits: list[Hit]


def cited_sources(answer: str, hits: list[Hit]) -> list[str]:
    nums = [int(n) for n in CITE_RE.findall(answer)]
    seen, out = set(), []
    for n in nums:
        if 1 <= n <= len(hits):
            c = hits[n - 1].citation
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out


async def ask(api_key: str, history: list[dict], question: str, retriever: Retriever) -> RagResult:
    q = await llm.rewrite_query(api_key, history, question)
    hits: list[Hit] = []
    if q["search"]:
        queries = [q["ru"], q["en"], question]
        # эмбеддинг на CPU — блокирующая операция, выносим из event loop
        hits = await asyncio.to_thread(retriever.search, queries)
    log.info("queries=%s hits=%s", q, [h.citation for h in hits])
    text = await llm.answer(api_key, history, question, hits)
    return RagResult(text, cited_sources(text, hits), q, hits)
