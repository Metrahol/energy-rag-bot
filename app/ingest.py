"""Индексация документов: PDF → Markdown → чанки → эмбеддинги → ChromaDB.

Запуск:
    python -m app.ingest
    python -m app.ingest --reparse
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import re
import time

import chromadb
import pymupdf4llm

from app.chunker import chunk_page
from app.config import DOCUMENTS, settings
from app.embeddings import Embedder

log = logging.getLogger("ingest")

MD_CACHE = settings.storage_dir / "md_cache"
CHUNKS_FILE = settings.storage_dir / "chunks.jsonl"
COLLECTION_NAME = "energy_docs"
PAGE_NUM_RE = re.compile(r"^\s*(page\s*\|?\s*)?\d{1,3}\s*$", re.IGNORECASE)


def parse_pdf(filename: str, reparse: bool) -> list[str]:
    """Парсинг PDF в Markdown по страницам с кэшированием."""
    cache_file = MD_CACHE / f"{filename}.json"
    if cache_file.exists() and not reparse:
        return json.loads(cache_file.read_text(encoding="utf-8"))

    t0 = time.time()
    log.info("Парсинг %s...", filename)
    pdf_path = str(settings.data_dir / filename)
    pages = pymupdf4llm.to_markdown(
        pdf_path, page_chunks=True, show_progress=False
    )
    texts = [p["text"] for p in pages]

    MD_CACHE.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(texts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("  %d стр. за %.1f c", len(texts), time.time() - t0)
    return texts


def strip_boilerplate(pages: list[str]) -> list[str]:
    """Удаление повторяющихся колонтитулов и номеров страниц."""
    counter = Counter()
    for p in pages:
        lines_on_page = {
            ln.strip() for ln in p.split("\n") if 0 < len(ln.strip()) < 80
        }
        counter.update(lines_on_page)

    threshold = max(5, int(len(pages) * 0.25))
    frequent = {
        ln
        for ln, count in counter.items()
        if count >= threshold and not ln.startswith("|")
    }

    cleaned_pages = []
    for p in pages:
        good_lines = [
            ln
            for ln in p.split("\n")
            if ln.strip() not in frequent and not PAGE_NUM_RE.match(ln)
        ]
        cleaned_pages.append("\n".join(good_lines))

    return cleaned_pages


def build_chunks(reparse: bool) -> list[dict]:
    records = []
    for filename, doc in DOCUMENTS.items():
        pages = strip_boilerplate(parse_pdf(filename, reparse))
        section = ""
        n_before = len(records)

        for page_no, md in enumerate(pages, start=1):
            chunks, section = chunk_page(
                md=md,
                page=page_no,
                section=section,
                size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )
            for i, ch in enumerate(chunks):
                records.append(
                    {
                        "id": f"{filename}:{page_no}:{i}",
                        "text": ch.text,
                        "doc": filename,
                        "doc_title": doc["title"],
                        "doc_short": doc["short"],
                        "lang": doc["lang"],
                        "page": page_no,
                        "section": ch.section,
                        "kind": ch.kind,
                    }
                )
        log.info("%s → %d чанков", filename, len(records) - n_before)
    return records


def embed_text(r: dict) -> str:
    """Контекстное обогащение текста чанка для эмбеддинга."""
    head = f"{r['doc_title']}. {r['section']}".strip(". ")
    return f"{head}\n{r['text']}"


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reparse", action="store_true", help="Игнорировать кэш Markdown"
    )
    args = parser.parse_args()

    settings.storage_dir.mkdir(parents=True, exist_ok=True)

    records = build_chunks(args.reparse)
    if not records:
        log.warning("Нет данных для индексации")
        return

    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Сохранено чанков: %d", len(records))

    t0 = time.time()
    embedder = Embedder()
    texts_to_embed = [embed_text(r) for r in records]
    vectors = embedder.embed_passages(texts_to_embed)
    log.info("Эмбеддинги рассчитаны за %.1f c", time.time() - t0)

    client = chromadb.PersistentClient(path=str(settings.storage_dir / "chroma"))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    col = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embed_model": settings.embed_model,
        },
    )

    batch_size = 2000
    for i in range(0, len(records), batch_size):
        part = records[i : i + batch_size]
        col.add(
            ids=[r["id"] for r in part],
            embeddings=vectors[i : i + batch_size],
            documents=[r["text"] for r in part],
            metadatas=[
                {
                    k: r[k]
                    for k in (
                        "doc",
                        "doc_short",
                        "lang",
                        "page",
                        "section",
                        "kind",
                    )
                }
                for r in part
            ],
        )

    log.info("Индексация завершена. Всего векторов: %d", col.count())


if __name__ == "__main__":
    main()