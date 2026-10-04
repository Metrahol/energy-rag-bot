from __future__ import annotations

from functools import lru_cache
import re

from sentence_transformers import SentenceTransformer
import snowballstemmer
import torch

from app.config import settings


class Embedder:
    def __init__(
        self,
        model_name: str = settings.embed_model,
        device: str | None = None,
    ):
        self.model_name = model_name

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = SentenceTransformer(model_name, device=self.device)
        self.is_e5 = "e5" in model_name.lower()

    def _prep(self, texts: list[str], prefix: str) -> list[str]:
        return [f"{prefix}{t}" for t in texts] if self.is_e5 else texts

    def embed_passages(
        self, texts: list[str], batch_size: int = 16
    ) -> list[list[float]]:
        vecs = self.model.encode(
            self._prep(texts, "passage: "),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.tolist()

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(
            self._prep(texts, "query: "), normalize_embeddings=True
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_queries([text])[0]


# ---------- BM25 токенизация (RU + EN со стеммингом) ----------
_ru = snowballstemmer.stemmer("russian")
_en = snowballstemmer.stemmer("english")

_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+(?:[.,][0-9]+)?", re.IGNORECASE)
_CYR_RE = re.compile(r"[а-яё]")


@lru_cache(maxsize=200_000)
def _stem(word: str) -> str:
    return _ru.stemWord(word) if _CYR_RE.search(word) else _en.stemWord(word)


def tokenize(text: str) -> list[str]:
    return [
        _stem(token)
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) > 1 or token.isdigit()
    ]