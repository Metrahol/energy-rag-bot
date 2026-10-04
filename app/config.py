"""Централизованная конфигурация приложения (читается из .env)."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # Telegram
    bot_token: str = ""

    # DeepSeek: ключ пользователь вводит в боте; этот используется только для локальных тестов (eval)
    deepseek_token: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # Пути
    data_dir: Path = ROOT / "data"
    storage_dir: Path = ROOT / "storage"

    # Эмбеддинги / поиск
    embed_model: str = "intfloat/multilingual-e5-base"
    chunk_size: int = 1200          # символов на чанк (≈300–400 токенов)
    chunk_overlap: int = 150
    top_k: int = 8                  # сколько чанков отдаём в LLM
    candidates_k: int = 30          # кандидатов из каждого поисковика до слияния

    # Диалог
    history_turns: int = 6          # сколько последних пар «вопрос-ответ» помним


settings = Settings()

# Человекочитаемые названия документов для цитирования
DOCUMENTS = {
    "sipr_ups_2025-30_fin.pdf": {
        "title": "СиПР ЕЭС России 2025–2030 (СО ЕЭС)",
        "short": "СиПР ЕЭС 2025–2030",
        "lang": "ru",
    },
    "Electricity2025.pdf": {
        "title": "IEA Electricity 2025",
        "short": "IEA Electricity 2025",
        "lang": "en",
    },
    "GlobalEnergyReview2025.pdf": {
        "title": "IEA Global Energy Review 2025",
        "short": "IEA Global Energy Review 2025",
        "lang": "en",
    },
}
