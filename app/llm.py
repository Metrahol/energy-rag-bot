"""Интеграция с LLM (DeepSeek / OpenAI API): переписывание запросов и генерация ответа."""

from __future__ import annotations

from datetime import date
import json
import logging

from openai import AsyncOpenAI

from app.config import settings
from app.retriever import Hit

log = logging.getLogger("llm")

BASE_SYSTEM_PROMPT = """Ты — ведущий отраслевой эксперт в области электроэнергетики с 20-летним опытом: \
балансы мощности, генерация (ТЭС, АЭС, ГЭС, ВИЭ), электросети, рынки электроэнергии и мировые тренды.

Твоя база знаний — фрагменты трёх документов в блоке КОНТЕКСТ:
• СиПР ЕЭС России на 2025–2030 гг. (Системный оператор) — на русском;
• IEA Electricity 2025 — на английском;
• IEA Global Energy Review 2025 — на английском.

Правила:
1. Отвечай на языке пользователя (по умолчанию — русский). Переводи англоязычные данные, \
сохраняя отраслевые единицы (TWh → ТВт·ч, GW → ГВт, Mt CO2 → млн т CO2).
2. Факты и цифры подтверждай ссылками на фрагменты в квадратных скобках: [1], [2][4]. Не выдумывай факты.
3. Если данных в контексте недостаточно — прямо скажи об этом («в материалах базы знаний данных нет»), \
после чего можешь дать экспертный комментарий из общих знаний, явно это обозначив.
4. Отвечай структурно: ключевые тезисы, интерпретация цифр для отрасли, расшифровка аббревиатур при первом упоминании.
5. Форматирование для Telegram: короткие абзацы, списки «•», **жирный шрифт** для цифр. \
Без таблиц Markdown и заголовков #. Объем: 100–250 слов."""

REWRITE_PROMPT = """Ты оптимизируешь поисковые запросы по отраслевой базе документов \
(СиПР ЕЭС России 2025–2030 на русском; IEA Electricity 2025 и Global Energy Review 2025 на английском).

По истории диалога и новому вопросу верни строго JSON:
{
  "search": true/false,
  "ru": "самостоятельный поисковый запрос на русском с ключевыми терминами",
  "en": "the same standalone search query in English"
}
Поле search=false ставится только для приветствий, благодарностей и оффтопа. \
Раскрывай контекст местоимений (например, «а в Сибири?» -> «баланс мощности ОЭС Сибири 2025-2030»)."""

_clients: dict[str, AsyncOpenAI] = {}


def get_system_prompt() -> str:
    """Генерирует системный промпт с актуальной сегодняшней датой."""
    return f"{BASE_SYSTEM_PROMPT}\n\nТекущая дата: {date.today():%d.%m.%Y}."


def get_client(api_key: str) -> AsyncOpenAI:
    """Пул клиентов для переиспользования HTTP-соединений."""
    if api_key not in _clients:
        _clients[api_key] = AsyncOpenAI(
            api_key=api_key,
            base_url=settings.deepseek_base_url,
            timeout=120,
            max_retries=2,
        )
    return _clients[api_key]


async def greet(api_key: str, user_name: str) -> str:
    """Приветствие эксперта. Валидирует API-ключ при первом обращении."""
    client = get_client(api_key)
    resp = await client.chat.completions.create(
        model=settings.deepseek_model,
        messages=[
            {"role": "system", "content": get_system_prompt()},
            {
                "role": "user",
                "content": (
                    f"Поприветствуй пользователя {user_name} (2–3 предложения): представься экспертом-энергетиком, "
                    "укажи доступные документы и предложи 3 примера вопросов списком."
                ),
            },
        ],
        temperature=0.6,
        max_tokens=400,
    )
    return resp.choices[0].message.content or ""


async def rewrite_query(
    api_key: str, history: list[dict], question: str
) -> dict:
    """Анализ контекста и генерация двуязычного поискового запроса (RU + EN)."""
    client = get_client(api_key)
    recent_history = "\n".join(
        f"{m['role']}: {m['content'][:300]}" for m in history[-4:]
    )

    try:
        resp = await client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[
                {"role": "system", "content": REWRITE_PROMPT},
                {
                    "role": "user",
                    "content": f"Контекст беседы:\n{recent_history or '—'}\n\nНовый вопрос: {question}",
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=300,
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        return {
            "search": bool(data.get("search", True)),
            "ru": data.get("ru") or question,
            "en": data.get("en") or "",
        }
    except Exception as e:
        log.warning("Сбой переписывания запроса (%s). Поиск по сырому вопросу.", e)
        return {"search": True, "ru": question, "en": ""}


def format_context(hits: list[Hit]) -> str:
    """Форматирование чанков в блок контекста с номерами цитат."""
    parts = []
    for i, h in enumerate(hits, 1):
        section_info = f" | Раздел: {h.section}" if h.section else ""
        parts.append(f"[{i}] ({h.citation}{section_info})\n{h.text}")
    return "\n\n".join(parts)


async def answer(
    api_key: str, history: list[dict], question: str, hits: list[Hit]
) -> str:
    """Генерация финального экспертного ответа по найденному контексту."""
    client = get_client(api_key)
    context_str = (
        format_context(hits) if hits else "(поиск по документам не выполнялся)"
    )

    # Ограничиваем историю диалога последними 6 репликами во избежание переполнения контекста
    messages = [
        {"role": "system", "content": get_system_prompt()},
        *history[-6:],
        {
            "role": "user",
            "content": f"КОНТЕКСТ ДЛЯ ОТВЕТА:\n{context_str}\n\nВОПРОС: {question}",
        },
    ]

    resp = await client.chat.completions.create(
        model=settings.deepseek_model,
        messages=messages,
        temperature=0.2,
        max_tokens=1500,
    )
    return resp.choices[0].message.content or ""