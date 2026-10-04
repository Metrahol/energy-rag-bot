"""Прогон тестовых вопросов через полный RAG-пайплайн (без Telegram).

Запуск:  python -m tests.run_eval            → tests/eval_results.md
Нужен DEEPSEEK_TOKEN в .env.
"""
import asyncio
import sys
import time

from app import rag
from app.config import ROOT, settings
from app.retriever import Retriever

# (вопрос, продолжает ли он предыдущий диалог)
QUESTIONS = [
    ("Какой прогноз потребления электроэнергии в ЕЭС России на 2030 год?", False),
    ("А какой при этом ожидается максимум потребления мощности?", True),
    ("Какие энергосистемы России названы территориями с дефицитом мощности?", False),
    ("Сколько солнечных и ветровых электростанций планируется ввести в ЕЭС до 2030 года?", False),
    ("Как росло мировое потребление электроэнергии в 2024 году и какой прогноз до 2027?", False),
    ("Какую роль играют дата-центры в росте спроса на электроэнергию?", False),
    ("Что происходило с выбросами CO2 в мире в 2024 году?", False),
    ("Как изменился мировой спрос на энергию в 2024 году по видам топлива?", False),
    ("Какая доля атомной генерации в мире и какие перспективы у АЭС?", False),
    ("Какие мероприятия предусмотрены для энергосистемы Юга России?", False),
    ("Сравни темпы роста спроса на электроэнергию в Китае, Индии и США.", False),
    ("Какой курс биткоина будет в 2030 году?", False),  # вне базы — бот должен честно сказать
]


async def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if not settings.deepseek_token:
        raise SystemExit("Задайте DEEPSEEK_TOKEN в .env")
    retriever = Retriever()
    out = ["# Результаты прогона тестовых вопросов\n", f"Модель: `{settings.deepseek_model}`, эмбеддинги: `{settings.embed_model}`\n"]
    history: list[dict] = []
    for i, (q, follow_up) in enumerate(QUESTIONS, 1):
        if not follow_up:
            history = []
        t = time.time()
        res = await rag.ask(settings.deepseek_token, history, q, retriever)
        dt = time.time() - t
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": res.answer}]
        print(f"[{i}/{len(QUESTIONS)}] {dt:.1f}s  {q}")
        out += [
            f"\n---\n\n## {i}. {q}\n",
            f"*Переписанный запрос:* RU: `{res.queries['ru']}` | EN: `{res.queries['en']}` | {dt:.1f} c\n",
            f"*Найдено:* {', '.join(h.citation for h in res.hits) or '—'}\n",
            f"\n{res.answer}\n",
            f"\n**Источники:** {'; '.join(res.sources) or '—'}\n",
        ]
    path = ROOT / "tests" / "eval_results.md"
    path.write_text("".join(out), encoding="utf-8")
    print("Сохранено:", path)


if __name__ == "__main__":
    asyncio.run(main())
