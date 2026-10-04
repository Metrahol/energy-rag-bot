"""Конвертация Markdown в безопасный HTML Telegram и нарезка сообщений."""

from __future__ import annotations

import html
import re

TG_LIMIT = 4000  # Лимит Telegram — 4096 символов

CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\n?(.*?)```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
HEADING_RE = re.compile(r"^#{1,6}\s*(.+)$", re.MULTILINE)
BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
LIST_ITEM_RE = re.compile(r"^\s*[-*]\s+", re.MULTILINE)
ITALIC_RE = re.compile(r"(?<![\w*])\*(?!\s)([^*\n]+?)\*(?!\w)")


def md_to_tg_html(md: str) -> str:
    """Безопасная конвертация Markdown в подмножество HTML для Telegram."""
    # 1. Экранируем спецсимволы (<, >, &), чтобы они не конфликтовали с HTML-тегами
    text = html.escape(md, quote=False)

    # 2. Блоки кода и инлайн-код
    text = CODE_BLOCK_RE.sub(r"<pre>\1</pre>", text)
    text = INLINE_CODE_RE.sub(r"<code>\1</code>", text)

    # 3. Маркеры списков меняем ДО курсива, чтобы звёздочки списков не путались с *курсивом*
    text = LIST_ITEM_RE.sub("• ", text)

    # 4. Заголовки и форматирование текста
    text = HEADING_RE.sub(r"<b>\1</b>", text)
    text = BOLD_RE.sub(r"<b>\1</b>", text)
    text = ITALIC_RE.sub(r"<i>\1</i>", text)

    return text


def split_message(text: str, limit: int = TG_LIMIT) -> list[str]:
    """Разбивает длинный текст по границам строк/абзацев с лимитом символов."""
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    current_chunk = ""

    for line in text.splitlines(keepends=True):
        # Если отдельная строка сама по себе больше лимита — режем её принудительно
        while len(line) > limit:
            if current_chunk:
                parts.append(current_chunk.strip())
                current_chunk = ""
            parts.append(line[:limit])
            line = line[limit:]

        if len(current_chunk) + len(line) > limit:
            parts.append(current_chunk.strip())
            current_chunk = ""

        current_chunk += line

    if current_chunk.strip():
        parts.append(current_chunk.strip())

    return parts