from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender
import openai

from app import llm, rag
from app.config import settings
from app.retriever import Retriever
from app.tg_format import md_to_tg_html, split_message

log = logging.getLogger("bot")
router = Router()


class Dialog(StatesGroup):
    waiting_key = State()
    chatting = State()


ASK_KEY_TEXT = (
    "⚡ <b>Эксперт по электроэнергетике</b>\n\n"
    "База знаний: СиПР ЕЭС России 2025–2030, отчёты IEA "
    "(Electricity 2025, Global Energy Review 2025).\n\n"
    "Для работы отправьте ваш <b>API-ключ DeepSeek</b> (формата <code>sk-...</code>). "
    "Получить его можно в личном кабинете platform.deepseek.com.\n\n"
    "🔒 Сообщение с ключом будет сразу удалено из соображений безопасности."
)

HELP_TEXT = (
    "<b>Доступные команды:</b>\n"
    "/start — перезапустить бота\n"
    "/key — сменить API-ключ\n"
    "/reset — сбросить контекст беседы\n"
    "/help — справка\n\n"
    "Пример вопроса: <i>«Какой прогноз ввода мощностей СЭС в ОЭС Юга к 2030 году?»</i>"
)


def api_error_text(e: Exception) -> str:
    """Формирует понятное пользователю сообщение об ошибке API."""
    if isinstance(e, openai.AuthenticationError):
        return "🔑 Неверный или отозванный ключ DeepSeek. Отправьте действующий ключ."
    if isinstance(e, openai.RateLimitError):
        return "⏳ Превышен лимит запросов к API. Попробуйте через минуту."
    if isinstance(e, openai.APIStatusError) and e.status_code == 402:
        return "💳 На балансе DeepSeek закончились средства. Пополните счет или пришлите другой ключ (/key)."
    if isinstance(e, (openai.APITimeoutError, openai.APIConnectionError)):
        return "🌐 Серверы DeepSeek временно недоступны. Попробуйте повторить запрос позже."
    return "⚠️ Произошла ошибка при обращении к языковой модели. Попробуйте еще раз."


async def send_long(message: Message, md_text: str):
    """Отправка ответа с разбиением на части по лимитам длины Telegram."""
    for part in split_message(md_to_tg_html(md_text)):
        await message.answer(part)


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    await state.set_state(Dialog.waiting_key)
    await message.answer(ASK_KEY_TEXT)


@router.message(Command("key"))
async def cmd_key(message: Message, state: FSMContext):
    await state.set_state(Dialog.waiting_key)
    await message.answer("Отправьте новый API-ключ DeepSeek.")


@router.message(Command("reset"))
async def cmd_reset(message: Message, state: FSMContext):
    await state.update_data(history=[])
    await message.answer("🧹 История диалога очищена. Задайте новый вопрос.")


@router.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(HELP_TEXT)


@router.message(Dialog.waiting_key, F.text)
async def on_key(message: Message, state: FSMContext):
    key = message.text.strip()

    # Стираем сообщение с ключом из чата
    try:
        await message.delete()
    except Exception:
        pass

    if not key.startswith("sk-") or len(key) < 20 or " " in key:
        await message.answer(
            "Некорректный формат ключа (ключ должен начинаться с <code>sk-</code>). Попробуйте ещё раз."
        )
        return

    status = await message.answer("🔄 Проверка ключа...")
    try:
        async with ChatActionSender.typing(
            bot=message.bot, chat_id=message.chat.id
        ):
            greeting = await llm.greet(
                key, message.from_user.first_name or "коллега"
            )
    except Exception as e:
        log.warning("Ошибка проверки ключа: %r", e)
        error_msg = api_error_text(e)
        if not isinstance(e, openai.AuthenticationError):
            error_msg += "\nПопробуйте отправить ключ повторно."
        await status.edit_text(error_msg)
        return

    await state.update_data(api_key=key, history=[])
    await state.set_state(Dialog.chatting)
    await status.edit_text("✅ Ключ успешно подтвержден.")
    await send_long(message, greeting)


@router.message(Dialog.chatting, F.text)
async def on_question(
    message: Message, state: FSMContext, retriever: Retriever
):
    data = await state.get_data()
    key = data.get("api_key")

    # Защита на случай потери сессии/перезапуска хранилища FSM
    if not key:
        await cmd_start(message, state)
        return

    history = data.get("history", [])

    try:
        async with ChatActionSender.typing(
            bot=message.bot, chat_id=message.chat.id
        ):
            result = await rag.ask(key, history, message.text, retriever)
    except openai.AuthenticationError as e:
        await state.set_state(Dialog.waiting_key)
        await message.answer(api_error_text(e))
        return
    except openai.OpenAIError as e:
        log.exception("Ошибка при обращении к DeepSeek")
        await message.answer(api_error_text(e))
        return

    response_text = result.answer
    if result.sources:
        response_text += "\n\n📚 <b>Источники:</b> " + "; ".join(result.sources)

    await send_long(message, response_text)

    # Сохраняем диалог в истории 
    history.extend(
        [
            {"role": "user", "content": message.text},
            {"role": "assistant", "content": result.answer},
        ]
    )
    max_history_len = settings.history_turns * 2
    await state.update_data(history=history[-max_history_len:])


@router.message(F.text)
async def on_no_state(message: Message, state: FSMContext):
    """Срабатывает, если бот был перезапущен и состояние FSM сбросилось."""
    await cmd_start(message, state)


@router.message()
async def on_unsupported_content(message: Message):
    await message.answer("Я поддерживаю только текстовые сообщения.")