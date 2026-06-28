"""Abstract contract test suite — same tests run against both repo implementations."""

from __future__ import annotations

from uuid import uuid4

from app.chat.domain import ChatMessage


# ── helpers ───────────────────────────────────────────────────────────────────

async def _make_chat(repo, **kwargs):
    return await repo.create_chat(
        owner_external_id=kwargs.get("owner_external_id", "user-1"),
        interface=kwargs.get("interface", "cli"),
        system_prompt=kwargs.get("system_prompt", None),
    )


# ── contract tests ────────────────────────────────────────────────────────────

async def test_create_and_get_chat(repo):
    chat = await _make_chat(repo, owner_external_id="tg-123", interface="telegram")
    fetched = await repo.get_chat(chat.id)
    assert fetched is not None
    assert fetched.id == chat.id
    assert fetched.owner_external_id == "tg-123"
    assert fetched.interface == "telegram"


async def test_get_chat_unknown_returns_none(repo):
    result = await repo.get_chat(uuid4())
    assert result is None


async def test_append_and_list_messages_chronological_order(repo):
    chat = await _make_chat(repo)
    msg1 = ChatMessage(chat_id=chat.id, role="user", content="hello")
    msg2 = ChatMessage(chat_id=chat.id, role="assistant", content="hi there")
    msg3 = ChatMessage(chat_id=chat.id, role="user", content="how are you?")

    await repo.append_message(chat.id, msg1)
    await repo.append_message(chat.id, msg2)
    await repo.append_message(chat.id, msg3)

    messages = await repo.list_messages(chat.id)
    assert len(messages) == 3
    assert messages[0].content == "hello"
    assert messages[1].content == "hi there"
    assert messages[2].content == "how are you?"


async def test_list_messages_limit_returns_last_n(repo):
    chat = await _make_chat(repo)
    for i in range(5):
        msg = ChatMessage(chat_id=chat.id, role="user", content=f"msg {i}")
        await repo.append_message(chat.id, msg)

    messages = await repo.list_messages(chat.id, limit=3)
    assert len(messages) == 3
    assert messages[-1].content == "msg 4"
    assert messages[0].content == "msg 2"


async def test_soft_delete_clears_history(repo):
    chat = await _make_chat(repo)
    msg = ChatMessage(chat_id=chat.id, role="user", content="secret")
    await repo.append_message(chat.id, msg)

    await repo.soft_delete_messages(chat.id)
    messages = await repo.list_messages(chat.id)
    assert messages == []


async def test_new_messages_visible_after_soft_delete(repo):
    chat = await _make_chat(repo)
    old = ChatMessage(chat_id=chat.id, role="user", content="old message")
    await repo.append_message(chat.id, old)

    await repo.soft_delete_messages(chat.id)

    new = ChatMessage(chat_id=chat.id, role="user", content="fresh start")
    await repo.append_message(chat.id, new)

    messages = await repo.list_messages(chat.id)
    assert len(messages) == 1
    assert messages[0].content == "fresh start"


async def test_list_messages_unknown_chat_returns_empty(repo):
    messages = await repo.list_messages(uuid4())
    assert messages == []
