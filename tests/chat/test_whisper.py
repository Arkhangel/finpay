"""Tests for the Whisper-1 speech-to-text path in app/chat/media.py."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

from fastapi import UploadFile
from starlette.datastructures import Headers

from app.chat.media import media_to_part, whisper_transcribe


def _make_client(transcript: str) -> MagicMock:
    client = MagicMock()
    client.audio.transcriptions.create = AsyncMock(
        return_value=MagicMock(text=transcript)
    )
    return client


async def test_whisper_transcribe_calls_openai_and_returns_text():
    client = _make_client("привет из голосового")

    text = await whisper_transcribe(client, b"fake-ogg-bytes", "voice.ogg")

    assert text == "привет из голосового"
    client.audio.transcriptions.create.assert_awaited_once()


async def test_media_to_part_audio_ogg_returns_prefixed_text_part():
    client = _make_client("расшифровка голоса")
    upload = UploadFile(
        file=BytesIO(b"fake-ogg-bytes"),
        filename="voice.ogg",
        headers=Headers({"content-type": "audio/ogg"}),
    )

    part = await media_to_part(upload, client)

    assert part == {
        "type": "text",
        "text": "[пользователь сказал голосом]:\nрасшифровка голоса",
    }
