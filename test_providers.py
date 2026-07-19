"""Live STT/TTS provider-adapter checks (needs DEEPGRAM_API_KEY, OPENAI_API_KEY,
CARTESIA_API_KEY, OPENROUTER_API_KEY, SARVAM_API_KEY, GNANI_API_KEY in .env — skips a
provider cleanly if its key is absent). Exercises the real models.transcribe()/speak() dispatchers with settings
switched per provider, then restores defaults.

Usage: python test_providers.py
"""

import asyncio
import os

os.environ["SETTINGS_PATH"] = "settings_test_providers.json"

from dotenv import load_dotenv

load_dotenv()

import models
import settings
from audio import pcm_to_wav, ulaw_to_pcm

SENT = "I have five years of experience with Python and Go."
EXPECT_WORDS = ("python", "go")


def cleanup():
    settings.reset()
    for p in ("settings_test_providers.json",):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


async def check_pair(stt_id: str, tts_id: str) -> None:
    settings.update({"stt_provider": stt_id, "tts_provider": tts_id})
    assert settings.stt_provider() == stt_id and settings.tts_provider() == tts_id
    ulaw = await models.speak(SENT)
    assert len(ulaw) > 8000, f"{tts_id}: implausibly little audio ({len(ulaw)} bytes)"
    stt = await models.transcribe(pcm_to_wav(ulaw_to_pcm(ulaw), 8000))
    text = stt["text"].lower()
    assert all(w in text for w in EXPECT_WORDS), f"{stt_id} misheard {tts_id}: {stt['text']!r}"
    assert stt.get("seconds"), f"{stt_id}: no duration returned"
    print(f"  {tts_id} -> {stt_id}: {stt['text']!r} ({len(ulaw)} ulaw bytes, {stt['seconds']:.1f}s)")


async def main():
    pairs = [
        ("openrouter-whisper", "cartesia-sonic"),  # default path
        ("deepgram-nova", "openai-tts"),           # Phase 2 adapters
        ("sarvam-saaras", "sarvam-bulbul"),        # Phase 4 adapters
        ("gnani-vachana", "gnani-vachana"),
    ]
    for stt_id, tts_id in pairs:
        missing = [
            p["key_env"]
            for vetted, pid in ((settings.VETTED_STT_PROVIDERS, stt_id), (settings.VETTED_TTS_PROVIDERS, tts_id))
            for p in vetted
            if p["id"] == pid and not os.environ.get(p["key_env"], "")
        ]
        if missing:
            print(f"  SKIP {stt_id}/{tts_id}: missing {', '.join(missing)}")
            continue
        await check_pair(stt_id, tts_id)
    cleanup()
    assert settings.stt_provider() == "openrouter-whisper"
    print("Provider adapters round-trip real speech correctly; defaults restored — OK")


try:
    asyncio.run(main())
finally:
    cleanup()
