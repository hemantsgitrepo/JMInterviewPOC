"""Thin OpenRouter wrappers: STT, LLM turn, TTS, JD-based question generation.
Retries with backoff, longer on 429 (rate limit) since an instant retry almost always
hits the same limit window."""

import asyncio
import json
import logging
import os
import re

import httpx
from dotenv import load_dotenv

import settings
from audio import pcm_to_ulaw, resample

load_dotenv()

logger = logging.getLogger("models")

if os.environ.get("LANGSMITH_TRACING", "").lower() == "true":
    from langsmith import traceable
else:
    def traceable(*dargs, **dkwargs):
        """No-op stand-in: tracing is opt-in, so when disabled we never import langsmith
        or wrap the call — zero added latency/cold-start cost on the hot path."""
        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return dargs[0]  # bare @traceable usage
        return lambda f: f  # @traceable(name=..., run_type=...) usage

STT_MODEL = "openai/whisper-large-v3"  # has 2 backing providers (Together + Groq) on
# OpenRouter, unlike the "-turbo" variant which is Groq-only and hard-fails on Groq 429s
# LLM model is settings-driven (settings.llm_model()); default = google/gemini-2.5-flash
CARTESIA_MODEL = "sonic-3.5"
CARTESIA_VOICE_ID = "79a125e8-cd45-4c13-8a67-188112f4dd22"
CARTESIA_PCM_RATE = 24_000
DEEPGRAM_STT_MODEL = "nova-3"
OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
OPENAI_TTS_VOICE = "alloy"
OPENAI_PCM_RATE = 24_000

_client = httpx.AsyncClient(
    base_url="https://openrouter.ai/api/v1",
    headers={"Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY', '')}"},
    timeout=60,
)

_cartesia = httpx.AsyncClient(
    base_url="https://api.cartesia.ai",
    headers={
        "X-API-Key": os.environ.get("CARTESIA_API_KEY", ""),
        "Cartesia-Version": "2024-06-10",
    },
    timeout=45,
)

_deepgram = httpx.AsyncClient(
    base_url="https://api.deepgram.com",
    headers={"Authorization": f"Token {os.environ.get('DEEPGRAM_API_KEY', '')}"},
    timeout=60,
)

_openai = httpx.AsyncClient(
    base_url="https://api.openai.com",
    headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"},
    timeout=45,
)


async def _retry(fn, attempts: int = 3):
    """Retries on failure. A 429 (rate limit) waits with backoff before retrying, since an
    instant retry almost always lands in the same limit window; other errors retry once."""
    for i in range(attempts):
        try:
            return await fn()
        except httpx.HTTPStatusError as e:
            last_exc = e
            if e.response.status_code != 429 or i == attempts - 1:
                raise
            wait = 2 ** (i + 1)  # 2s, 4s, ...
            logger.warning("429 rate limited (attempt %d/%d), retrying in %ds", i + 1, attempts, wait)
            await asyncio.sleep(wait)
        except Exception as e:
            last_exc = e
            if i == attempts - 1:
                raise
    raise last_exc


@traceable(name="stt_transcribe", run_type="tool")
async def transcribe(wav: bytes) -> dict:
    """Returns {"text": str, "seconds": float|None, "cost": float|None}. Dispatches on
    the settings STT provider; the default path is the pre-settings code unchanged."""
    if settings.stt_provider() == "deepgram-nova":
        return await _transcribe_deepgram(wav)
    return await _transcribe_openrouter(wav)


async def _transcribe_openrouter(wav: bytes) -> dict:
    """OpenRouter Whisper — cost/seconds come straight from its usage block (real)."""

    async def go():
        r = await _client.post(
            "/audio/transcriptions",
            # language is pinned: auto-detect on 8kHz telephony audio occasionally lands on
            # the wrong language and returns a fluent-but-wrong transcript (seen once as Welsh),
            # which reaches the LLM as a garbled answer.
            data={"model": STT_MODEL, "language": "en"},
            files={"file": ("utterance.wav", wav, "audio/wav")},
        )
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage") or {}
        return {
            "text": data.get("text", "").strip(),
            "seconds": usage.get("seconds"),
            "cost": usage.get("cost"),
        }

    return await _retry(go)


async def _transcribe_deepgram(wav: bytes) -> dict:
    """Deepgram prerecorded API. Returns cost=None: Deepgram doesn't report cost per
    request and its per-minute rate isn't pinned in this repo yet (see docs/backlog.md),
    so the UI shows n/a rather than a made-up number."""

    async def go():
        r = await _deepgram.post(
            f"/v1/listen?model={DEEPGRAM_STT_MODEL}&language=en",
            content=wav,
            headers={"Content-Type": "audio/wav"},
        )
        r.raise_for_status()
        data = r.json()
        return {
            "text": data["results"]["channels"][0]["alternatives"][0]["transcript"].strip(),
            "seconds": data.get("metadata", {}).get("duration"),
            "cost": None,
        }

    return await _retry(go)


@traceable(name="llm_next_turn", run_type="llm")
async def next_turn(system: str, messages: list[dict]) -> tuple[dict, dict]:
    """Returns ({"reply": str, "action": ...}, usage) where usage is
    {"prompt_tokens": int, "completion_tokens": int, "cost": float|None} — cost is real,
    from OpenRouter's usage-accounting feature (`usage: {include: true}`), not estimated."""

    async def go():
        r = await _client.post(
            "/chat/completions",
            json={
                "model": settings.llm_model(),
                "messages": [{"role": "system", "content": system}] + messages,
                "response_format": {"type": "json_object"},
                "temperature": 0.4,
                "usage": {"include": True},
            },
        )
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        return _parse_json(content), {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cost": usage.get("cost"),
        }

    return await _retry(go)


# Phrases Whisper reliably hallucinates on noise / near-silence. A lone one of these is
# never a real interview answer; call.py discards them silently so a cough transcribed as
# "Bye" can't end the call.
HALLUCINATIONS = {
    "thank you", "thanks", "thank you very much", "thank you so much",
    "thank you for watching", "thanks for watching", "bye", "you", "the end",
}


def _parse_json(content: str) -> dict:
    content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", content, re.S)
    out = json.loads(m.group(0) if m else content)
    if out.get("action") not in ("stay", "ask_next", "skip", "repeat", "end_call"):
        out["action"] = "stay"
    out.setdefault("reply", "")
    return out


@traceable(name="tts_speak", run_type="tool")
async def speak(text: str) -> bytes:
    """Text -> 8kHz mu-law bytes ready for a Twilio media stream. Dispatches on the
    settings TTS provider; the default path is the pre-settings Cartesia code unchanged."""
    if settings.tts_provider() == "openai-tts":
        return await _speak_openai(text)
    return await _speak_cartesia(text)


async def _speak_cartesia(text: str) -> bytes:
    async def go():
        r = await _cartesia.post(
            "/tts/bytes",
            json={
                "transcript": text,
                "model_id": CARTESIA_MODEL,
                "voice": {"mode": "id", "id": CARTESIA_VOICE_ID},
                "output_format": {
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": CARTESIA_PCM_RATE,
                },
            },
        )
        r.raise_for_status()
        return r.content

    pcm = await _retry(go)
    return pcm_to_ulaw(resample(pcm, CARTESIA_PCM_RATE, 8000))


async def _speak_openai(text: str) -> bytes:
    async def go():
        r = await _openai.post(
            "/v1/audio/speech",
            json={
                "model": OPENAI_TTS_MODEL,
                "voice": OPENAI_TTS_VOICE,
                "input": text,
                "response_format": "pcm",  # 24kHz mono s16le
            },
        )
        r.raise_for_status()
        return r.content

    pcm = await _retry(go)
    return pcm_to_ulaw(resample(pcm, OPENAI_PCM_RATE, 8000))


JD_PARSE_SYSTEM_PROMPT = """You review a piece of text and decide whether it is a job description
(JD) — text describing an open role a company is hiring for (job title, responsibilities,
required skills/qualifications, or "about the role" language). If it is one, also extract
the hiring company's name and the role/job title exactly as stated — null when not stated.

Respond ONLY with JSON:
{"is_job_description": true|false, "reason": "<short reason, required if false, else empty>", "company_name": "<company name or null>", "role_name": "<job title or null>"}"""

QUESTION_GEN_SYSTEM_PROMPT = """You are given a job description. Write 5 to 8 clear interview
questions tailored to the specific responsibilities and skills in it. Each must be answerable
out loud in under a minute. Plain spoken sentences, no markdown, no numbering, no bullet symbols.

Respond ONLY with JSON: {"questions": ["<question 1>", "<question 2>", ...]}"""


@traceable(name="llm_parse_jd", run_type="llm")
async def parse_jd(jd_text: str) -> tuple[dict, dict]:
    """Classifies whether jd_text looks like a job description (separate call from question
    generation so each step's real token usage/cost can be tracked and shown individually).
    Returns ({"is_job_description": bool, "reason": str}, usage)."""

    async def go():
        r = await _client.post(
            "/chat/completions",
            json={
                "model": settings.llm_model(),
                "messages": [
                    {"role": "system", "content": JD_PARSE_SYSTEM_PROMPT},
                    {"role": "user", "content": jd_text},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.2,
                "usage": {"include": True},
            },
        )
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        result = _parse_json_object(content)
        result.setdefault("is_job_description", False)
        result.setdefault("reason", "")
        result.setdefault("company_name", None)
        result.setdefault("role_name", None)
        return result, {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cost": usage.get("cost"),
        }

    return await _retry(go)


@traceable(name="llm_generate_questions", run_type="llm")
async def generate_questions_from_jd(jd_text: str) -> tuple[list[str], dict]:
    """Generates up to 8 interview questions from an already-validated JD.
    Returns ([str, ...], usage)."""

    async def go():
        r = await _client.post(
            "/chat/completions",
            json={
                "model": settings.llm_model(),
                "messages": [
                    {"role": "system", "content": QUESTION_GEN_SYSTEM_PROMPT},
                    {"role": "user", "content": jd_text},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.5,
                "usage": {"include": True},
            },
        )
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        result = _parse_json_object(content)
        return result.get("questions", []), {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cost": usage.get("cost"),
        }

    return await _retry(go)


def _parse_json_object(content: str) -> dict:
    content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", content, re.S)
    return json.loads(m.group(0) if m else content)
