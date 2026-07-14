"""Thin OpenRouter wrappers: STT, LLM turn, TTS, JD-based question generation. One retry each."""

import json
import os
import re

import httpx
from dotenv import load_dotenv

from audio import pcm_to_ulaw, resample

load_dotenv()

STT_MODEL = "openai/whisper-large-v3-turbo"
LLM_MODEL = "google/gemini-2.5-flash"
CARTESIA_MODEL = "sonic-3.5"
CARTESIA_VOICE_ID = "79a125e8-cd45-4c13-8a67-188112f4dd22"
CARTESIA_PCM_RATE = 24_000

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


async def _retry(fn):
    try:
        return await fn()
    except Exception:
        return await fn()


async def transcribe(wav: bytes) -> dict:
    """Returns {"text": str, "seconds": float|None, "cost": float|None} — cost/seconds
    come straight from OpenRouter's transcription usage block (real, not estimated)."""

    async def go():
        r = await _client.post(
            "/audio/transcriptions",
            data={"model": STT_MODEL},
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


async def next_turn(system: str, messages: list[dict]) -> tuple[dict, dict]:
    """Returns ({"reply": str, "action": ...}, usage) where usage is
    {"prompt_tokens": int, "completion_tokens": int, "cost": float|None} — cost is real,
    from OpenRouter's usage-accounting feature (`usage: {include: true}`), not estimated."""

    async def go():
        r = await _client.post(
            "/chat/completions",
            json={
                "model": LLM_MODEL,
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


def _parse_json(content: str) -> dict:
    content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", content, re.S)
    out = json.loads(m.group(0) if m else content)
    if out.get("action") not in ("stay", "ask_next", "end_call"):
        out["action"] = "stay"
    out.setdefault("reply", "")
    return out


async def speak(text: str) -> bytes:
    """Text -> 8kHz mu-law bytes ready for a Twilio media stream, via Cartesia Sonic 3.5."""

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


JD_PARSE_SYSTEM_PROMPT = """You review a piece of text and decide whether it is a job description
(JD) — text describing an open role a company is hiring for (job title, responsibilities,
required skills/qualifications, or "about the role" language).

Respond ONLY with JSON:
{"is_job_description": true|false, "reason": "<short reason, required if false, else empty>"}"""

QUESTION_GEN_SYSTEM_PROMPT = """You are given a job description. Write 5 to 8 clear interview
questions tailored to the specific responsibilities and skills in it. Each must be answerable
out loud in under a minute. Plain spoken sentences, no markdown, no numbering, no bullet symbols.

Respond ONLY with JSON: {"questions": ["<question 1>", "<question 2>", ...]}"""


async def parse_jd(jd_text: str) -> tuple[dict, dict]:
    """Classifies whether jd_text looks like a job description (separate call from question
    generation so each step's real token usage/cost can be tracked and shown individually).
    Returns ({"is_job_description": bool, "reason": str}, usage)."""

    async def go():
        r = await _client.post(
            "/chat/completions",
            json={
                "model": LLM_MODEL,
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
        return result, {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cost": usage.get("cost"),
        }

    return await _retry(go)


async def generate_questions_from_jd(jd_text: str) -> tuple[list[str], dict]:
    """Generates up to 8 interview questions from an already-validated JD.
    Returns ([str, ...], usage)."""

    async def go():
        r = await _client.post(
            "/chat/completions",
            json={
                "model": LLM_MODEL,
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
