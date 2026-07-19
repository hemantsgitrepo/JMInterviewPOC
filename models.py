"""Thin OpenRouter wrappers: STT, LLM turn, TTS, JD-based question generation.
Retries with backoff, longer on 429 (rate limit) since an instant retry almost always
hits the same limit window."""

import asyncio
import base64
import io
import json
import logging
import os
import re
import wave

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
# Sarvam and Gnani are India-focused: both require a BCP-47 language code and neither
# offers a bare "en", so English is pinned as en-IN (same reasoning as STT_MODEL's
# language pin — auto-detect on 8kHz telephony audio can land on the wrong language).
INDIAN_ENGLISH = "en-IN"
SARVAM_STT_MODEL = "saaras:v3"
SARVAM_TTS_MODEL = "bulbul:v3"
SARVAM_TTS_SPEAKER = "shubh"
GNANI_TTS_MODEL = "vachana-voice-v3"
GNANI_TTS_VOICE = "Pranav"
# Both TTS providers can synthesize straight to 8kHz, so their audio needs no resampling
# before mu-law encoding — unlike Cartesia/OpenAI, which only emit 24kHz.
TELEPHONY_RATE = 8000

# Explicit female/male voice per TTS provider (settings "Agent voice"). The historical
# per-provider defaults above are mixed-gender (Cartesia "British Lady" is female, Sarvam
# shubh and Gnani Pranav are male), so gender "default" keeps them untouched rather than
# pretending they share one. OpenAI publishes no gender labels; nova/onyx are the
# conventional female-/male-sounding picks. Cartesia female is the original default voice;
# male is "Grant - Friendly Support" (neutral American, support-tuned), verified live.
TTS_VOICES = {
    "cartesia-sonic": {"female": CARTESIA_VOICE_ID, "male": "d46abd1d-2d02-43e8-819f-51fb652c1c61"},
    "openai-tts": {"female": "nova", "male": "onyx"},
    "sarvam-bulbul": {"female": "ishita", "male": SARVAM_TTS_SPEAKER},
    "gnani-vachana": {"female": "Kaveri", "male": GNANI_TTS_VOICE},
}

# Cartesia voices are language-bound (the English default can't speak Hindi), so Hindi
# gets its own pair; gender "default" resolves female, matching the English default's
# gender. Female is "Arushi - Hinglish Speaker" (built for code-mixed content — exactly
# what LLM-generated Hindi with English tech terms is); male is "Rohan - Steady
# Communicator" (corporate-tuned). The other three TTS providers keep their voices
# across languages: Sarvam switches via target_language_code, OpenAI and Gnani follow
# the script of the input text.
CARTESIA_VOICES_HI = {"female": "95d51f79-c397-46f9-b49a-23763d3eaa2d", "male": "4877b818-c7fe-4c89-b1cf-eadf8e23da72"}


def _tts_voice(provider: str, default: str) -> str:
    """Resolve the voice for a provider: the provider's original voice on "default",
    else the mapped voice for the selected gender."""
    gender = settings.tts_voice_gender()
    if gender in ("female", "male"):
        return TTS_VOICES.get(provider, {}).get(gender, default)
    return default


# Perceived gender of each provider's DEFAULT voice (gender setting "default"), so the
# prompt can align the agent's self-reference grammar with what the caller hears.
# OpenAI alloy is None on purpose: officially unlabeled and genuinely androgynous.
DEFAULT_VOICE_GENDER = {
    "cartesia-sonic": "female",  # British Lady (en) / Arushi (hi)
    "openai-tts": None,          # alloy
    "sarvam-bulbul": "male",     # shubh
    "gnani-vachana": "male",     # Pranav
}


def agent_voice_gender() -> str | None:
    """The gender the current TTS voice is heard as ("female"/"male"), or None when
    genuinely ambiguous. Feeds the self-reference grammar rule in the system prompt so
    a female-sounding agent never speaks about itself in masculine forms (pervasive in
    Hindi, where first-person verbs are gendered; rare but possible in English)."""
    g = settings.tts_voice_gender()
    if g in ("female", "male"):
        return g
    return DEFAULT_VOICE_GENDER.get(settings.tts_provider())


def tts_voice_signature() -> str:
    """Identity of the audio the current TTS settings produce, for caches of
    pre-synthesized clips (filler acks): same signature -> same voice. Includes the
    language because it changes the voice (Cartesia), the pronunciation target
    (Sarvam), and the filler phrases themselves."""
    provider = settings.tts_provider()
    lang = settings.language()
    if provider == "cartesia-sonic" and lang == "hi":
        voice = CARTESIA_VOICES_HI["male" if settings.tts_voice_gender() == "male" else "female"]
    else:
        defaults = {
            "cartesia-sonic": CARTESIA_VOICE_ID,
            "openai-tts": OPENAI_TTS_VOICE,
            "sarvam-bulbul": SARVAM_TTS_SPEAKER,
            "gnani-vachana": GNANI_TTS_VOICE,
        }
        voice = _tts_voice(provider, defaults.get(provider, ""))
    return f"{provider}|{voice}|{lang}"

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

_sarvam = httpx.AsyncClient(
    base_url="https://api.sarvam.ai",
    headers={"api-subscription-key": os.environ.get("SARVAM_API_KEY", "")},
    timeout=45,
)

# Gnani's speech APIs are served under their Vachana product domain (the GNANI_API_KEY
# is a Vachana key — hence the vach* prefix), not under gnani.ai.
_gnani = httpx.AsyncClient(
    base_url="https://api.vachana.ai",
    headers={"X-API-Key-ID": os.environ.get("GNANI_API_KEY", "")},
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
    provider = settings.stt_provider()
    if provider == "deepgram-nova":
        return await _transcribe_deepgram(wav)
    if provider == "sarvam-saaras":
        return await _transcribe_sarvam(wav)
    if provider == "gnani-vachana":
        return await _transcribe_gnani(wav)
    return await _transcribe_openrouter(wav)


def _wav_seconds(wav: bytes) -> float | None:
    """Duration of a WAV we built ourselves. Used for providers that don't report one:
    the utterance length is known locally, so usage stays populated instead of blank."""
    try:
        with wave.open(io.BytesIO(wav)) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None


async def _transcribe_openrouter(wav: bytes) -> dict:
    """OpenRouter Whisper — cost/seconds come straight from its usage block (real)."""

    async def go():
        r = await _client.post(
            "/audio/transcriptions",
            # language is pinned, never auto-detected: auto-detect on 8kHz telephony audio
            # occasionally lands on the wrong language and returns a fluent-but-wrong
            # transcript (seen once as Welsh), which reaches the LLM as a garbled answer.
            # Whisper has no code-switching mode, so Hindi mode hard-pins "hi" — the
            # weakest Hinglish option here; Deepgram (multi) or Sarvam (codemix) handle
            # mixed speech better.
            data={"model": STT_MODEL, "language": "hi" if settings.language() == "hi" else "en"},
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
        # Hindi mode uses language=multi (nova-3 code-switching, Hindi included) rather
        # than a hard hi pin: interview answers are Hinglish in practice, and multi keeps
        # the English technical vocabulary intact instead of forcing it into Hindi.
        lang = "multi" if settings.language() == "hi" else "en"
        r = await _deepgram.post(
            f"/v1/listen?model={DEEPGRAM_STT_MODEL}&language={lang}",
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


async def _transcribe_sarvam(wav: bytes) -> dict:
    """Sarvam Saaras v3. Sync REST caps an utterance at 30s; longer audio is rejected
    rather than truncated, so that surfaces as a failed turn, not a silently clipped
    answer. cost=None: Sarvam bills per-minute off-request (see docs/backlog.md)."""

    async def go():
        # Hindi mode uses Sarvam's codemix mode — their recommended setting for Hinglish:
        # English words stay in Latin script, Hindi in Devanagari, numbers normalized.
        lang, mode = (
            ("hi-IN", "codemix") if settings.language() == "hi" else (INDIAN_ENGLISH, "transcribe")
        )
        r = await _sarvam.post(
            "/speech-to-text",
            data={"model": SARVAM_STT_MODEL, "language_code": lang, "mode": mode},
            files={"file": ("utterance.wav", wav, "audio/wav")},
        )
        r.raise_for_status()
        return {
            "text": (r.json().get("transcript") or "").strip(),
            "seconds": _wav_seconds(wav),
            "cost": None,
        }

    return await _retry(go)


async def _transcribe_gnani(wav: bytes) -> dict:
    """Gnani Vachana v3, in its ITN mode. Returns lowercase and unpunctuated ("i have five
    years..."), unlike every other adapter here; harmless because classify_utterance and
    ends_midthought both case-fold and strip punctuation before matching."""

    async def go():
        # Gnani has no code-switching mode; Hindi mode pins hi-IN.
        lang = "hi-IN" if settings.language() == "hi" else INDIAN_ENGLISH
        r = await _gnani.post(
            "/stt/v3",
            data={"language_code": lang, "format": "transcribe"},
            files={"audio_file": ("utterance.wav", wav, "audio/wav")},
        )
        r.raise_for_status()
        return {
            "text": (r.json().get("transcript") or "").strip(),
            "seconds": _wav_seconds(wav),
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
    # Hindi Whisper artifacts on silence/noise. The subscribe lines are the classic
    # YouTube-training-data tell; नमस्ते is deliberately NOT here — it's a real greeting.
    "धन्यवाद", "शुक्रिया", "देखने के लिए धन्यवाद",
    "सब्सक्राइब करें", "कृपया सब्सक्राइब करें", "चैनल को सब्सक्राइब करें",
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
    provider = settings.tts_provider()
    if provider == "openai-tts":
        return await _speak_openai(text)
    if provider == "sarvam-bulbul":
        return await _speak_sarvam(text)
    if provider == "gnani-vachana":
        return await _speak_gnani(text)
    return await _speak_cartesia(text)


def _wav_to_ulaw(wav: bytes) -> bytes:
    """WAV bytes -> 8kHz mu-law. Both Sarvam and Gnani are asked to synthesize at 8kHz
    directly, but the rate is read back from the header rather than assumed: a provider
    that quietly ignores the request would otherwise emit chipmunk audio down the line."""
    with wave.open(io.BytesIO(wav)) as w:
        rate, pcm = w.getframerate(), w.readframes(w.getnframes())
    if rate != TELEPHONY_RATE:
        pcm = resample(pcm, rate, TELEPHONY_RATE)
    return pcm_to_ulaw(pcm)


async def _speak_cartesia(text: str) -> bytes:
    async def go():
        if settings.language() == "hi":
            voice = CARTESIA_VOICES_HI["male" if settings.tts_voice_gender() == "male" else "female"]
        else:
            voice = _tts_voice("cartesia-sonic", CARTESIA_VOICE_ID)
        body = {
            "transcript": text,
            "model_id": CARTESIA_MODEL,
            "voice": {"mode": "id", "id": voice},
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": CARTESIA_PCM_RATE,
            },
        }
        if settings.language() == "hi":
            body["language"] = "hi"
        r = await _cartesia.post("/tts/bytes", json=body)
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
                "voice": _tts_voice("openai-tts", OPENAI_TTS_VOICE),
                "input": text,
                "response_format": "pcm",  # 24kHz mono s16le
            },
        )
        r.raise_for_status()
        return r.content

    pcm = await _retry(go)
    return pcm_to_ulaw(resample(pcm, OPENAI_PCM_RATE, 8000))


async def _speak_sarvam(text: str) -> bytes:
    """Sarvam Bulbul v3. Returns base64-encoded audio in JSON (not raw bytes like the
    other TTS providers), split across an `audios` list when the text is long enough to
    be sentence-split — the chunks concatenate into one WAV stream."""

    async def go():
        r = await _sarvam.post(
            "/text-to-speech",
            json={
                "text": text,
                "model": SARVAM_TTS_MODEL,
                "speaker": _tts_voice("sarvam-bulbul", SARVAM_TTS_SPEAKER),
                "target_language_code": "hi-IN" if settings.language() == "hi" else INDIAN_ENGLISH,
                "speech_sample_rate": TELEPHONY_RATE,
                "output_audio_codec": "wav",
            },
        )
        r.raise_for_status()
        return r.json().get("audios") or []

    chunks = await _retry(go)
    return b"".join(_wav_to_ulaw(base64.b64decode(c)) for c in chunks)


async def _speak_gnani(text: str) -> bytes:
    async def go():
        r = await _gnani.post(
            "/api/v1/tts/inference",
            json={
                "text": text,
                "model": GNANI_TTS_MODEL,
                "voice": _tts_voice("gnani-vachana", GNANI_TTS_VOICE),
                "audio_config": {
                    "sample_rate": TELEPHONY_RATE,
                    "num_channels": 1,
                    "sample_width": 2,
                    "encoding": "linear_pcm",
                    "container": "wav",
                },
            },
        )
        r.raise_for_status()
        return r.content

    return _wav_to_ulaw(await _retry(go))


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
