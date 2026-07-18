"""In-memory store. ponytail: no DB — add SQLite only if statuses must survive restarts."""

import time
import uuid

config = {
    "opening_line": (
        "Hello [Candidate Name], this is an interview call from JobManch AI on behalf of "
        "[Company Name] for the [Role Name] role. This call may be recorded for quality "
        "and training purposes. Are you ready to begin?"
    ),
    "company_name": "Acme Corp",
    "role_name": "Software Engineer",
    "questions": [],  # no defaults — added manually or via the AI generator
    "end_call_line": "Thank you for your time today. Our team will review and be in touch. Goodbye!",
    "jd_text": "",  # full uploaded/pasted JD; injected into the system prompt when present
    "ai_usage": {
        "jd_text": None,
        "questions_from_jd": False,
        "jd_parsing_usage": None,        # {"prompt_tokens", "completion_tokens", "cost"}
        "question_generation_usage": None,
    },
    # Conversational behavior knobs (server-side; not part of the UI's ConfigIn, so
    # /api/config saves leave this block untouched).
    "behavior": {
        "silence_tier1_ms": 4000,     # pure reassurance ("Take your time.") — no question
        "silence_tier2_ms": 9000,     # offer to repeat or move on
        # True hesitation noises only — "like/so/actually" are content words, keep them out.
        "filler_words": ["um", "uh", "erm", "hmm", "uhh", "umm", "huh", "mmm", "mm", "hm", "er", "ah"],
        "filler_ratio": 0.8,          # >= this fraction filler tokens -> silent wait
        "filler_extra_wait_ms": 6000, # extra listening window after a filler-only utterance
        "low_volume_rms": 150,        # spoke but transcription empty + at least this loud -> reprompt
        "max_skips": 3,               # soft: past this, warmly offer end-or-continue
        "confirm_key_facts": False,   # paraphrase-confirm load-bearing factual answers
    },
}

candidates: dict[str, dict] = {}  # id -> candidate dict
order: list[str] = []

session = {"running": False, "current": None, "call_done": None}

# Short "thinking" clips played the instant the candidate stops talking, to mask
# STT+LLM+TTS latency. Pre-synthesized once by the dialer; empty in loopback (skipped).
# The dialer appends None entries so sometimes no acknowledgment plays — a beat of
# silence before answering is the most human backchannel of all.
FILLER_PHRASES = ["Right.", "Okay.", "Sure.", "Alright.", "I see.", "Got it.", "Mm, okay.", "Mhm."]
FILLER_ULAW: list[bytes | None] = []


def candidates_list() -> list[dict]:
    return [candidates[i] for i in order]


def add_candidate(name: str, phone: str) -> dict:
    cand = {
        "id": uuid.uuid4().hex[:8],
        "name": name,
        "phone": phone,
        "status": "pending",
        "call_sid": None,
        "answered_by": None,
        "started_at": None,
        "ended_at": None,
        "partial": False,
        "transcript": [],
        "created_at": time.time(),
        "recording_sid": None,
        "recording_path": None,
        "recording_duration": None,
        "usage": None,
        "total_cost": None,
    }
    candidates[cand["id"]] = cand
    order.append(cand["id"])
    return cand


def reset_candidates():
    candidates.clear()
    order.clear()


def by_call_sid(sid: str) -> dict | None:
    return next((c for c in candidates.values() if c["call_sid"] == sid), None)
