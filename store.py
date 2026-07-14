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
    "ai_usage": {
        "jd_text": None,
        "questions_from_jd": False,
        "jd_parsing_usage": None,        # {"prompt_tokens", "completion_tokens", "cost"}
        "question_generation_usage": None,
    },
}

candidates: dict[str, dict] = {}  # id -> candidate dict
order: list[str] = []

session = {"running": False, "current": None, "call_done": None}

# Short "thinking" clips played the instant the candidate stops talking, to mask
# STT+LLM+TTS latency. Pre-synthesized once by the dialer; empty in loopback (skipped).
FILLER_PHRASES = ["Right.", "Okay.", "Sure."]
FILLER_ULAW: list[bytes] = []


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
