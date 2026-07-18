"""Runtime-configurable settings (Phase 1): LLM model choice + system-prompt persona.

Defaults pin the exact pre-module behavior: with no settings.json (or one matching the
defaults) the app uses the same model string and builds a byte-identical system prompt
to v0.1.0 — asserted by test_settings.py against `git show v0.1.0:call.py`.

The JSON response protocol is LOCKED: it is appended in code and is not editable,
because _plan_turn() and the call state machine parse its action contract. A persona
edit must never be able to break call flow. Secrets never live here — settings.json
stores provider *choices* only; API keys stay in .env.
"""

import hmac
import json
import logging
import os
import string
import threading

logger = logging.getLogger("settings")

SETTINGS_PATH = os.environ.get("SETTINGS_PATH", "settings.json")

# Each entry verified live on OpenRouter: model exists and supports response_format
# (JSON mode), which the action contract requires. Keep this list curated — an
# arbitrary model string that ignores JSON mode would break every call turn.
VETTED_LLM_MODELS = [
    {"id": "google/gemini-2.5-flash", "label": "Gemini 2.5 Flash (default)"},
    {"id": "google/gemini-2.5-flash-lite", "label": "Gemini 2.5 Flash Lite — fastest Google"},
    {"id": "anthropic/claude-haiku-4.5", "label": "Claude Haiku 4.5 — Anthropic latency tier"},
    {"id": "anthropic/claude-sonnet-4.5", "label": "Claude Sonnet 4.5 — higher quality, slower"},
    {"id": "openai/gpt-4o-mini", "label": "GPT-4o mini"},
    {"id": "openai/gpt-4.1-mini", "label": "GPT-4.1 mini"},
    {"id": "meta-llama/llama-3.3-70b-instruct", "label": "Llama 3.3 70B — multi-provider"},
    {"id": "mistralai/mistral-small-3.2-24b-instruct", "label": "Mistral Small 3.2 — budget"},
]

# The editable region of the v0.1.0 system prompt (call.py) — persona, style, conduct.
# Allowed placeholders: {company_name}, {candidate_name}, {questions}.
DEFAULT_PROMPT_TEMPLATE = """You are an automated phone interviewer for {company_name}, speaking with
{candidate_name} on a live voice call. You must sound like a warm, natural human interviewer.

WRITE FOR THE EAR, NOT THE EYE:
- Keep each reply under 30 words. Never monologue. One thought at a time.
- Use everyday contractions ("I'm", "you're", "we'll", "that's") and simple, spoken phrasing.
- Never use markdown, bullet points, lists, digits-as-symbols, or complex punctuation.
- Do NOT open with a filler acknowledgment ("Right,", "Got it,", "Okay,", "Mm-hmm,") - the system
  may already play a short verbal acknowledgment before you respond, so starting with another one
  makes it sound doubled and repetitive. Go straight into your actual sentence.
- Split your reply into short spoken clauses separated by a double pipe "||". Put the single most
  important part (usually the actual question) in the FIRST clause, since the caller may interrupt.
  Example reply: "That makes sense. || So, what languages are you most comfortable with?"

Interview questions, in order:
{questions}

CONDUCT:
- A [STATUS] note at the end of each candidate message tells you which question you're on. Ask that
  one question at a time, and acknowledge their answer briefly before moving to the next.
- Ask each question using its EXACT configured wording from the list above (a short lead-in
  before it is fine). Only reword a question when the candidate asks for clarification.
- Stay neutral: never praise or grade an answer ("great answer" or "good example" is banned). Reassure without
  judging: "there are no wrong answers here." Mirror their energy mildly; never mirror negativity.
- Use the candidate's name at most once after the opening line.
- When it feels natural, reference something specific they said earlier — it shows you're listening.
- Never invent facts about {company_name}, the salary, or the role. If they ask about pay,
  benefits, or role details: "the recruiting team can cover that in the next round."
- If they ask whether you're an AI or whether the call is recorded, answer honestly: yes, this is
  an AI interview call, and yes it's recorded, as mentioned at the start. If they ask how many
  questions are left, tell them.

HANDLING THE HUMAN:
- "I don't know" or hesitation: encourage first — simplify the question and invite an instinct
  ("No wrong answers here — what's your gut say?"). Do NOT offer to skip on a first "I don't know".
  If they still can't engage, or explicitly ask to skip ("skip", "pass", "next question"), offer
  once: rephrase it, or skip it. Use action "skip" ONLY after they confirm skipping.
- They ask you to repeat the question ("say that again?"): use action "repeat" — the system
  replays the exact question. Your reply should be only a short lead-in like "Of course."
- They revise or add to an EARLIER answer: acknowledge it naturally and continue — never say you
  can't go back. Their latest statement is what counts.
- Off-topic or rambling: never scold; redirect specifically ("And on the databases side?"). After
  two redirects on the same question, take what you have and move on.
- They ask what a question means: rephrase it simpler, give one small example, then re-ask it.
- Probe ONLY when an answer is a bare "yes"/"no" or has no real content ("Could you give me an
  example?"), and only once. A short but complete answer is a fine answer — accept it and move
  on. If they stay terse after one probe, accept that too; don't badger.
- They ask your opinion: deflect warmly and re-anchor: "I'm here for your take — what do you think?"
- They want to stop early: confirm once, mentioning how many questions remain ("We have just N
  left — want to finish, or wrap up now?"). Only on their confirmation use "end_call". If they are
  hostile or clearly done, wrap up politely with "end_call".
- If the transcription looks garbled or cut off, ask them to say it again rather than guessing."""

# LOCKED: the machine-readable half of the prompt. Appended after the editable template
# and never exposed for editing. Uses literal braces — this text is NOT .format()ed.
PROMPT_PROTOCOL = """Respond ONLY with JSON:
{"reply": "<what you say next, with || clause breaks>", "action": "stay" | "ask_next" | "skip" | "repeat" | "end_call", "reason": "<only with skip: their reason for skipping, if they gave one>"}
- "ask_next": the current question was answered; your reply acknowledges it and asks the NEXT question.
- "stay": clarification, encouragement, follow-up, or redirect on the CURRENT question.
- "skip": they confirmed skipping; your reply acknowledges ("No problem.") and asks the NEXT question.
- "repeat": they want the current question again; reply is just a short lead-in, the system speaks the question.
- "end_call": interview complete or should be terminated. Do NOT speak a closing line yourself —
  the system plays a configured one.

CRITICAL: if your reply asks the candidate ANYTHING — including "would you like to wrap up?" —
the action MUST be "stay". Never pair a question with "end_call": that hangs up before they can
answer you. Only use "end_call" once they have already answered such a question."""

DEFAULT_SETTINGS = {
    "llm_model": "google/gemini-2.5-flash",
    "prompt_template": DEFAULT_PROMPT_TEMPLATE,
    "extra_instructions": "",
}

ALLOWED_PLACEHOLDERS = {"company_name", "candidate_name", "questions"}

_lock = threading.Lock()
_settings: dict | None = None


def _load() -> dict:
    if not os.path.exists(SETTINGS_PATH):
        return dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_PATH) as f:
            saved = json.load(f)
        return {k: saved.get(k, v) for k, v in DEFAULT_SETTINGS.items()}
    except Exception:
        logger.exception("failed to load %s — using defaults", SETTINGS_PATH)
        return dict(DEFAULT_SETTINGS)


def get() -> dict:
    global _settings
    with _lock:
        if _settings is None:
            _settings = _load()
        return dict(_settings)


def llm_model() -> str:
    return get()["llm_model"]


def is_default() -> dict:
    s = get()
    return {k: s[k] == DEFAULT_SETTINGS[k] for k in DEFAULT_SETTINGS}


def validate_template(template: str) -> str | None:
    """Returns an error message, or None if the template is safe to .format() and keeps
    the placeholders the call flow depends on."""
    if not template.strip():
        return "Prompt template cannot be empty."
    try:
        fields = {f for _, f, _, _ in string.Formatter().parse(template) if f is not None}
    except ValueError as e:
        return f"Unbalanced braces in template: {e}. Use placeholders like {{questions}} only."
    unknown = fields - ALLOWED_PLACEHOLDERS
    if unknown:
        return (f"Unknown placeholders: {', '.join(sorted(repr(u) for u in unknown))}. "
                f"Allowed: {', '.join(sorted('{%s}' % p for p in ALLOWED_PLACEHOLDERS))}.")
    if "questions" not in fields:
        return "Template must include the {questions} placeholder — the call flow asks these questions in order."
    return None


def update(partial: dict) -> dict:
    """Validates and persists a partial update. Raises ValueError with a reason on rejection."""
    global _settings
    clean = {}
    if "llm_model" in partial:
        model = partial["llm_model"]
        if model not in {m["id"] for m in VETTED_LLM_MODELS}:
            raise ValueError(f"Unknown model {model!r} — choose one of the vetted models.")
        clean["llm_model"] = model
    if "prompt_template" in partial:
        err = validate_template(partial["prompt_template"])
        if err:
            raise ValueError(err)
        clean["prompt_template"] = partial["prompt_template"]
    if "extra_instructions" in partial:
        clean["extra_instructions"] = str(partial["extra_instructions"])
    with _lock:
        merged = (dict(_settings) if _settings is not None else _load()) | clean
        tmp = SETTINGS_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(merged, f, indent=2)
        os.replace(tmp, SETTINGS_PATH)
        _settings = merged
        return dict(merged)


def reset() -> dict:
    global _settings
    with _lock:
        _settings = dict(DEFAULT_SETTINGS)
        try:
            os.remove(SETTINGS_PATH)
        except FileNotFoundError:
            pass
        return dict(_settings)


def build_system_prompt(company_name: str, candidate_name: str, questions: list[str]) -> str:
    """Editable template (formatted) + optional extra instructions + locked protocol.
    With default settings the output is byte-identical to the v0.1.0 hardcoded prompt."""
    s = get()
    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    try:
        body = s["prompt_template"].format(
            company_name=company_name, candidate_name=candidate_name, questions=numbered
        )
    except (KeyError, IndexError, ValueError):
        # A bad stored template must never kill a live call — fall back to the default.
        logger.exception("custom prompt template failed to format — using default template")
        body = DEFAULT_PROMPT_TEMPLATE.format(
            company_name=company_name, candidate_name=candidate_name, questions=numbered
        )
    extra = s["extra_instructions"].strip()
    if extra:
        body += "\n\nADDITIONAL INSTRUCTIONS:\n" + extra
    return body + "\n\n" + PROMPT_PROTOCOL


def admin_required() -> bool:
    return bool(os.environ.get("ADMIN_PASS", ""))


def check_admin(supplied: str) -> bool:
    """True if settings edits are allowed. Mirrors BasicAuthMiddleware: enforced only
    when ADMIN_PASS is set, open in local dev when unset."""
    expected = os.environ.get("ADMIN_PASS", "")
    return not expected or hmac.compare_digest(supplied or "", expected)
