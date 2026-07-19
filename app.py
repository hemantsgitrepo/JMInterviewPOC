"""Jobmanch.ai AI Interview Caller — POC backend."""

import asyncio
import base64
import hmac
import io
import logging
import os
import re

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from fastapi import FastAPI, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from pypdf import PdfReader

import dialer
import models
import settings
import store
from call import handle_media_ws

app = FastAPI(title="Jobmanch.ai Interview Caller POC")


class BasicAuthMiddleware:
    """Gates the UI and /api/* behind a shared username/password. Twilio's webhooks and
    the media WebSocket are exempt — Twilio's servers can't answer a login prompt, so
    gating those would break real calls. Auth is skipped entirely if BASIC_AUTH_USER
    isn't set, so local dev (uvicorn, loopback.py, test scripts) keeps working unchanged."""

    def __init__(self, app):
        self.app = app
        self.user = os.environ.get("BASIC_AUTH_USER", "")
        self.password = os.environ.get("BASIC_AUTH_PASS", "")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.user or scope["path"].startswith("/twilio/"):
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        auth = headers.get(b"authorization", b"").decode(errors="ignore")
        if auth.startswith("Basic "):
            try:
                u, _, p = base64.b64decode(auth[6:]).decode().partition(":")
            except Exception:
                u, p = "", ""
            if hmac.compare_digest(u, self.user) and hmac.compare_digest(p, self.password):
                return await self.app(scope, receive, send)
        response = Response(
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="JMInterviewPOC"'},
        )
        return await response(scope, receive, send)


app.add_middleware(BasicAuthMiddleware)

E164 = re.compile(r"^\+[1-9]\d{6,14}$")


class ConfigIn(BaseModel):
    opening_line: str
    company_name: str
    role_name: str
    questions: list[str]
    end_call_line: str


class CandidateIn(BaseModel):
    name: str
    phone: str


class CandidatesIn(BaseModel):
    candidates: list[CandidateIn]


class JDIn(BaseModel):
    jd_text: str


class SettingsIn(BaseModel):
    llm_model: str | None = None
    stt_provider: str | None = None
    tts_provider: str | None = None
    tts_voice_gender: str | None = None
    language: str | None = None
    prompt_template: str | None = None
    extra_instructions: str | None = None


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/api/config")
def get_config():
    return store.config


@app.post("/api/config")
def save_config(cfg: ConfigIn):
    questions = [q.strip() for q in cfg.questions if q.strip()]
    if not questions:
        raise HTTPException(400, "At least one question is required")
    store.config.update(cfg.model_dump() | {"questions": questions})
    return {"ok": True}


def _require_admin(request: Request):
    if not settings.check_admin(request.headers.get("x-admin-pass", "")):
        raise HTTPException(403, "Admin password required (X-Admin-Pass header)")


@app.get("/api/settings")
def get_settings():
    return {
        "settings": settings.get(),
        "defaults": settings.DEFAULT_SETTINGS,
        "is_default": settings.is_default(),
        "vetted_models": settings.VETTED_LLM_MODELS,
        "vetted_stt_providers": settings.provider_status(settings.VETTED_STT_PROVIDERS),
        "vetted_tts_providers": settings.provider_status(settings.VETTED_TTS_PROVIDERS),
        "languages": settings.LANGUAGES,
        "locked_protocol": settings.PROMPT_PROTOCOL,
        "admin_required": settings.admin_required(),
    }


@app.post("/api/settings")
def update_settings(body: SettingsIn, request: Request):
    _require_admin(request)
    try:
        return {"ok": True, "settings": settings.update(body.model_dump(exclude_none=True))}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/settings/reset")
def reset_settings(request: Request):
    _require_admin(request)
    return {"ok": True, "settings": settings.reset()}


@app.post("/api/settings/verify-admin")
def verify_admin(request: Request):
    """Gate for revealing the Settings UI: 200 iff the supplied password is valid
    (or no ADMIN_PASS is configured, mirroring the auth-optional local-dev mode)."""
    _require_admin(request)
    return {"ok": True}


async def _validate_and_store_jd(text: str) -> dict:
    """Validates the text as a JD, stores it for prompt grounding, and auto-populates
    company/role when the JD states them. Never generates questions — that only happens
    on the explicit Generate button (see /api/questions/generate)."""
    parse_result, parse_usage = await models.parse_jd(text)
    store.config["ai_usage"]["jd_text"] = text[:500]
    store.config["ai_usage"]["jd_parsing_usage"] = parse_usage
    if not parse_result.get("is_job_description"):
        raise HTTPException(400, parse_result.get("reason") or "That doesn't look like a job description.")
    store.config["jd_text"] = text  # full validated JD: grounds answers to role questions
    if (parse_result.get("company_name") or "").strip():
        store.config["company_name"] = parse_result["company_name"].strip()
    if (parse_result.get("role_name") or "").strip():
        store.config["role_name"] = parse_result["role_name"].strip()
    return parse_result


@app.post("/api/questions/generate")
async def generate_questions(body: JDIn):
    text = body.jd_text.strip()
    if len(text) < 40:
        raise HTTPException(400, "Please paste a longer job description (at least a few sentences).")
    await _validate_and_store_jd(text)
    questions, gen_usage = await models.generate_questions_from_jd(text)
    store.config["ai_usage"]["questions_from_jd"] = True
    store.config["ai_usage"]["question_generation_usage"] = gen_usage
    questions = [q.strip() for q in questions if q.strip()]
    if not questions:
        raise HTTPException(502, "The AI didn't return any questions — try again.")
    return {"questions": questions}


@app.post("/api/jd/upload-pdf")
async def upload_jd_pdf(file: UploadFile):
    """Extract + validate a JD PDF and populate config from it. Does NOT generate
    questions — the user triggers that explicitly with the Generate button."""
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Only PDF files are supported.")
    try:
        pdf_bytes = await file.read()
        pdf = PdfReader(io.BytesIO(pdf_bytes))
        text = "".join(page.extract_text() for page in pdf.pages).strip()
    except Exception as e:
        raise HTTPException(500, f"Error reading PDF: {str(e)}")
    if len(text) < 40:
        raise HTTPException(400, "PDF is too short or empty. Please upload a longer job description.")
    await _validate_and_store_jd(text)
    return {
        "jd_text": text,
        "company_name": store.config["company_name"],
        "role_name": store.config["role_name"],
    }


@app.post("/api/candidates")
def save_candidates(body: CandidatesIn):
    if store.session["running"]:
        raise HTTPException(409, "A calling session is running")
    bad = [c.phone for c in body.candidates if not E164.match(c.phone.strip())]
    if bad:
        raise HTTPException(400, f"Invalid E.164 phone numbers: {', '.join(bad)}")
    if not body.candidates:
        raise HTTPException(400, "Candidate list is empty")
    store.reset_candidates()
    for c in body.candidates:
        store.add_candidate(c.name.strip(), c.phone.strip())
    return {"count": len(body.candidates)}


@app.post("/api/session/start", status_code=202)
async def start_session():
    if store.session["running"]:
        raise HTTPException(409, "Session already running")
    if not store.candidates:
        raise HTTPException(400, "No candidates configured")
    if not any(c["status"] == "pending" for c in store.candidates.values()):
        raise HTTPException(400, "No pending candidates left — re-submit the list to retry")
    asyncio.create_task(dialer.run_session())
    return {"started": True}


@app.post("/api/session/end_call")
async def end_call():
    cid = store.session.get("current")
    if not cid:
        raise HTTPException(400, "No call in progress")
    await dialer.end_current_call(cid)
    return {"ok": True}


@app.post("/api/session/reset")
def reset_all():
    if store.session["running"]:
        raise HTTPException(409, "Cannot reset while a session is running")
    store.config["questions"] = ["Tell me something about yourself."]
    store.config["jd_text"] = ""
    store.config["ai_usage"] = {
        "jd_text": None,
        "questions_from_jd": False,
        "generation_usage": None,
    }
    store.reset_candidates()
    return {"ok": True}


@app.post("/api/candidates/{cid}/retry")
def retry_candidate(cid: str):
    if store.session["running"]:
        raise HTTPException(409, "A calling session is running")
    cand = store.candidates.get(cid)
    if not cand:
        raise HTTPException(404, "Unknown candidate")
    if cand["status"] not in ("failed", "no_answer", "completed"):
        raise HTTPException(400, f"Candidate is {cand['status']}, not eligible for retry")
    cand.update(
        status="pending", call_sid=None, answered_by=None,
        started_at=None, ended_at=None, partial=False, transcript=[],
        recording_sid=None, recording_path=None, recording_duration=None,
        usage=None, total_cost=None,
    )
    return {"ok": True}


@app.get("/api/session")
def session_status():
    return {
        "running": store.session["running"],
        "current_candidate_id": store.session["current"],
        "candidates": [
            {k: c[k] for k in ("id", "name", "phone", "status", "started_at", "ended_at", "partial")}
            | {"turns": len(c["transcript"])}
            for c in store.candidates_list()
        ],
    }


@app.get("/api/candidates/{cid}/transcript")
def transcript(cid: str):
    cand = store.candidates.get(cid)
    if not cand:
        raise HTTPException(404, "Unknown candidate")
    return {"candidate": cand["name"], "status": cand["status"], "turns": cand["transcript"]}


@app.get("/api/candidates/{cid}/details")
def candidate_details(cid: str):
    cand = store.candidates.get(cid)
    if not cand:
        raise HTTPException(404, "Unknown candidate")
    has_recording = bool(cand["recording_path"] and os.path.exists(cand["recording_path"]))
    return {
        "id": cand["id"],
        "name": cand["name"],
        "phone": cand["phone"],
        "status": cand["status"],
        "partial": cand["partial"],
        "started_at": cand["started_at"],
        "ended_at": cand["ended_at"],
        "answered_by": cand["answered_by"],
        "usage": cand["usage"],
        "total_known_cost": cand["total_cost"],
        "ai_usage": store.config["ai_usage"],
        "recording_available": has_recording,
        "recording_duration": cand["recording_duration"],
    }


@app.get("/api/candidates/{cid}/recording")
def recording(cid: str):
    cand = store.candidates.get(cid)
    if not cand or not cand["recording_path"] or not os.path.exists(cand["recording_path"]):
        raise HTTPException(404, "Recording not available")
    return FileResponse(
        cand["recording_path"], media_type="audio/mpeg", filename=f"{cand['name']}-call.mp3"
    )


@app.post("/twilio/status")
async def twilio_status(request: Request):
    await dialer.on_status_callback(dict(await request.form()))
    return Response(status_code=204)


@app.post("/twilio/recording")
async def twilio_recording(request: Request):
    await dialer.on_recording_callback(dict(await request.form()))
    return Response(status_code=204)


@app.websocket("/twilio/media")
async def twilio_media(ws: WebSocket):
    await handle_media_ws(ws)
