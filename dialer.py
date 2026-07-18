"""Sequential call queue: dial one candidate at a time, advance on terminal status."""

import asyncio
import logging
import os
import re
import time
from xml.sax.saxutils import quoteattr

import httpx
from dotenv import load_dotenv
from twilio.rest import Client

import models
import store

load_dotenv()

logger = logging.getLogger("dialer")


def _host_only(raw: str) -> str:
    """The callback/stream URLs below prepend their own scheme, so a scheme in the env
    value yields https://https://host/... — which Twilio rejects (error 21609) and the
    call drops on connect. Accept either form."""
    return re.sub(r"^\w+://", "", raw.strip()).rstrip("/")


PUBLIC_HOST = _host_only(os.environ.get("PUBLIC_HOST", ""))
FROM_NUMBER = os.environ.get("TWILIO_FROM_NUMBER", "")
twilio_client = Client(
    os.environ.get("TWILIO_ACCOUNT_SID", "AC" + "0" * 32),
    os.environ.get("TWILIO_AUTH_TOKEN", "unset"),
)

RING_TIMEOUT = 25
CALL_TIME_LIMIT = 600
TERMINAL = {"completed", "no-answer", "busy", "failed", "canceled"}
VOICEMAIL = {"machine_start", "machine_end_beep", "machine_end_silence", "machine_end_other", "fax"}


def _create_call(cand: dict):
    # PUBLIC_HOST going stale (tunnel restarted, new ngrok URL) is the most common reason a
    # call connects and then drops on us, so make the host we're handing Twilio visible.
    logger.info("dialing %s — Twilio will call back on %s", cand["phone"], PUBLIC_HOST)
    twiml = (
        f'<Response><Connect><Stream url="wss://{PUBLIC_HOST}/twilio/media">'
        f'<Parameter name="candidate_id" value={quoteattr(cand["id"])}/>'
        f"</Stream></Connect></Response>"
    )
    return twilio_client.calls.create(
        to=cand["phone"],
        from_=FROM_NUMBER,
        twiml=twiml,
        # Async AMD: sync AMD holds the TwiML (and the opening line) until the
        # machine/human verdict — a silent answerer meant 10+ s of dead air. Async
        # starts the media stream at answer and posts AnsweredBy to the status
        # callback, which already hangs up on a voicemail verdict whenever it lands.
        machine_detection="Enable",
        async_amd=True,
        async_amd_status_callback=f"https://{PUBLIC_HOST}/twilio/status",
        async_amd_status_callback_method="POST",
        timeout=RING_TIMEOUT,
        time_limit=CALL_TIME_LIMIT,
        status_callback=f"https://{PUBLIC_HOST}/twilio/status",
        status_callback_event=["initiated", "ringing", "answered", "completed"],
        record=True,
        recording_channels="mono",
        recording_status_callback=f"https://{PUBLIC_HOST}/twilio/recording",
        recording_status_callback_event=["completed"],
    )


async def run_session():
    store.session["running"] = True
    try:
        if not store.FILLER_ULAW:  # pre-warm latency-masking clips once
            try:
                # the None entries mean an occasional silent beat instead of a spoken ack
                store.FILLER_ULAW = [await models.speak(p) for p in store.FILLER_PHRASES] + [None, None]
            except Exception:
                store.FILLER_ULAW = []
        for cand in store.candidates_list():
            if cand["status"] != "pending":
                continue
            store.session["current"] = cand["id"]
            store.session["call_done"] = asyncio.Event()
            opening = (
                store.config["opening_line"]
                .replace("[Candidate Name]", cand["name"])
                .replace("{candidate_name}", cand["name"])
                .replace("[Company Name]", store.config.get("company_name", ""))
                .replace("[Role Name]", store.config.get("role_name", ""))
            )
            cand["opening_text"] = opening
            cand["status"] = "calling"
            cand["started_at"] = time.time()
            try:
                # pre-synthesize so there's no dead air on pickup
                cand["opening_ulaw"] = await models.speak(opening)
                cand["opening_chars"] = len(opening)
                call = await asyncio.to_thread(_create_call, cand)
                cand["call_sid"] = call.sid
            except Exception:
                cand["status"] = "failed"
                cand["ended_at"] = time.time()
                continue
            try:
                await asyncio.wait_for(
                    store.session["call_done"].wait(), timeout=CALL_TIME_LIMIT + 60
                )
            except asyncio.TimeoutError:
                cand["status"] = "failed"  # watchdog: lost callback can't wedge the queue
            cand["ended_at"] = cand["ended_at"] or time.time()
            await asyncio.sleep(2)
    finally:
        store.session["running"] = False
        store.session["current"] = None


async def end_current_call(cid: str):
    """User-initiated hangup of whichever call is currently active."""
    cand = store.candidates.get(cid)
    if not cand:
        return
    if cand["call_sid"]:
        try:
            await asyncio.to_thread(
                lambda: twilio_client.calls(cand["call_sid"]).update(status="completed")
            )
        except Exception:
            pass  # call may already have ended on Twilio's side
    if cand["status"] in ("calling", "in_progress"):
        cand["status"] = "failed"
        cand["partial"] = True
    cand["ended_at"] = cand["ended_at"] or time.time()
    done = store.session.get("call_done")
    if done:
        done.set()  # unblocks run_session() to advance to the next candidate


async def on_recording_callback(form: dict):
    """Twilio finished recording the call — download it as MP3 and attach to the candidate."""
    cand = store.by_call_sid(form.get("CallSid", ""))
    if not cand or form.get("RecordingStatus") != "completed":
        return
    url = form.get("RecordingUrl", "")
    cand["recording_sid"] = form.get("RecordingSid")
    duration = form.get("RecordingDuration")
    cand["recording_duration"] = float(duration) if duration else None
    if not url:
        return
    os.makedirs("recordings", exist_ok=True)
    path = f"recordings/{cand['id']}.mp3"
    try:
        auth = (
            os.environ.get("TWILIO_ACCOUNT_SID", ""),
            os.environ.get("TWILIO_AUTH_TOKEN", ""),
        )
        async with httpx.AsyncClient(timeout=30) as http:
            r = await http.get(url + ".mp3", auth=auth)
            r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        cand["recording_path"] = path
    except Exception:
        pass  # recording stays unavailable in the UI; not fatal to the call record


async def on_status_callback(form: dict):
    cand = store.by_call_sid(form.get("CallSid", ""))
    if not cand:
        return
    answered_by = form.get("AnsweredBy")
    if answered_by:
        cand["answered_by"] = answered_by
        if answered_by in VOICEMAIL and cand["status"] in ("calling", "in_progress"):
            cand["status"] = "no_answer"
            try:
                await asyncio.to_thread(
                    lambda: twilio_client.calls(cand["call_sid"]).update(status="completed")
                )
            except Exception:
                pass
    status = form.get("CallStatus", "")
    if status in TERMINAL:
        if cand["status"] in ("calling", "in_progress"):
            cand["status"] = {
                "completed": "completed",
                "no-answer": "no_answer",
                "busy": "no_answer",
                "failed": "failed",
                "canceled": "failed",
            }[status]
        cand["ended_at"] = time.time()
        done = store.session.get("call_done")
        if done:
            done.set()
