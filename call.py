"""Agent orchestrator: full-duplex state machine over the Twilio media WebSocket.

States: SPEAKING (audio queued/playing) -> LISTENING (VAD) -> PROCESSING (STT->LLM->TTS).
Full-duplex: inbound audio is watched even while SPEAKING so the caller can barge in.
All outbound sends funnel through a single `player` task (one writer -> no ws races)."""

import asyncio
import base64
import json
import os
import random
import re
import time

from fastapi import WebSocket, WebSocketDisconnect

import audio
import models
import store
from audio import BargeInDetector, Endpointer, pcm_to_wav, ulaw_to_pcm

LISTEN_TIMEOUT_MS = 4500  # wait 4-5s of total silence before the "still there?" reprompt
LISTEN_TIMEOUT_FRAMES = LISTEN_TIMEOUT_MS // 20
CHUNK = 4000  # mu-law bytes per outbound ws message (~0.5s)
MAX_FOLLOWUPS = 2
NOISE_MULT = 3.0       # live speech threshold = noise_floor * this
BARGE_MARGIN = 1.4     # barge-in needs to clear the listen threshold by this factor (resists echo)
THRESHOLD_CEIL = 2500  # clamp so a noisy burst can't blind the VAD

SYSTEM_PROMPT = """You are an automated phone interviewer for {company_name}, speaking with
{candidate_name} on a live voice call. You must sound like a warm, natural human on the phone.

WRITE FOR THE EAR, NOT THE EYE:
- Keep each reply under 25 words. Never monologue. One thought at a time.
- Use everyday contractions ("I'm", "you're", "we'll", "that's") and simple, spoken phrasing.
- Never use markdown, bullet points, lists, digits-as-symbols, or complex punctuation.
- Do NOT open with a filler acknowledgment ("Right,", "Got it,", "Okay,", "Mm-hmm,") - the system
  already plays a short verbal acknowledgment before you respond, so starting with another one
  makes it sound doubled and repetitive. Go straight into your actual sentence.
- Split your reply into short spoken clauses separated by a double pipe "||". Put the single most
  important part (usually the actual question) in the FIRST clause, since the caller may interrupt.
  Example reply: "That makes sense. || So, what languages are you most comfortable with?"

Interview questions, in order:
{questions}

RULES:
- A [STATUS] note at the end of each candidate message tells you which question you're on. Ask that
  one question at a time, and acknowledge their answer briefly before moving to the next.
- If an answer is off-topic or unclear, or they ask something you can't answer, gently redirect once,
  then move on. Never invent facts about {company_name}, the salary, or the role.
- If the candidate wants to stop, is hostile, or clearly wants to end, wrap up politely.
- When all questions are answered or the call must end, use action "end_call". Do NOT speak a closing
  line yourself — the system plays a configured one.

Respond ONLY with JSON: {{"reply": "<what you say next, with || clause breaks>", "action": "stay" | "ask_next" | "end_call"}}
- "ask_next": the current question was answered; your reply acknowledges it and asks the NEXT question.
- "stay": you need a clarification or follow-up on the CURRENT question.
- "end_call": interview complete or should be terminated."""


def split_clauses(text: str) -> list[str]:
    """Break an LLM reply into short spoken clauses: on '||' if present, else sentences."""
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in text.split("||") if p.strip()]
    if len(parts) <= 1:
        parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return parts or [text]


class CallSession:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.stream_sid = None
        self.cand = None
        self.config = None
        self.state = "SPEAKING"
        self.endpointer = Endpointer()
        self.barge = BargeInDetector()
        self.question_index = -1  # -1 = no question asked yet (see status_line)
        self.followups = 0
        self.history: list[dict] = []
        self.silent_frames = 0
        self.reprompted = False
        self.ending = False
        self.opening_in_progress = False  # protects the opening line from barge-in
        # adaptive VAD
        self.noise_ema = audio.SPEECH_RMS / NOISE_MULT
        self.threshold = audio.SPEECH_RMS
        # outbound audio pipeline
        self.out_q: asyncio.Queue = asyncio.Queue()
        self.player_task: asyncio.Task | None = None
        self.turn_task: asyncio.Task | None = None
        # current agent turn, for barge-in truncation + deferred commit
        self.clauses: list[str] = []
        self.clauses_played = 0
        self.pending = None        # (next_index, next_followups) committed on clean turn end
        self._last_agent_idx = None
        self._last_hist_idx = None
        # usage/cost tracking (STT + LLM cost is real, from OpenRouter; TTS chars are a proxy
        # since Cartesia's /tts/bytes doesn't return cost)
        self.usage = {
            "stt": {"calls": 0, "seconds": 0.0, "cost": 0.0, "cost_known": True},
            "llm": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "cost_known": True},
            "tts": {"calls": 0, "characters": 0},
        }

    # ---- outbound pipeline (single writer) ------------------------------

    async def player(self):
        """Sole websocket writer. Serializes every outbound frame."""
        try:
            while True:
                kind, payload = await self.out_q.get()
                if kind == "media":
                    for i in range(0, len(payload), CHUNK):
                        await self.ws.send_json({
                            "event": "media", "streamSid": self.stream_sid,
                            "media": {"payload": base64.b64encode(payload[i:i + CHUNK]).decode()},
                        })
                elif kind == "mark":
                    await self.ws.send_json({
                        "event": "mark", "streamSid": self.stream_sid, "mark": {"name": payload},
                    })
                elif kind == "clear":
                    await self.ws.send_json({"event": "clear", "streamSid": self.stream_sid})
                self.out_q.task_done()
        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            pass

    def enqueue(self, ulaw: bytes, mark: str):
        self.out_q.put_nowait(("media", ulaw))
        self.out_q.put_nowait(("mark", mark))

    def flush_out(self):
        """Drop everything not yet sent, then tell Twilio to discard its buffer."""
        while not self.out_q.empty():
            self.out_q.get_nowait()
            self.out_q.task_done()
        self.out_q.put_nowait(("clear", None))

    async def deliver(self, clauses: list[str], closing_mark: str):
        """Synthesize clauses one at a time and stream them out; first audio starts
        after clause 1, not after the whole reply. Bails if barged mid-synthesis."""
        self.state = "SPEAKING"
        self.clauses = clauses
        self.clauses_played = 0
        self.cand["transcript"].append({"role": "agent", "text": " ".join(clauses), "at": time.time()})
        self._last_agent_idx = len(self.cand["transcript"]) - 1
        for i, clause in enumerate(clauses):
            ulaw = await models.speak(clause)
            self.usage["tts"]["calls"] += 1
            self.usage["tts"]["characters"] += len(clause)
            if self.state != "SPEAKING":  # barged in during synthesis
                return
            self.enqueue(ulaw, f"c{i}")
        self.out_q.put_nowait(("mark", closing_mark))

    # ---- twilio events ---------------------------------------------------

    async def on_start(self, msg: dict):
        start = msg["start"]
        self.stream_sid = start["streamSid"]
        cid = start.get("customParameters", {}).get("candidate_id")
        self.cand = store.candidates[cid]
        self.cand["status"] = "in_progress"
        self.config = dict(store.config)  # snapshot
        opening = self.cand.get("opening_text") or self.config["opening_line"]
        self.cand["transcript"].append({"role": "agent", "text": opening, "at": time.time()})
        self.history.append({"role": "assistant", "content": opening})
        self.state = "SPEAKING"
        self.opening_in_progress = True  # always finish the opening line, even if talked over
        self.clauses = [opening]
        self.clauses_played = 0
        ulaw = self.cand.get("opening_ulaw")
        if ulaw:
            self.usage["tts"]["calls"] += 1
            self.usage["tts"]["characters"] += self.cand.get("opening_chars", len(opening))
        else:
            ulaw = await models.speak(opening)
            self.usage["tts"]["calls"] += 1
            self.usage["tts"]["characters"] += len(opening)
        self.enqueue(ulaw, "turn_done")

    async def on_media(self, msg: dict):
        if self.state == "ENDING":
            return  # goodbye line has played; just waiting out the hangup grace period
        frame = base64.b64decode(msg["media"]["payload"])
        self._update_noise(frame)
        if self.state == "SPEAKING":
            if self.opening_in_progress or self.ending:
                return  # opening line and end-of-call line always play out fully
            onset = self.barge.feed(frame, self.threshold * BARGE_MARGIN)
            if onset is not None:
                await self.handle_barge_in(onset)
        elif self.state == "LISTENING":
            utt = self.endpointer.feed(frame, self.threshold)
            if utt is not None:
                self.start_turn(utt)
                return
            if self.endpointer.speaking:
                self.silent_frames = 0
            else:
                self.silent_frames += 1
                if self.silent_frames >= LISTEN_TIMEOUT_FRAMES:
                    self.silent_frames = 0
                    if self.reprompted:
                        self.start_line("It seems we got disconnected. " + self.config["end_call_line"], "end")
                    else:
                        self.reprompted = True
                        self.start_line("Are you still there?", "turn_done")
        # PROCESSING: ignore inbound (brief window between utterance end and first audio)

    async def on_mark(self, msg: dict):
        name = msg.get("mark", {}).get("name", "")
        if name.startswith("c"):        # a clause finished playing
            self.clauses_played += 1
        elif name == "turn_done":       # agent's turn fully heard -> commit + listen
            self.opening_in_progress = False
            self.commit_pending()
            self.state = "LISTENING"
            self.silent_frames = 0
            self.endpointer.reset()
            self.barge.reset()
        elif name == "end":
            # Twilio's mark only confirms audio was handed to the telephony leg, not that
            # it has actually reached the caller's ear (carrier/PSTN buffering adds more).
            # Wait a beat so the goodbye line doesn't get clipped by an immediate hangup.
            self.state = "ENDING"
            await asyncio.sleep(0.6)
            await self.hangup()
        # "filler" and anything else: no state change

    # ---- turn taking -----------------------------------------------------

    def start_turn(self, utt: bytes):
        self.state = "PROCESSING"
        self.turn_task = asyncio.create_task(self.process_utterance(utt))

    def start_line(self, text: str, mark: str):
        self.state = "PROCESSING"
        if mark == "end":
            self.ending = True
        self.turn_task = asyncio.create_task(self.deliver(split_clauses(text), mark))

    async def handle_barge_in(self, onset_frames: list[bytes]):
        # stop the in-flight turn (STT/LLM/TTS or clause synthesis)
        if self.turn_task and not self.turn_task.done():
            self.turn_task.cancel()
            try:
                await self.turn_task
            except (asyncio.CancelledError, Exception):
                pass
        self.flush_out()  # drop queued audio + Twilio buffer
        # reconcile: the caller only heard up to the clause playing when they cut in
        heard = " ".join(self.clauses[: max(1, self.clauses_played + 1)]) if self.clauses else ""
        self._truncate_agent(heard)
        self.pending = None  # interrupted turn never committed -> stay on same question
        self.state = "LISTENING"
        self.silent_frames = 0
        self.endpointer.reset()
        self.barge.reset()
        for f in onset_frames:  # don't lose the start of their interruption
            self.endpointer.feed(f, self.threshold)

    async def process_utterance(self, utt: bytes):
        try:
            # new turn: clear stale clause bookkeeping so a barge during the filler/think
            # window can't truncate the previous agent turn's transcript entry
            self.clauses = []
            self._last_agent_idx = None
            if store.FILLER_ULAW:  # mask latency: acknowledge instantly while we think
                self.state = "SPEAKING"
                self.enqueue(random.choice(store.FILLER_ULAW), "filler")
            t0 = time.monotonic()
            stt = await models.transcribe(pcm_to_wav(ulaw_to_pcm(utt), 8000))
            t1 = time.monotonic()
            self.usage["stt"]["calls"] += 1
            if stt.get("seconds") is not None:
                self.usage["stt"]["seconds"] += stt["seconds"]
            if stt.get("cost") is not None:
                self.usage["stt"]["cost"] += stt["cost"]
            else:
                self.usage["stt"]["cost_known"] = False
            text = stt["text"]
            if not text:
                self.state = "LISTENING"
                self.endpointer.reset()
                self.barge.reset()
                return
            self.reprompted = False
            self.cand["transcript"].append({"role": "candidate", "text": text, "at": time.time()})
            self.history.append({"role": "user", "content": text + self.status_line()})
            turn, llm_usage = await models.next_turn(self.system_prompt(), self.history)
            t2 = time.monotonic()
            self.usage["llm"]["calls"] += 1
            self.usage["llm"]["prompt_tokens"] += llm_usage.get("prompt_tokens") or 0
            self.usage["llm"]["completion_tokens"] += llm_usage.get("completion_tokens") or 0
            if llm_usage.get("cost") is not None:
                self.usage["llm"]["cost"] += llm_usage["cost"]
            else:
                self.usage["llm"]["cost_known"] = False
            self.history.append({"role": "assistant", "content": json.dumps(turn)})
            self._last_hist_idx = len(self.history) - 1
            clauses, mark = self._plan_turn(turn)
            print(f"[latency] cand={self.cand['id']} stt={(t1-t0)*1000:.0f}ms "
                  f"llm={(t2-t1)*1000:.0f}ms action={turn['action']}", flush=True)
            await self.deliver(clauses, mark)
        except asyncio.CancelledError:
            raise  # barge-in cancelled us; handle_barge_in owns cleanup
        except Exception:
            self.cand["status"] = "failed"
            self.ending = True  # protect the closing line from barge-in like any other end-of-call
            try:
                await self.deliver(
                    split_clauses(
                        "I'm sorry, we're having a technical issue. || "
                        + self.config["end_call_line"]
                    ),
                    "end",
                )
            except Exception:
                await self.hangup()

    def _plan_turn(self, turn: dict):
        """Decide clauses + closing mark, and stash the index change to commit on clean end."""
        action, reply = turn["action"], turn.get("reply", "")
        qs = self.config["questions"]
        next_i, next_f, effective = self.question_index, self.followups, action
        if action == "ask_next":
            next_i, next_f = self.question_index + 1, 0
            if next_i >= len(qs):
                effective = "end_call"
        elif action == "stay":
            next_f = self.followups + 1
            if next_f > MAX_FOLLOWUPS:
                next_i, next_f = self.question_index + 1, 0
                if next_i >= len(qs):
                    effective = "end_call"
        if effective == "end_call":
            self.ending = True
            self.pending = None
            end_line = self.config["end_call_line"]
            return split_clauses((reply + " || " + end_line) if reply else end_line), "end"
        self.pending = (next_i, next_f)
        return split_clauses(reply), "turn_done"

    def commit_pending(self):
        if self.pending is not None:
            self.question_index, self.followups = self.pending
            self.pending = None

    def _truncate_agent(self, heard: str):
        """After a barge-in, correct the record to what the caller actually heard."""
        if self._last_agent_idx is not None:
            self.cand["transcript"][self._last_agent_idx]["text"] = (heard + " [interrupted]").strip()
        if self._last_hist_idx is not None and self._last_hist_idx < len(self.history):
            self.history[self._last_hist_idx]["content"] = json.dumps(
                {"reply": heard, "action": "interrupted"}
            )

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            company_name=self.config["company_name"],
            candidate_name=self.cand["name"],
            questions="\n".join(f"{i+1}. {q}" for i, q in enumerate(self.config["questions"])),
        )

    def status_line(self) -> str:
        qs = self.config["questions"]
        if self.question_index < 0:
            return "\n\n[STATUS] No interview question has been asked yet. Ask the FIRST question now."
        i = min(self.question_index, len(qs) - 1)
        note = " This is the LAST question." if i == len(qs) - 1 else ""
        if self.followups >= MAX_FOLLOWUPS:
            note += " No more follow-ups allowed on this question; move on."
        return f'\n\n[STATUS] You are on question {i+1} of {len(qs)}: "{qs[i]}".{note}'

    def _update_noise(self, frame: bytes):
        r = audio.rms(frame)
        if r < self.threshold:  # treat as ambient; drift the floor toward it
            self.noise_ema = 0.97 * self.noise_ema + 0.03 * r
            self.threshold = min(THRESHOLD_CEIL, max(audio.SPEECH_RMS, self.noise_ema * NOISE_MULT))

    # ---- teardown --------------------------------------------------------

    async def hangup(self):
        sid = self.cand.get("call_sid") if self.cand else None
        if not sid:
            await self.ws.close()  # loopback mode: no real Twilio call to end
            return
        try:
            from dialer import twilio_client

            await asyncio.to_thread(lambda: twilio_client.calls(sid).update(status="completed"))
        except Exception:
            pass

    def finish(self):
        if not self.cand:
            return
        if self.cand["status"] == "in_progress":
            self.cand["status"] = "completed"
            self.cand["partial"] = not self.ending
        self.cand["usage"] = self.usage
        known_cost, any_known = 0.0, False
        if self.usage["stt"]["calls"] and self.usage["stt"]["cost_known"]:
            known_cost += self.usage["stt"]["cost"]
            any_known = True
        if self.usage["llm"]["calls"] and self.usage["llm"]["cost_known"]:
            known_cost += self.usage["llm"]["cost"]
            any_known = True
        self.cand["total_cost"] = known_cost if any_known else None
        os.makedirs("transcripts", exist_ok=True)
        with open(f"transcripts/{self.cand['id']}.json", "w") as f:
            json.dump(
                {
                    "candidate": {k: self.cand[k] for k in ("id", "name", "phone", "status", "partial")},
                    "config_snapshot": self.config,
                    "turns": self.cand["transcript"],
                },
                f,
                indent=2,
            )
        done = store.session.get("call_done")
        if done:
            done.set()


async def handle_media_ws(ws: WebSocket):
    await ws.accept()
    sess = CallSession(ws)
    sess.player_task = asyncio.create_task(sess.player())
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            event = msg.get("event")
            if event == "start":
                await sess.on_start(msg)
            elif event == "media":
                await sess.on_media(msg)
            elif event == "mark":
                await sess.on_mark(msg)
            elif event == "stop":
                break
    except (WebSocketDisconnect, RuntimeError, KeyError):
        pass
    finally:
        for t in (sess.turn_task, sess.player_task):
            if t and not t.done():
                t.cancel()
        sess.finish()
