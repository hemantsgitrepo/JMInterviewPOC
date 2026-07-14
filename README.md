# Jobmanch.ai — AI Interview Caller (POC)

Sequentially calls candidates via Twilio, conducts a configurable AI interview over a live
bidirectional Media Stream, and captures transcripts. STT and the interview LLM run through
**OpenRouter**; TTS runs through **Cartesia** (Sonic 3.5).

Architecture doc: see the approved plan. Model constants live at the top of `models.py`:
STT `openai/whisper-large-v3-turbo` (OpenRouter), LLM `google/gemini-2.5-flash` (OpenRouter),
TTS `sonic-3.5` (Cartesia, voice id configurable via `CARTESIA_VOICE_ID`).

## Run

```bash
uv venv && uv pip install -r requirements.txt
cp .env.example .env          # fill in Twilio + OpenRouter + Cartesia creds
ngrok http 8010               # put the hostname (no scheme) in .env as PUBLIC_HOST
.venv/bin/uvicorn app:app --port 8010
open http://localhost:8010
```

(Port 8010, not 8000 — 8000 is commonly taken by Docker Desktop on macOS.)

Configure the interview (add questions manually or generate them from a pasted/uploaded job
description), paste candidates (`Name, +14155551234` per line), hit **Start Calling**. Transcripts
land in `transcripts/*.json`, call recordings in `recordings/*.mp3`, both browsable in the UI via
each candidate's **Info** button.

## Checks

```bash
.venv/bin/python -m pytest test_audio.py    # codec / VAD / endpointing / barge / clause-split
.venv/bin/python loopback.py                # full STT->LLM->TTS loop, no phone call
.venv/bin/python test_bargein.py            # interrupt the agent mid-reply, assert Twilio `clear`
                                            # (all need OPENROUTER_API_KEY + CARTESIA_API_KEY + running server)
```

## Conversation quality (Phase 0 + 1)

- **Barge-in:** the caller can talk over the agent. Full-duplex — inbound audio is watched even while the agent speaks; a sustained onset (120ms) sends Twilio `clear`, cancels in-flight synthesis, and truncates the agent turn to what was actually heard (`[interrupted]`). The interview question index only commits after a reply is fully heard, so an interrupted question isn't skipped.
- **Per-clause playback:** LLM replies are split on `||` (see the system prompt) and synthesized/streamed clause-by-clause, so first audio starts after clause 1 instead of the whole reply.
- **Latency masking:** a short "Mm-hmm."/"Right." clip plays the instant the caller stops, covering STT+LLM+TTS time. Pre-synthesized once per session.
- **Ear-writing prompt:** short spoken clauses, contractions, acknowledgment tokens, no markdown.
- **Adaptive VAD:** the speech threshold tracks the measured noise floor instead of a fixed constant.
- **Latency logging:** every turn prints `[latency] stt=..ms llm=..ms action=..` to the server log.

Not yet done (Phase 2/3, see the improvement plan): LLM token streaming, Cartesia WebSocket streaming (we do per-clause batch calls today), streaming STT / semantic end-of-turn / back-channel filtering (would need Deepgram).

## Question management

- **Manual:** Add/Edit/Delete rows in the Interview Questions table. No default questions are seeded — the list starts empty.
- **AI-generated:** paste a job description (or upload a `.txt` file) into the JD box and click **Generate Questions with AI**. The same LLM call both validates the text looks like a JD and drafts 5–8 tailored questions — if it doesn't look like a JD, the API returns a 400 with the model's stated reason instead of guessing. Generated questions are appended to the table, not replacing manual ones. Only `.txt` upload is supported (no PDF/DOCX parsing — that would need an added dependency).

## Call recording & cost tracking

- **Recording:** each Twilio call is recorded server-side (`record=True` on `calls.create`), and downloaded as MP3 to `recordings/{candidate_id}.mp3` once Twilio's recording-status webhook (`/twilio/recording`) reports `completed`. Playable/downloadable from each candidate's **Info** panel.
- **Cost:** STT and LLM cost are **real dollar figures from OpenRouter** (`/audio/transcriptions` returns `usage.cost` natively; `/chat/completions` returns it via `usage: {include: true}`) — not estimated. Cartesia's `/tts/bytes` doesn't return cost, so TTS usage is shown as characters synthesized, honestly labeled "not reported by API" rather than guessing a number. "Total known cost" sums only the components with real cost data.
- The **Info** button (Interview Results table) replaces the earlier Retry button and opens this detail panel: status, duration, start/end timestamps, answered-by, recording player, and the cost/usage breakdown above. (The retry API endpoint itself — `POST /api/candidates/{id}/retry` — is still there if you want to wire a retry action back in; it's just no longer surfaced in this table.)

## Notes

- Calls are recorded/AI-disclosed in the default opening line — only dial consenting test numbers (TCPA / two-party consent).
- POC scope: no auth, no DB (in-memory + transcript/recording files), no Twilio webhook signature validation.
- Twilio trial accounts can only call verified numbers.
- Twilio webhooks (`/twilio/status`, `/twilio/recording`) parse form data, which requires `python-multipart` — now in `requirements.txt` (it was missing before; without it those endpoints would 500 on a real call).
