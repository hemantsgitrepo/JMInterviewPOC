# Phase 3 Prompt: Configurable Telephony Provider

Status: **deferred, not started**. Use the prompt below verbatim (or lightly adapted) to
kick off Phase 3 in a future session. It is self-contained — the session running it will
have no memory of this one.

---

## Prompt to use

> Implement Phase 3 of the configurable-settings plan: make the Twilio telephony
> provider configurable via the Settings tab, following the exact pattern already
> established for LLM/STT/TTS providers in `settings.py`. Read
> `docs/settings-module-design.md` (architecture context) and this file's
> "Grounding" section below (exact coupling points) before writing any code. Work on
> a new branch off `main` named `phase-3-telephony-provider`; do not touch `v0.1.0`
> or any other tag.
>
> **Scope for this pass:** add ONE additional telephony provider alongside Twilio
> (default) — a Twilio-compatible provider (SignalWire or Telnyx) — not a
> from-scratch abstraction over an incompatible provider. Confirm with the user
> which one before implementing, since it determines exact API shape; SignalWire's
> LAML dialect is closer to Twilio's TwiML than Telnyx's TeXML, so it is the lower-risk
> default recommendation absent other constraints.
>
> **Required outcome, mirroring the existing pattern exactly:**
> 1. A `VETTED_TELEPHONY_PROVIDERS` list in `settings.py`, same shape as
>    `VETTED_STT_PROVIDERS`/`VETTED_TTS_PROVIDERS` (id, label, key_env), validated with
>    `_validate_provider` so a provider whose `.env` key is absent can never be selected.
> 2. `telephony_provider()` accessor + `DEFAULT_SETTINGS["telephony_provider"] = "twilio"`.
> 3. A `dialer.py` dispatch exactly like `models.py`'s `transcribe()`/`speak()`: the
>    Twilio code path becomes the default branch, byte-for-byte unchanged; the new
>    provider is an additive function alongside it, never a replacement.
> 4. Settings UI: one more provider dropdown in the Settings card, same
>    `fillProviderSelect()` mechanism already used for STT/TTS in `static/index.html`.
> 5. `docs/settings-module-design.md` updated to mark Phase 3 done, same as the
>    Phase 2 rows.
> 6. Tests: extend `test_settings.py` for the new provider's validation, add a live
>    round-trip test analogous to `test_providers.py` (skip cleanly if the new
>    provider's key is absent), and re-run the full existing suite
>    (`test_guards.py`, `test_end_confirm.py`, `test_closing_qa.py`, `test_langsmith_noop.py`,
>    the loopback scripts) to confirm zero regression to the Twilio default path.
>
> **Follow this repo's established workflow:** one phase, one branch, commit
> frequently with descriptive messages, run the full automated suite before handing
> back, then STOP and wait for explicit manual-test approval before merging — do not
> merge on your own judgment. Report progress at each major step.

---

## Grounding: exact coupling points as of v0.2.0

This is what makes telephony harder than the STT/TTS/LLM adapters that shipped in
Phases 1–2, and exactly where the seams are.

### Why this one is different

STT/TTS/LLM adapters wrap **one HTTP call each** (`transcribe()`, `speak()`,
`next_turn()`) behind a clean function contract. Telephony is not one call — it's
**call placement + inbound audio framing + outbound playback-completion signaling +
answering-machine detection + status/recording callbacks**, all riding on a
provider-specific wire protocol that `call.py`'s state machine was written directly
against. There is no existing seam to dispatch on; one must be built.

### Everything that currently assumes Twilio

**`dialer.py`** (the whole file is Twilio-specific):
- `twilio_client = Client(...)` ([dialer.py:32](dialer.py:32)) — the SDK client itself
- `_create_call()` ([dialer.py:43](dialer.py:43)) builds Twilio TwiML inline
  (`<Response><Connect><Stream url="wss://.../twilio/media">...`) and calls
  `twilio_client.calls.create(...)` with Twilio-specific kwargs: `machine_detection`,
  `async_amd`, `async_amd_status_callback`, `status_callback_event`, `recording_channels`,
  `recording_status_callback_event`
- `end_current_call()`, `hangup()` in `call.py` ([call.py:535](call.py:535)) —
  `twilio_client.calls(sid).update(status="completed")`
- `on_recording_callback()` ([dialer.py:154](dialer.py:154)) — downloads via
  `RecordingUrl + ".mp3"` with Twilio basic-auth (account SID + auth token), a
  Twilio-specific URL convention
- `on_status_callback()` ([dialer.py:182](dialer.py:182)) — parses Twilio's specific
  form fields: `CallSid`, `CallStatus`, `AnsweredBy`, maps Twilio's status vocabulary
  (`no-answer`, `busy`, `canceled`, ...) to internal candidate statuses
- `VOICEMAIL` set ([dialer.py:40](dialer.py:40)) — Twilio's specific AMD verdict strings
  (`machine_start`, `machine_end_beep`, ...)
- `PUBLIC_HOST` normalization ([dialer.py:30](dialer.py:30)) is provider-agnostic and can
  stay as-is

**`app.py`** — three routes are Twilio's specific webhook contract:
- `POST /twilio/status` ([app.py:336](app.py:336)) → `dialer.on_status_callback(dict(await request.form()))`
- `POST /twilio/recording` ([app.py:342](app.py:342)) → `dialer.on_recording_callback(...)`
- `WS /twilio/media` ([app.py:348](app.py:348)) → `handle_media_ws(ws)`, and the
  `BasicAuthMiddleware` exemption at [app.py:46](app.py:46) (`scope["path"].startswith("/twilio/")`)
  exists BECAUSE these are Twilio's own servers hitting the app, not a browser — a new
  provider's callback paths need the same exemption

**`call.py`** — the deepest coupling, in the media WebSocket protocol itself:
- `handle_media_ws()` / `CallSession` ([call.py:572](call.py:572) onward) parse Twilio
  Media Streams' specific JSON event shape: `{"event": "start", "start": {"streamSid":
  ..., "customParameters": {"candidate_id": ...}}}`, `{"event": "media", "media":
  {"payload": <base64 mulaw>}}`, `{"event": "mark", "mark": {"name": ...}}`, `{"event": "stop"}`
- **Mark echo is load-bearing, not cosmetic.** `enqueue()` ([call.py:156](call.py:156))
  sends a `mark` event after each audio chunk; `on_mark()` ([call.py:249](call.py:249))
  only knows a clause finished playing, or the whole turn finished, when Twilio echoes
  that mark back. Barge-in truncation (`clauses_played` bookkeeping), the deferred-ack
  timing, and the end-of-call hangup grace period (`asyncio.sleep(0.6)` at
  [call.py:265](call.py:265), justified by the comment that a mark only confirms
  Twilio received the audio, not that the PSTN leg played it) all depend on this
  signal existing. **A provider without an equivalent mark-echo mechanism needs an
  emulated one** (e.g. estimate playback duration from audio byte count and fire a
  synthetic "done" event after a timer) — this is the single riskiest piece of a full
  abstraction and is why this document scopes Phase 3 down to a Twilio-**compatible**
  provider (SignalWire/Telnyx), which do echo marks, rather than an incompatible one
  (Vonage, Plivo) that would require building the emulation from scratch.
- Audio format assumption: 8kHz μ-law in both directions
  (`ulaw_to_pcm`/`pcm_to_ulaw` in `audio.py`) — SignalWire/Telnyx match this; a
  different provider might not
- `on_start()` ([call.py:186](call.py:186)) reads `customParameters.candidate_id` — the
  mechanism for passing the candidate ID into the stream is TwiML's `<Parameter>` tag;
  a compatible provider needs an equivalent

### What's genuinely reusable across a Twilio-compatible switch

- `audio.py` (VAD, μ-law/PCM conversion, resampling) — unchanged
- The entire `CallSession` conversational state machine (guards, barge-in, silence
  tiers, closing sequence) — unchanged, AS LONG AS the new provider echoes marks in
  the same shape
- `store.py`, `settings.py` provider-picker pattern — unchanged
- Recording download only needs a URL + auth-scheme adapter, not new logic

### Provider prerequisites (already researched, see `docs/settings-module-design.md` §7)

| Provider | Credentials needed | Note |
|---|---|---|
| SignalWire (recommended first) | Project ID, API token, Space URL, number purchase | LAML — closest to Twilio's TwiML; media streams supported |
| Telnyx | API key, TeXML application, number purchase | TeXML is a Twilio-compat layer; typically cheaper |
| Exotel (India) | Account SID, key/token, ExoPhone, KYC/DLT | Explicitly out of scope — different streaming model, heavier lift |

None of these credentials exist in `.env` yet — Phase 3 must add them and document in
`.env.example`, following the same "selection rejected while key absent" rule already
enforced for Deepgram/OpenAI in `settings.py`'s `_validate_provider`.

### Explicitly out of scope for this pass

- A generic N-provider abstraction interface — build against the second provider only;
  generalize on a third if one is ever added (YAGNI)
- Any provider requiring mark-echo emulation (Vonage, Plivo) or a fundamentally
  different streaming model (Exotel)
- Per-call provider selection — mirrors the existing settings model (global, applies
  to the next call)
