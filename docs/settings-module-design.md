# Design: Configurable Settings Module

Status: Phase 1 in progress · Baseline: v0.1.0 (`671f7b2`) + main fixes (`dfa0082`)

## 1. Purpose

Add a Settings module that lets an authorised admin, at runtime and without code changes:

- switch the **LLM model** used for live call turns (and JD parsing / question generation),
- edit the **system prompt persona** (e.g. repurpose interviewer → sales agent),
- add **supplementary instructions / guardrails**,
- (Phase 2) switch **STT** and **TTS** providers,
- (Phase 3, optional) switch **telephony** providers.

**Hard requirement:** the existing implementation is the default state and must remain
byte-for-byte unaffected when settings are absent or unmodified.

## 2. Non-goals (v1)

- Streaming STT (the pipeline is batch-per-utterance by design; see `audio.py` VAD endpointing).
- Storing API keys in the Settings UI or `settings.json` (secrets stay in `.env`, always).
- Per-call provider overrides (settings are global; benchmarking = change setting between calls).

## 3. Current-state seams (why this is feasible)

| Component | Seam | Coupling |
|---|---|---|
| LLM | `models.next_turn(system, messages)` — OpenRouter model string | Trivial: OpenRouter is already multi-provider |
| STT | `models.transcribe(wav) -> {text, seconds, cost}` | Clean function contract; adapters slot in (Phase 2) |
| TTS | `models.speak(text) -> 8kHz μ-law bytes` | Clean; `audio.resample()` + `pcm_to_ulaw()` already exist (Phase 2) |
| Telephony | `call.py` state machine ⇄ Twilio Media Streams protocol | Deep: mark echo, `<Connect><Stream>`, AMD, callbacks (Phase 3) |
| Prompt | `CallSession.system_prompt()` formats one template | Easy, with the protocol-lock below |

## 4. Architecture

### 4.1 Settings store (`settings.py`, new)

- `DEFAULT_SETTINGS` pins today's exact values:
  `llm_model = "google/gemini-2.5-flash"`, `prompt_template` = the current prompt's
  editable region verbatim, `extra_instructions = ""`.
- Persisted to `settings.json` (repo root, **gitignored** — runtime state like `transcripts/`).
  Loaded at import; unknown keys dropped; corrupt file ⇒ defaults + logged error.
- Atomic writes (tmp + rename) under a lock.
- **No settings file ⇒ identical behavior to pre-module code.** This is asserted by test.

### 4.2 Prompt split: editable template + locked protocol

The v0.1.0 system prompt (`call.py:39-103`) splits at the `Respond ONLY with JSON:` line:

- **Editable `prompt_template`** — persona, "write for the ear" style rules, question-list
  block, conduct, human-handling. Placeholders: `{company_name}`, `{candidate_name}`,
  `{questions}` (numbered list injected by code).
- **Locked `PROMPT_PROTOCOL`** (code constant, never editable, always appended last) — the
  JSON action contract (`stay | ask_next | skip | repeat | end_call`) and the
  never-ask-and-hang-up rule. `_plan_turn()` and the end-confirmation guard parse these;
  an edit that removed them would break every call at the first turn.
- `extra_instructions` (if non-empty) is inserted between template and protocol under an
  `ADDITIONAL INSTRUCTIONS:` header.
- `behavior.confirm_key_facts` still appends `CONFIRM_KEY_FACTS_RULE` after the protocol,
  unchanged.

Validation on save: balanced braces; only allowed placeholders; `{questions}` required;
non-empty. Runtime belt-and-braces: if a stored template fails to `.format()`, the call
falls back to the default template and logs — a bad prompt must never kill a live call.

Known degradation (accepted, documented): the editable region includes the `[STATUS]`
explanation and the `||` clause-split instruction. Removing `||` degrades gracefully
(`split_clauses()` falls back to sentence splitting); removing the `[STATUS]` bullet
leaves status notes unexplained to the model but never breaks parsing.

### 4.3 LLM model picker

`models.py` reads the model string from settings at call time (default branch = the same
constant as today). Only **vetted** model IDs are accepted — each verified live on
OpenRouter for existence + `response_format` (JSON mode) support:

| ID | Notes |
|---|---|
| `google/gemini-2.5-flash` | **Default** — current baseline |
| `google/gemini-2.5-flash-lite` | Fastest/cheapest Google |
| `anthropic/claude-haiku-4.5` | Anthropic latency tier |
| `anthropic/claude-sonnet-4.5` | Higher quality, more latency/cost |
| `openai/gpt-4o-mini` | OpenAI mini tier |
| `openai/gpt-4.1-mini` | OpenAI mini tier, newer |
| `meta-llama/llama-3.3-70b-instruct` | Many providers, low latency |
| `mistralai/mistral-small-3.2-24b-instruct` | Budget option |

Applies to `next_turn`, `parse_jd`, `generate_questions_from_jd` (one knob, consistent
benchmarking). No new credentials: everything routes through the existing OpenRouter key.

### 4.4 Admin gating

Mirrors the existing auth philosophy (`BasicAuthMiddleware`: enforced only when env is set):

- `ADMIN_PASS` env var set ⇒ `POST /api/settings*` requires `X-Admin-Pass` header
  (constant-time compare). Unset ⇒ open, for local dev.
- Reads (`GET /api/settings`) stay behind the normal site auth only.
- Secrets policy: `settings.json` holds provider *choices* only, never keys. The UI never
  sees key material.

### 4.5 API

- `GET /api/settings` → `{settings, defaults, is_default (per key), vetted_models,
  locked_protocol, admin_required}`
- `POST /api/settings` (admin) → partial update, validated; 400 with reason on rejection
- `POST /api/settings/reset` (admin) → restore defaults

### 4.6 UI

New full-width "Settings (admin)" card in `static/index.html`: model dropdown,
prompt-template textarea, extra-instructions textarea, read-only locked-protocol viewer,
admin-password field (only when required), per-field **Default/Modified** badges,
Save / Restore Defaults.

## 5. Baseline-preservation guarantees

1. Defaults registry == today's values; absent `settings.json` ⇒ default path.
2. `test_settings.py` asserts the built default prompt is **byte-identical** to the
   v0.1.0 hardcoded prompt.
3. Dispatch default branch is the same code path (same model string, same request shape).
4. Existing unit + loopback test suites keep running unchanged as the regression net.
5. Locked protocol is not reachable from any user input.

## 6. Phase plan & workflow

- **Phase 1** (branch `phase-1-settings-module`): everything above.
- **Phase 2** (branch after Phase 1 merges): STT adapter (Deepgram Nova; Whisper via
  OpenRouter stays default) + TTS adapter (OpenAI TTS; Cartesia stays default), provider
  stamp in per-call usage records, pricing table for non-OpenRouter providers.
  Keys already in `.env`: `DEEPGRAM_API_KEY`, `OPENAI_API_KEY`.
- **Phase 3** (optional, on explicit request only): telephony abstraction — Twilio-compatible
  providers (SignalWire/Telnyx TeXML) first; Indian SIP trunk (Exotel) out of scope.

Per phase: implement on branch → automated tests pass → manual test round → **merge only
on explicit approval** → next branch. `v0.1.0` tag is never modified.

## 7. Provider prerequisites (Phases 2–3 reference)

| Provider | Category | Credentials | Notes |
|---|---|---|---|
| Deepgram Nova | STT | `DEEPGRAM_API_KEY` (present) | Accepts 8kHz μ-law/WAV; fast, telephony-tuned |
| OpenAI (whisper / gpt-4o-mini-tts) | STT / TTS | `OPENAI_API_KEY` (present) | TTS returns 24kHz PCM → existing resample path |
| ElevenLabs | TTS | API key (not yet in `.env`) | Native `ulaw_8000` output; add only if benchmarking demands |
| SignalWire | Telephony | Project ID, token, Space URL, number | Twilio-compatible (LAML + media streams) |
| Telnyx | Telephony | API key, TeXML app, number | Twilio-compat layer; cheaper minutes |
| Exotel | Telephony (India) | Account SID, key/token, ExoPhone, KYC/DLT | Out of scope per project decision |

## 8. Risks

| Risk | Mitigation |
|---|---|
| Prompt edit breaks action contract | Locked protocol block + save validation + runtime fallback |
| Latency regression from model choice | Defaults pinned; per-turn latency already logged; vetted list only |
| Mid-call settings change alters an active call's next turn | Accepted for v1 (admin-only, rare); documented |
| Cost display wrong for non-OpenRouter models | OpenRouter returns real `usage.cost` for all vetted models — unaffected in Phase 1 |
| `settings.json` corruption | Load falls back to defaults + logs; atomic writes |
