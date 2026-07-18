# Backlog — assessed 2026-07-18

Observations raised during Phase 1 review, triaged against the settings-module scope.
Rule of thumb applied: the Phase 1 branch must not change conversational behavior
(its baseline test pins the v0.1.0 prompt byte-for-byte), so behavior changes land
as their own changes after Phase 1 merges.

| # | Item | Disposition |
|---|---|---|
| 1 | Graceful decline: end call without the configured end line when the candidate declines to continue | **Accepted — standalone follow-up** (small). Requires extending the locked action protocol (e.g. `end_call` with a `declined` reason that suppresses the closing line) plus a `_plan_turn()` branch. Intentionally updates the prompt-baseline test. Not Phase 1: it changes conversation behavior. |
| 2 | 10+ s dead air at call start when the candidate answers silently | **Fixed on `hotfix-async-amd`.** `dialer.py` uses synchronous AMD (`machine_detection="Enable"` without `async_amd`), so Twilio holds TwiML — and therefore the media stream and opening line — until machine-detection concludes; a silent answerer is AMD's worst case. Fix: `async_amd=True` (+ AMD result via the status callback, which `on_status_callback` already handles at dialer.py:167-177). Trade-off: voicemail hears a few seconds of the opening line before the machine verdict hangs up — acceptable. ~2-line change + one real-call verification. |
| 3 | Offer "any questions for us?" at interview close | **Zero-code today:** add it as the final configured interview question — the existing flow fields it naturally (conduct rules answer or deflect to the recruiting team). First-class closing-Q&A stage in the state machine: deferred; pairs with #4. |
| 4 | Answer candidate questions from the uploaded JD | **Done in Phase 2.** Full validated JD stored and injected as a bounded (4k-char) prompt context block; salary/benefits still deflect to the recruiting team. No JD on file ⇒ prompt stays byte-identical to baseline. |
| 5 | Visible "JD uploaded" indicator | **Done in Phase 2.** Status line under the JD box once a validated JD is on file; cleared by Reset All. |
| 6 | Remove separate Save buttons now that Start Calling auto-saves | **Done in Phase 2** (owner approved): Save Config removed, Save Candidates kept for early E.164 validation. |

Deferred open item: pin the Deepgram Nova 3 per-minute rate (its API doesn't return
per-request cost) so the UI can estimate STT cost like it does for Twilio/Cartesia —
until then Deepgram STT cost shows n/a.
