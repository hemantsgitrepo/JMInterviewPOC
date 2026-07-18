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
| 4 | Answer candidate questions from the uploaded JD | **Accepted — fold into Phase 2.** Store the full JD text (today only the first 500 chars survive, `ai_usage.jd_text`), inject it as a context block in the system prompt, and soften the "never discuss role details" conduct rule to "answer from the JD when it covers it, else deflect." Cost: ~1–2k extra prompt tokens/turn. |
| 5 | Visible "JD uploaded" indicator | **Accepted — fold into Phase 2** with #4 (same surface): a status chip near the JD box once a JD is on file, cleared by Reset All. |
| 6 | Remove separate Save buttons now that Start Calling auto-saves | **Product decision — recommendation:** keep **Save Candidates** (it validates E.164 numbers and refreshes the status table *before* you're mid-dial); **Save Config** is now near-redundant and can be removed or demoted to a hint. Awaiting owner's call; no code until then. |
