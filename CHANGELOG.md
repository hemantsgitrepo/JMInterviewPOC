# Changelog

## v0.1.0 — Conversational interview agent (POC milestone)

First tagged release of the Jobmanch.ai AI Interview Caller. The agent conducts a full outbound
phone interview end to end — dial, converse, record, transcribe, and report per-call AI cost —
and this release is the point at which it stopped behaving like a form-filling bot and started
holding a conversation.

### Why this matters

Before this release the agent treated **any** inbound sound as a final answer. A cough, a stray
"um", or a half-second pause would advance the interview. Callers were talked over, could not
skip a question they couldn't answer, and — in the worst case — background noise transcribed as
"Bye" could end the call outright. This release makes the interview robust to how people
actually speak on the phone.

### Conversational handling

- **Silent wait on filler.** A filler-only utterance ("um…", "hmm") no longer triggers a
  response. The agent stays quiet and extends its listening window, the way a human interviewer
  waits out a thinking noise.
- **Deferred acknowledgment.** "Okay"/"Right" now plays only once the agent has committed to
  answering a complete thought, so it never acknowledges over a caller who is still mid-sentence.
- **Continuation stitching.** If speech trails off on a connective ("...Python **and**"), the
  agent keeps that audio, waits, and stitches it to what comes next instead of answering a
  fragment.
- **Tiered silence.** Escalates gently: reassurance ("Take your time.") → an offer to repeat or
  move on → the disconnect line. Replaces the single abrupt "Are you still there?".
- **Skip and repeat intents.** Callers can skip a question (recorded with the reason they gave,
  as assessment signal for the employer) or ask to hear it again (replayed verbatim).
- **Interview conduct.** Encourages before offering a skip on "I don't know" rather than
  dismissing the candidate; redirects rambling at a pause; rephrases and re-asks on request;
  probes only contentless answers, once; stays neutral (no grading answers); answers honestly
  when asked whether it's an AI or recorded; defers pay/role questions to the recruiting team.

### Bug fixes

- **Noise could end a call.** Whisper hallucinations on near-silence ("Thank you.", "Bye") were
  read as an exit cue. These are now discarded silently.
- **The agent hung up mid-question.** It could ask "would you like to wrap up?" and play the
  closing line in the same breath. An early end is now treated as a *proposal*: the agent speaks
  the offer, keeps listening, and requires explicit confirmation before ending. Genuine
  completion still ends without an extra prompt.
- **System-spoken lines were invisible to the model.** Prompts like "want me to repeat the
  question?" never reached the LLM history, so the caller's "yes please" arrived without context.

### Observability

- **LangSmith tracing**, opt-in and off by default. Traces STT, LLM, and TTS calls with real
  token counts. When `LANGSMITH_TRACING` is unset the library is never imported and the
  decorators are no-ops — **zero added latency and no cold-start cost** on the call path.
  Enable with `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY`.

### Configuration

New server-side `behavior` block in `store.py` (silence tiers, filler list, skip limit,
low-volume threshold, key-fact confirmation toggle). Not part of the UI config payload, so
existing saves leave it intact. Not yet exposed in the UI.

### Tests

Loopback and unit suites, no framework: `test_guards.py`, `test_end_confirm.py`,
`test_langsmith_noop.py` (offline) and `test_skip.py`, `test_tiered_silence.py`,
`test_last_question.py` (drive the real media WebSocket).

### Known limitations

- Acoustic emotion / raised-voice detection is not feasible with this stack (Whisper discards
  prosody); emotional cues are read from wording only.
- STT confidence is not returned by the OpenRouter Whisper path — "speak louder" prompts use
  audio energy (RMS) only.
- Continuation stitching only catches pauses that trail off on a connective word. Pauses after a
  content word still rely on the 700ms endpoint window. Full coverage needs semantic endpointing.
- Cartesia does not report TTS cost, so the cost panel marks it "not reported" rather than
  estimating.
- State is in memory; transcripts and recordings persist to disk, but statuses reset on restart.
