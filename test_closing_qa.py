"""Unit check for the closing Q&A gate (no server / API keys needed):

- A naturally completed interview WITH a JD on file must offer the candidate their own
  questions (turn_done, no closing line) exactly once, then end normally afterwards.
- Without a JD the flow is unchanged: natural completion ends immediately.
- A candidate who asked to end early is never trapped in Q&A (explicit end_call skips it).

Usage: python test_closing_qa.py
"""

import json
import os

os.environ["SETTINGS_PATH"] = "settings_test_closing_qa.json"  # hermetic: system lines
# and _plan_turn are language-sensitive, so never read the operator's live settings.json

from call import CallSession

QS = ["q1", "q2", "q3"]
END_LINE = "Thanks, goodbye!"
OFFER_SNIPPET = "do you have any questions for me"


def fresh(question_index, jd=""):
    s = CallSession(ws=None)
    s.config = {"questions": QS, "end_call_line": END_LINE, "jd_text": jd}
    s.question_index = question_index
    s.history = [{"role": "assistant", "content": json.dumps({"reply": "prev", "action": "stay"})}]
    s._last_hist_idx = 0
    return s


# --- with a JD: natural completion offers Q&A instead of ending --------------
s = fresh(question_index=len(QS) - 1, jd="We build billing systems in Python.")
clauses, mark = s._plan_turn({"action": "ask_next", "reply": "Thanks for that."})
assert mark == "turn_done", f"must keep listening for candidate questions, got {mark!r}"
assert s.offered_closing_questions is True and s.ending is False
assert any(OFFER_SNIPPET in c.lower() for c in clauses), f"no Q&A offer in {clauses!r}"
assert not any(END_LINE in c for c in clauses), "closing line must NOT play on the offer turn"
s.commit_pending()
assert s.question_index == len(QS), "progress must be committed past the last question"

# status line switches to closing-Q&A guidance
status = s.status_line()
assert "job description" in status.lower() and "end_call" in status

# candidate asks something -> stay keeps the Q&A going; then LLM ends -> closing line plays
clauses, mark = s._plan_turn({"action": "stay", "reply": "Good question — the role owns billing."})
assert mark == "turn_done" and s.ending is False
clauses, mark = s._plan_turn({"action": "end_call", "reply": "Thanks for asking!"})
assert mark == "end" and s.ending is True
assert any(END_LINE in c for c in clauses), "closing line must play when Q&A wraps"

# --- Q&A is offered at most once (stay overflow after the offer must end) ----
s = fresh(question_index=len(QS) - 1, jd="jd text")
s._plan_turn({"action": "ask_next", "reply": "Thanks."})
s.commit_pending()
s.followups = 99  # force the follow-up cap during Q&A
clauses, mark = s._plan_turn({"action": "stay", "reply": "Anything else?"})
assert mark == "end" and s.ending is True, "capped Q&A must close out, not loop"

# --- without a JD: natural completion ends immediately (baseline unchanged) --
s = fresh(question_index=len(QS) - 1, jd="")
clauses, mark = s._plan_turn({"action": "ask_next", "reply": "Thanks for that."})
assert mark == "end" and s.ending is True
assert not any(OFFER_SNIPPET in c.lower() for c in clauses), "no JD -> no forced Q&A"

# --- early end with a JD: candidate who wants out is not trapped in Q&A ------
s = fresh(question_index=0, jd="jd text")
s._plan_turn({"action": "end_call", "reply": "We have two left. || Wrap up or keep going?"})  # offer
assert s.awaiting_end_confirm is True
clauses, mark = s._plan_turn({"action": "end_call", "reply": "Understood."})  # confirmed
assert mark == "end" and s.ending is True
assert not any(OFFER_SNIPPET in c.lower() for c in clauses), "confirmed early end must skip Q&A"

print("OK: closing Q&A offered once with a JD, skipped without one, never traps an early end.")
