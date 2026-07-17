"""Unit check for the early-end confirmation guard: the agent must never hang up on the same
turn it proposes ending — it delivers the offer, keeps listening, and only ends after an
explicit second end_call. Natural completion (ask_next running past the last question) still
ends without an extra confirm. No server / API keys needed.

Regression case at the bottom is the real live failure: the agent asked "Would you like to
finish, or wrap up now?" AND played the closing line in the same turn, while on the last
question. Usage: python test_end_confirm.py
"""

import json

from call import CallSession

QS = ["q1", "q2", "q3", "q4", "q5"]
END_LINE = "Thanks, goodbye!"


def fresh(question_index):
    s = CallSession(ws=None)
    s.config = {"questions": QS, "end_call_line": END_LINE}
    s.question_index = question_index
    s.history = [{"role": "assistant", "content": json.dumps({"reply": "prev", "action": "stay"})}]
    s._last_hist_idx = 0
    return s


# --- early end while questions remain: FIRST end_call only offers, never hangs up ---
s = fresh(question_index=1)  # on q2, three questions still ahead
clauses, mark = s._plan_turn({"action": "end_call", "reply": "We have three left. || Wrap up or keep going?"})
assert mark == "turn_done", f"early end must keep listening, got mark {mark!r}"
assert s.awaiting_end_confirm is True and s.ending is False, "should be awaiting confirmation, not ending"
assert not any(END_LINE in c for c in clauses), "closing line must NOT play on the offer turn"
# history was rewritten so the model doesn't think it already ended
assert json.loads(s.history[0]["content"])["action"] == "stay"

# --- candidate confirms (second end_call): NOW it ends ---
clauses, mark = s._plan_turn({"action": "end_call", "reply": "Understood."})
assert mark == "end" and s.ending is True, "confirmed early end must actually end"
assert any(END_LINE in c for c in clauses), "closing line must play once confirmed"

# --- candidate declines instead (ask_next): offer is cleared, interview continues ---
s = fresh(question_index=1)
s._plan_turn({"action": "end_call", "reply": "Wrap up?"})          # offer
clauses, mark = s._plan_turn({"action": "ask_next", "reply": "Great, next one then."})
assert s.awaiting_end_confirm is False and mark == "turn_done", "declining must resume normally"
assert s.ending is False

# --- an end_call whose reply has no question still gets a real question appended ---
s = fresh(question_index=1)
clauses, mark = s._plan_turn({"action": "end_call", "reply": "Okay, I'll stop there."})
assert mark == "turn_done" and any("?" in c for c in clauses), "a bare end announcement must still ask"

# --- natural completion (last question answered -> ask_next) ends without extra confirm ---
s = fresh(question_index=len(QS) - 1)  # on the last question
clauses, mark = s._plan_turn({"action": "ask_next", "reply": "Thanks for that."})
assert mark == "end" and s.ending is True, "finishing the last question should end normally"

# --- direct end_call on the last question, reply is a STATEMENT -> genuine completion, ends ---
s = fresh(question_index=len(QS) - 1)
clauses, mark = s._plan_turn({"action": "end_call", "reply": "That's everything I needed."})
assert mark == "end" and s.ending is True, "a decided end on the last question needs no confirmation"

# --- REGRESSION (the live failure): asking a question while ending, on the LAST question.
# Previously the on_last_or_past exemption let this hang up mid-question. It must not. ---
s = fresh(question_index=len(QS) - 1)
clauses, mark = s._plan_turn({
    "action": "end_call",
    "reply": "No problem at all. || We have just one question left. Would you like to finish, or wrap up now?",
})
assert mark == "turn_done", "asking a question must never hang up, even on the last question"
assert s.ending is False, "must not be ending while awaiting the answer"
assert not any(END_LINE in c for c in clauses), "closing line must not ride along with a question"
assert s.awaiting_end_confirm is True
# and their answer ("yes, wrap up") then ends it for real
clauses, mark = s._plan_turn({"action": "end_call", "reply": "Of course."})
assert mark == "end" and any(END_LINE in c for c in clauses), "confirmation should end the call"

print("OK: no hangup on a question turn; offers-then-confirms; genuine completion ends directly.")
