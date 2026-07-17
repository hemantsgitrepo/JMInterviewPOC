"""Unit checks for the pre-LLM utterance guards (no server or API keys needed):
filler-only -> silent wait, Whisper hallucinations -> silent discard, empty-but-audible
-> reprompt, real answers pass through, and the widened action contract parses.

Usage: python test_guards.py
"""

import store
from call import classify_utterance, ends_midthought, split_clauses
from models import _parse_json

B = store.config["behavior"]
LOUD, QUIET = B["low_volume_rms"] + 100, B["low_volume_rms"] - 100

# real answers pass through untouched
assert classify_utterance("I was a backend engineer for three years.", LOUD, B) == "answer"
assert classify_utterance("Mostly Python and Go.", LOUD, B) == "answer"
assert classify_utterance("Um, mostly Python and Go, I think.", LOUD, B) == "answer"  # filler + content = answer
assert classify_utterance("No.", LOUD, B) == "answer"  # terse but real; the LLM probes, we don't block

# filler-only -> wait silently (they're thinking)
assert classify_utterance("Um...", LOUD, B) == "wait"
assert classify_utterance("uh, umm, hmm", LOUD, B) == "wait"
assert classify_utterance("Hmm.", QUIET, B) == "wait"

# Whisper hallucinations on noise -> ignore silently (must NOT reach the LLM as "end the call")
assert classify_utterance("Thank you.", LOUD, B) == "ignore"
assert classify_utterance("Bye", LOUD, B) == "ignore"
assert classify_utterance("Thanks for watching!", QUIET, B) == "ignore"
assert classify_utterance("Bye everyone, thanks for having me!", LOUD, B) == "answer"  # real sentence, not a lone artifact

# nothing transcribed: audible speech -> reprompt, near-silence -> ignore
assert classify_utterance("", LOUD, B) == "reprompt"
assert classify_utterance("", QUIET, B) == "ignore"
assert classify_utterance("...", QUIET, B) == "ignore"

# widened action contract: new verbs parse, unknown still falls back to "stay"
assert _parse_json('{"reply": "Of course.", "action": "repeat"}')["action"] == "repeat"
skip = _parse_json('{"reply": "No problem.", "action": "skip", "reason": "no Go experience"}')
assert skip["action"] == "skip" and skip["reason"] == "no Go experience"
assert _parse_json('{"reply": "x", "action": "go_back"}')["action"] == "stay"

# mid-thought detection: trailing connective/article = keep listening; complete = respond
assert ends_midthought("Mostly Python and")           # trailing conjunction
assert ends_midthought("I worked on the billing, so")  # trailing "so"
assert ends_midthought("It was really about the")      # trailing article
assert not ends_midthought("Mostly Python and Go.")    # complete
assert not ends_midthought("I was a backend engineer for three years.")
assert not ends_midthought("")                          # empty is not a continuation

# clause splitting still behaves for the repeat path (lead-in + verbatim question)
assert split_clauses("Of course. || Sure thing.")[:1] == ["Of course."]

print("OK: guards classify correctly, action contract widened, fallbacks intact.")
