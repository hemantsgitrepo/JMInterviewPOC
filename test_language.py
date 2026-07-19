"""Hindi-mode conversation-engine checks (no server / API keys needed):

- Unicode normalization: a pure-Devanagari answer must classify as "answer" — before
  the fix it normalized to empty and classified "reprompt", an infinite loop.
- Hindi filler-only utterances -> "wait"; Hindi Whisper hallucinations -> "ignore".
- ends_midthought: Hindi postposition/conjunction-final = mid-thought; a complete
  verb-final sentence is not.
- LINES table: same keys in every language; line() switches with the setting.
- _plan_turn: in Hindi mode a "repeat" action must NOT replay the configured English
  question verbatim (it becomes a stay; the LLM restates in Hindi per the directive).

Usage: python test_language.py
"""

import json
import os

os.environ["SETTINGS_PATH"] = "settings_test_language.json"

import settings
from call import LINES, CallSession, classify_utterance, ends_midthought, line
import store


def cleanup():
    settings.reset()
    for p in ("settings_test_language.json",):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


BEHAVIOR = store.config["behavior"]

# --- classification on Devanagari text ---------------------------------------
# real Hindi answer -> answer (the Unicode-normalization fix; "reprompt" here = loop bug)
assert classify_utterance("मुझे पाँच साल का अनुभव है", 500, BEHAVIOR) == "answer"
# Hinglish answer -> answer
assert classify_utterance("मैंने Python पर काम किया है", 500, BEHAVIOR) == "answer"
# Hindi filler-only -> wait (they're thinking)
assert classify_utterance("उम्म हम्म", 500, BEHAVIOR) == "wait"
assert classify_utterance("मतलब", 500, BEHAVIOR) == "wait"
# Hindi Whisper hallucinations -> ignore
assert classify_utterance("धन्यवाद", 500, BEHAVIOR) == "ignore"
assert classify_utterance("कृपया सब्सक्राइब करें", 500, BEHAVIOR) == "ignore"
# नमस्ते is a real greeting, never filtered
assert classify_utterance("नमस्ते", 500, BEHAVIOR) == "answer"

# --- mid-thought detection (Hindi is verb-final) ------------------------------
for frag in ("मैंने तीन साल काम किया और", "मेरा अनुभव मुख्य रूप से बैकएंड में",
             "मैं बताना चाहता हूँ कि", "उसके बाद मैंने"):
    assert ends_midthought(frag), f"should be mid-thought: {frag!r}"
for done in ("मुझे पाँच साल का अनुभव है", "मैंने बिलिंग सिस्टम पर काम किया था",
             "जी हाँ बिल्कुल"):
    assert not ends_midthought(done), f"should be complete: {done!r}"

# --- LINES completeness + switching ------------------------------------------
assert set(LINES["hi"]) == set(LINES["en"]), "every system line needs a Hindi translation"
cleanup()
assert line("reassure") == LINES["en"]["reassure"]
settings.update({"language": "hi"})
assert line("reassure") == LINES["hi"]["reassure"]
assert line("reprompt") == LINES["hi"]["reprompt"]

# --- repeat action never replays the English question in Hindi mode -----------
QS = ["Tell me about your last role?", "q2"]
s = CallSession(ws=None)
s.config = {"questions": QS, "end_call_line": "Goodbye!", "jd_text": ""}
s.question_index = 0
s.history = [{"role": "assistant", "content": json.dumps({"reply": "prev", "action": "stay"})}]
s._last_hist_idx = 0
clauses, mark = s._plan_turn({"action": "repeat", "reply": "ज़रूर।"})
assert mark == "turn_done"
assert QS[0] not in clauses, "hi mode must not replay the configured English question verbatim"
# in English mode the verbatim replay behavior is unchanged
settings.update({"language": "en"})
clauses, mark = s._plan_turn({"action": "repeat", "reply": "Of course."})
assert QS[0] in clauses, "en mode must keep the verbatim question replay"

# --- filler phrases exist per language, gender-neutral hi set -----------------
assert set(store.FILLER_PHRASES) >= {"en", "hi"} and store.FILLER_PHRASES["hi"]

# --- voice-gender rule in the system prompt -----------------------------------
# the LLM must know what gender the caller HEARS so self-reference grammar matches
# (a female voice must never say "मैं समझ गया")
from call import VOICE_GENDER_RULES, VOICE_NEUTRAL_RULE
import models


def prompt_for(**kv):
    settings.update(kv)
    s2 = CallSession(ws=None)
    s2.cand = {"name": "Jane"}
    s2.config = {"company_name": "Acme", "questions": QS, "jd_text": ""}
    return s2.system_prompt()


# explicit setting wins regardless of provider
p = prompt_for(tts_provider="cartesia-sonic", tts_voice_gender="male", language="hi")
assert VOICE_GENDER_RULES["male"] in p and VOICE_GENDER_RULES["female"] not in p
p = prompt_for(tts_voice_gender="female")
assert VOICE_GENDER_RULES["female"] in p
# gender "default" resolves the provider's actual default voice: sarvam=male, cartesia=female
p = prompt_for(tts_provider="sarvam-bulbul", tts_voice_gender="default")
assert VOICE_GENDER_RULES["male"] in p
p = prompt_for(tts_provider="cartesia-sonic", tts_voice_gender="default")
assert VOICE_GENDER_RULES["female"] in p
# openai alloy is genuinely ambiguous: no gender rule; in hi, neutral guidance instead
p = prompt_for(tts_provider="openai-tts", tts_voice_gender="default", language="hi")
assert VOICE_GENDER_RULES["female"] not in p and VOICE_GENDER_RULES["male"] not in p
assert VOICE_NEUTRAL_RULE in p
p = prompt_for(language="en")
assert VOICE_NEUTRAL_RULE not in p, "neutral guidance is Hindi-only"
assert models.agent_voice_gender() is None

cleanup()
print("OK: Devanagari classification, Hindi mid-thought detection, line table, "
      "hi repeat-as-stay, and per-language fillers all correct.")
