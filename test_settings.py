"""Settings module unit tests (no API keys, no network, no server needed):

1. BASELINE GUARANTEE — with default settings, build_system_prompt() output is
   byte-identical to the v0.1.0 hardcoded prompt, extracted from the immutable
   git tag itself (`git show v0.1.0:call.py`), not from a copy that could drift.
2. Persistence round-trip: update -> settings.json -> fresh load.
3. Validation: unvetted model, bad placeholders, unbalanced braces, empty template.
4. Runtime fallback: a template that formats badly must fall back to the default,
   never raise into a live call.
5. extra_instructions injection sits between template and locked protocol.
6. Admin gate: enforced only when ADMIN_PASS is set.

Usage: python test_settings.py
"""

import os
import re
import subprocess

os.environ["SETTINGS_PATH"] = "settings_test.json"  # never touch real runtime state

import settings


def cleanup():
    for p in ("settings_test.json", "settings_test.json.tmp"):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
    settings._settings = None


QS = ["Describe your last role?", "What languages do you use?", "A hard bug you fixed?"]


def numbered(qs):
    return "\n".join(f"{i+1}. {q}" for i, q in enumerate(qs))


# --- 1. baseline: byte-identical to the v0.1.0 tag ---------------------------
cleanup()
v010 = subprocess.run(
    ["git", "show", "v0.1.0:call.py"], capture_output=True, text=True, check=True
).stdout
original = re.search(r'SYSTEM_PROMPT = """(.*?)"""\n', v010, re.S).group(1)
expected = original.format(
    company_name="Acme Corp", candidate_name="Jane Doe", questions=numbered(QS)
)
built = settings.build_system_prompt("Acme Corp", "Jane Doe", QS)
assert built == expected, "default prompt is NOT byte-identical to v0.1.0"

# --- 2. persistence round-trip ----------------------------------------------
s = settings.update({"llm_model": "anthropic/claude-haiku-4.5"})
assert s["llm_model"] == "anthropic/claude-haiku-4.5"
assert os.path.exists("settings_test.json")
settings._settings = None  # force reload from disk
assert settings.get()["llm_model"] == "anthropic/claude-haiku-4.5", "reload lost the update"
assert settings.is_default() == {
    "llm_model": False, "stt_provider": True, "tts_provider": True,
    "prompt_template": True, "extra_instructions": True,
}
settings.reset()
assert not os.path.exists("settings_test.json")
assert settings.get() == settings.DEFAULT_SETTINGS

# --- 3. validation -----------------------------------------------------------
for bad_model in ("evil/nonexistent", "", "google/gemini-2.5-flash "):
    try:
        settings.update({"llm_model": bad_model})
        raise AssertionError(f"accepted unvetted model {bad_model!r}")
    except ValueError:
        pass

# provider validation: unknown id rejected; vetted id with a missing key rejected
for field, bad in (("stt_provider", "evil-stt"), ("tts_provider", "evil-tts")):
    try:
        settings.update({field: bad})
        raise AssertionError(f"accepted unvetted {field} {bad!r}")
    except ValueError:
        pass
settings.VETTED_STT_PROVIDERS.append({"id": "tmp", "label": "Tmp", "key_env": "NO_SUCH_KEY_SET"})
try:
    settings.update({"stt_provider": "tmp"})
    raise AssertionError("accepted a provider whose .env key is missing")
except ValueError as e:
    assert "NO_SUCH_KEY_SET" in str(e)
finally:
    settings.VETTED_STT_PROVIDERS.pop()
cleanup()

for bad_template, why in [
    ("a persona with no placeholders", "missing {questions}"),
    ("has {questions} and {evil}", "unknown placeholder"),
    ("unbalanced { brace with {questions}", "unbalanced braces"),
    ("   ", "empty"),
    ("json example {\"reply\": ...} with {questions}", "stray brace pair"),
]:
    err = settings.validate_template(bad_template)
    assert err, f"validate_template accepted bad input ({why})"
    try:
        settings.update({"prompt_template": bad_template})
        raise AssertionError(f"update accepted bad template ({why})")
    except ValueError:
        pass
assert settings.validate_template(settings.DEFAULT_PROMPT_TEMPLATE) is None

# --- 4. runtime fallback on a template that validates but breaks nothing -----
# Simulate corrupt stored state (e.g. hand-edited settings.json bypassing validation):
settings._settings = dict(settings.DEFAULT_SETTINGS) | {"prompt_template": "broken {nope}"}
out = settings.build_system_prompt("Acme", "Jane", QS)
assert expected.split("\n\n", 1)[0] not in ("",) and out.endswith(settings.PROMPT_PROTOCOL)
assert "You are an automated phone interviewer" in out, "fallback to default template failed"
cleanup()

# --- 5. extra_instructions placement ----------------------------------------
settings.update({"extra_instructions": "Never discuss politics."})
out = settings.build_system_prompt("Acme", "Jane", QS)
extra_pos = out.index("ADDITIONAL INSTRUCTIONS (these override any conflicting rules above):\nNever discuss politics.")
proto_pos = out.index(settings.PROMPT_PROTOCOL)
assert extra_pos < proto_pos, "extra instructions must precede the locked protocol"
assert out.endswith(settings.PROMPT_PROTOCOL), "locked protocol must be last"
# a valid custom persona keeps the protocol locked in place
settings.update({"prompt_template": "You are a friendly sales agent for {company_name}.\nScript:\n{questions}"})
out = settings.build_system_prompt("Acme", "Jane", QS)
assert out.startswith("You are a friendly sales agent for Acme.")
assert "1. Describe your last role?" in out
assert out.endswith(settings.PROMPT_PROTOCOL)
cleanup()

# --- 5b. JD context block ----------------------------------------------------
# absent JD -> byte-identical baseline (already asserted in 1); present JD -> block
# sits between the template body and the locked protocol, truncated at the cap
out = settings.build_system_prompt("Acme", "Jane", QS, job_description="We build billing systems in Python.")
assert "JOB DESCRIPTION CONTEXT:" in out
assert "We build billing systems in Python." in out
assert out.index("JOB DESCRIPTION CONTEXT:") < out.index(settings.PROMPT_PROTOCOL)
assert out.endswith(settings.PROMPT_PROTOCOL)
long_jd = "x" * (settings.JD_CONTEXT_MAX_CHARS + 500)
out = settings.build_system_prompt("Acme", "Jane", QS, job_description=long_jd)
assert "x" * settings.JD_CONTEXT_MAX_CHARS in out and long_jd not in out, "JD not truncated at cap"

# --- 6. admin gate -----------------------------------------------------------
os.environ.pop("ADMIN_PASS", None)
assert not settings.admin_required()
assert settings.check_admin("") and settings.check_admin("anything"), "unset ADMIN_PASS must be open"
os.environ["ADMIN_PASS"] = "s3cret"
assert settings.admin_required()
assert settings.check_admin("s3cret")
assert not settings.check_admin("") and not settings.check_admin("wrong")
os.environ.pop("ADMIN_PASS", None)

cleanup()
print("OK: default prompt byte-identical to v0.1.0; persistence, validation, "
      "fallback, extra-instructions placement, and admin gate all correct.")
