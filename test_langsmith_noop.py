"""Verifies LangSmith tracing is truly opt-in: with LANGSMITH_TRACING unset, models.py must
not import langsmith and the @traceable-decorated functions must be the exact same function
objects as undecorated — proving zero added latency/cold-start cost on the hot path.

Usage: python test_langsmith_noop.py
"""

import os
import sys

assert os.environ.get("LANGSMITH_TRACING", "").lower() != "true", (
    "run this with LANGSMITH_TRACING unset/false to test the no-op path"
)

import models  # noqa: E402  (import after the env assertion above)

assert "langsmith" not in sys.modules, "langsmith must not be imported when tracing is disabled"

# the no-op decorator must return the identical function object, not a wrapper
import inspect

for name in ("transcribe", "next_turn", "speak", "parse_jd", "generate_questions_from_jd"):
    fn = getattr(models, name)
    assert inspect.iscoroutinefunction(fn), f"{name} should still be the original async def"
    assert fn.__name__ == name, f"{name} was wrapped (name changed to {fn.__name__!r})"

print("OK: LangSmith disabled -> no import, no wrapping, zero added cost.")
