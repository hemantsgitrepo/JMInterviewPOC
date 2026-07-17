"""Regression test for the "call ends before the last question is answered" bug:
question_index used to drift one turn ahead of the question actually spoken, so the
end-of-interview check could fire while asking the last question instead of after it
was answered. Runs the loopback flow with exactly N configured questions and N canned
answers, then asserts every question got its own answer turn before the goodbye line.

Usage: uvicorn app:app --port 8010, with config set to the 3 questions below,
then   python test_last_question.py
"""

import asyncio
import base64
import json

import httpx
import websockets

import models

BASE = "http://localhost:8010"
QUESTIONS = [
    "Can you briefly describe your most recent role and responsibilities?",
    "What programming languages are you most comfortable with?",
    "Tell me about a challenging technical problem you solved recently.",
]
# Answers are deliberately complete (the agent now probes genuinely thin answers, by design)
# and single-sentence (a TTS inter-sentence pause > 700ms would split one answer into two
# utterances and desync this test's fixed one-answer-per-question script).
ANSWERS = [
    "Yes, I'm ready to begin.",
    "I was a backend engineer at a fintech startup for three years where I owned the billing services and mentored two junior developers.",
    "I'm most comfortable with Python which I've used daily for five years along with Go for the last two.",
    "We had a race condition in the billing worker that double charged customers and I fixed it with idempotency keys after tracing the requests.",
]
# Distinctive core of each question — the agent asks the configured wording but may vary the
# lead-in modality ("Can you" -> "could you"), so match on the part that identifies the question.
QUESTION_CORES = [
    "your most recent role and responsibilities",
    "programming languages are you most comfortable with",
    "challenging technical problem you solved recently",
]
SILENT = base64.b64encode(b"\xff" * 160).decode()


async def main():
    async with httpx.AsyncClient(base_url=BASE) as http:
        cfg = (await http.get("/api/config")).json()
        assert cfg["questions"] == QUESTIONS, "server config must match this test's QUESTIONS list"
        await http.post("/api/candidates", json={"candidates": [{"name": "Last-Q Tester", "phone": "+15550004444"}]})
        cid = (await http.get("/api/session")).json()["candidates"][0]["id"]

    async with websockets.connect(BASE.replace("http", "ws") + "/twilio/media") as ws:
        await ws.send(json.dumps({"event": "start", "start": {"streamSid": "LASTQ", "customParameters": {"candidate_id": cid}}}))
        answer = iter(ANSWERS)
        while True:
            msg = json.loads(await ws.recv())
            if msg["event"] != "mark":
                continue
            name = msg["mark"]["name"]
            await ws.send(json.dumps({"event": "mark", "mark": {"name": name}}))
            if name == "end":
                break
            if name != "turn_done":
                continue
            text = next(answer, None)
            if text is None:
                break
            ulaw = await models.speak(text)
            for i in range(0, len(ulaw), 160):
                await ws.send(json.dumps({"event": "media", "media": {"payload": base64.b64encode(ulaw[i:i+160]).decode()}}))
            for _ in range(45):
                await ws.send(json.dumps({"event": "media", "media": {"payload": SILENT}}))

    async with httpx.AsyncClient(base_url=BASE) as http:
        turns = (await http.get(f"/api/candidates/{cid}/transcript")).json()["turns"]

    print("--- transcript ---")
    for t in turns:
        print(f"[{t['role']}] {t['text']}")

    # Every configured question must appear in some agent turn, AND be followed by a
    # real candidate turn (not immediately by the end-call line) before the call ends.
    for i, q in enumerate(QUESTION_CORES):
        q_idx = next((j for j, t in enumerate(turns) if t["role"] == "agent" and q.lower() in t["text"].lower()), None)
        assert q_idx is not None, f"question {i+1} was never asked: {q!r}"
        after = turns[q_idx + 1:]
        assert after and after[0]["role"] == "candidate", (
            f"question {i+1} ({q!r}) was not followed by a candidate answer — "
            f"next turn was: {after[0] if after else 'END OF CALL'}"
        )

    assert turns[-1]["role"] == "agent" and "Goodbye" in turns[-1]["text"], "call must end with the configured end line"
    print(f"\nAll {len(QUESTIONS)} questions were asked and answered before the goodbye — OK")


if __name__ == "__main__":
    asyncio.run(main())
