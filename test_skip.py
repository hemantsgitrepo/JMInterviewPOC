"""Skip-intent loopback check (live STT/LLM/TTS — needs all API keys):
1. "I don't know" on Q1 must get encouragement, NOT a skip or an advance to Q2.
2. An explicit skip request must lead (possibly via a confirm step) to the question being
   skipped: recorded in the skips list and the interview advancing to Q2.
3. Remaining questions proceed normally to the goodbye.

Uses the same 3-question config as test_last_question.py.
Usage: uvicorn app:app --port 8010, then python test_skip.py
"""

import asyncio
import base64
import json

import httpx
import websockets

import models

BASE = "http://localhost:8010"
SILENT = base64.b64encode(b"\xff" * 160).decode()
Q2_CORE = "programming languages"
Q3_CORE = "challenging technical problem"


async def say(ws, text):
    ulaw = await models.speak(text)
    for i in range(0, len(ulaw), 160):
        await ws.send(json.dumps({"event": "media", "media": {"payload": base64.b64encode(ulaw[i:i+160]).decode()}}))
    for _ in range(45):  # 900ms of silence so the endpointer closes the utterance
        await ws.send(json.dumps({"event": "media", "media": {"payload": SILENT}}))


async def next_agent_line(ws, http, cid):
    """Echo marks until the agent's turn finishes, then return its latest transcript line."""
    while True:
        msg = json.loads(await ws.recv())
        if msg["event"] != "mark":
            continue
        name = msg["mark"]["name"]
        await ws.send(json.dumps({"event": "mark", "mark": {"name": name}}))
        if name in ("turn_done", "end"):
            turns = (await http.get(f"/api/candidates/{cid}/transcript")).json()["turns"]
            agent = [t["text"] for t in turns if t["role"] == "agent"]
            return agent[-1], name


async def main():
    async with httpx.AsyncClient(base_url=BASE) as http:
        cfg = (await http.get("/api/config")).json()
        assert len(cfg["questions"]) == 3, "server config must have the 3 regression questions"
        await http.post("/api/candidates", json={"candidates": [{"name": "Skip Tester", "phone": "+15550006666"}]})
        cid = (await http.get("/api/session")).json()["candidates"][0]["id"]

        async with websockets.connect(BASE.replace("http", "ws") + "/twilio/media") as ws:
            await ws.send(json.dumps({"event": "start", "start": {"streamSid": "SKIP", "customParameters": {"candidate_id": cid}}}))
            line, _ = await next_agent_line(ws, http, cid)   # opening
            await say(ws, "Yes, I'm ready.")
            line, _ = await next_agent_line(ws, http, cid)   # Q1 asked

            # 1) "I don't know" must NOT advance or skip — expect encouragement, still on Q1
            await say(ws, "I don't know.")
            line, _ = await next_agent_line(ws, http, cid)
            assert Q2_CORE not in line.lower(), f"'I don't know' must not advance to Q2; agent said: {line!r}"
            print(f"[idk -> encouragement] {line}")

            # 2) explicit skip request; agent may skip directly or ask to confirm first
            await say(ws, "Honestly, please just skip this question, I'd rather not answer it.")
            line, _ = await next_agent_line(ws, http, cid)
            if Q2_CORE not in line.lower():   # it asked for confirmation
                print(f"[skip -> confirm offer] {line}")
                await say(ws, "Yes, skip it please.")
                line, _ = await next_agent_line(ws, http, cid)
            assert Q2_CORE in line.lower(), f"after confirmed skip the agent must ask Q2; said: {line!r}"
            print(f"[skipped -> Q2] {line}")

            # 3) finish normally
            await say(ws, "I'm most comfortable with Python which I've used daily for five years along with Go for the last two.")
            line, mark = await next_agent_line(ws, http, cid)
            assert Q3_CORE in line.lower(), f"expected Q3; agent said: {line!r}"
            await say(ws, "We had a race condition in the billing worker that double charged customers and I fixed it with idempotency keys after tracing the requests.")
            line, mark = await next_agent_line(ws, http, cid)
            assert mark == "end" and "Goodbye" in line, f"expected goodbye, got mark={mark!r}: {line!r}"

    record = None
    for _ in range(20):  # finish() writes the file after ws teardown + hangup grace
        try:
            with open(f"transcripts/{cid}.json") as f:
                record = json.load(f)
            break
        except FileNotFoundError:
            await asyncio.sleep(0.5)
    assert record is not None, "transcript JSON was never written"
    skips = record["skips"]
    print(f"[skips recorded] {skips}")
    assert len(skips) == 1, f"exactly one skip expected, got {skips!r}"
    assert skips[0]["index"] == 0 and "recent role" in skips[0]["question"]
    print("\n'I don't know' encouraged, explicit skip recorded + advanced, interview completed — OK")


if __name__ == "__main__":
    asyncio.run(main())
