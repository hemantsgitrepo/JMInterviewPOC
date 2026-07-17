"""Tiered-silence loopback check: a silent candidate should get, in order, a gentle
reassurance ("Take your time."), then an offer to repeat/move on, then the disconnected
goodbye — instead of the old single "Are you still there?" then hangup.

Usage: uvicorn app:app --port 8010, then python test_tiered_silence.py
(Needs CARTESIA_API_KEY for the spoken prompts; no STT/LLM calls — silence never transcribes.)
"""

import asyncio
import base64
import json

import httpx
import websockets

import store

BASE = "http://localhost:8010"
SILENT = base64.b64encode(b"\xff" * 160).decode()
B = store.config["behavior"]
# frames needed per tier, plus margin (server counts 20ms frames of consecutive silence)
T1 = B["silence_tier1_ms"] // 20 + 20
T2 = (B["silence_tier2_ms"] - B["silence_tier1_ms"]) // 20 + 20
T3 = B["silence_tier2_ms"] // 20 + 20


async def send_silence(ws, frames):
    for _ in range(frames):
        await ws.send(json.dumps({"event": "media", "media": {"payload": SILENT}}))


async def wait_for_mark(ws, names):
    """Echo every mark back (as Twilio would after playing it); return when one in names arrives."""
    while True:
        msg = json.loads(await ws.recv())
        if msg["event"] != "mark":
            continue
        name = msg["mark"]["name"]
        await ws.send(json.dumps({"event": "mark", "mark": {"name": name}}))
        if name in names:
            return name


async def main():
    async with httpx.AsyncClient(base_url=BASE) as http:
        await http.post("/api/candidates", json={"candidates": [{"name": "Silent Tester", "phone": "+15550005555"}]})
        cid = (await http.get("/api/session")).json()["candidates"][0]["id"]

    async with websockets.connect(BASE.replace("http", "ws") + "/twilio/media") as ws:
        await ws.send(json.dumps({"event": "start", "start": {"streamSid": "SILENCE", "customParameters": {"candidate_id": cid}}}))
        await wait_for_mark(ws, {"turn_done"})   # opening line finished -> LISTENING
        for frames in (T1, T2, T3):
            await send_silence(ws, frames)
            mark = await wait_for_mark(ws, {"turn_done", "end"})
        assert mark == "end", f"third silence tier should end the call, got mark {mark!r}"

    async with httpx.AsyncClient(base_url=BASE) as http:
        turns = (await http.get(f"/api/candidates/{cid}/transcript")).json()["turns"]
    agent_lines = [t["text"] for t in turns if t["role"] == "agent"]
    print("--- agent lines ---")
    for line in agent_lines:
        print(line)

    assert any("Take your time" in l for l in agent_lines), "tier 1 reassurance missing"
    assert any("repeat the question" in l for l in agent_lines), "tier 2 repeat/move-on offer missing"
    assert "disconnected" in agent_lines[-1] and "Goodbye" in agent_lines[-1], "final tier should say disconnected + end line"
    t1 = agent_lines.index(next(l for l in agent_lines if "Take your time" in l))
    t2 = agent_lines.index(next(l for l in agent_lines if "repeat the question" in l))
    assert t1 < t2 < len(agent_lines) - 1 or t2 == len(agent_lines) - 2, "tiers out of order"
    print("\nSilence escalated through reassure -> offer -> disconnect, in order — OK")


if __name__ == "__main__":
    asyncio.run(main())
