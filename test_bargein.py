"""Barge-in integration check: interrupt the agent mid-reply and assert the server
issues a Twilio `clear` and records the interruption. Needs a running server + creds.

Usage: uvicorn app:app --port 8010   then   python test_bargein.py
"""

import asyncio
import base64
import json

import httpx
import websockets

import models

BASE = "http://localhost:8010"
SILENT = base64.b64encode(b"\xff" * 160).decode()


async def send_audio(ws, ulaw):
    for i in range(0, len(ulaw), 160):
        await ws.send(json.dumps({"event": "media", "media": {"payload": base64.b64encode(ulaw[i:i+160]).decode()}}))


async def main():
    async with httpx.AsyncClient(base_url=BASE) as http:
        await http.post("/api/candidates", json={"candidates": [{"name": "Barge Tester", "phone": "+15550000002"}]})
        cid = (await http.get("/api/session")).json()["candidates"][0]["id"]

    interruption = await models.speak("Wait, sorry, || hold on one second.")
    answer = await models.speak("Yes, I'm ready to begin.")
    cleared = False
    interrupted_during_reply = False

    async with websockets.connect(BASE.replace("http", "ws") + "/twilio/media") as ws:
        await ws.send(json.dumps({"event": "start", "start": {"streamSid": "BARGE", "customParameters": {"candidate_id": cid}}}))

        # 1) let the opening finish, echoing marks
        async for raw in ws:
            m = json.loads(raw)
            if m.get("event") == "mark":
                await ws.send(json.dumps({"event": "mark", "mark": {"name": m["mark"]["name"]}}))
                if m["mark"]["name"] == "turn_done":
                    break

        # 2) answer the opening so the agent generates a real reply
        await send_audio(ws, answer)
        for _ in range(45):
            await ws.send(json.dumps({"event": "media", "media": {"payload": SILENT}}))

        # 3) when the agent's first clause starts playing, talk over it
        barged = False
        async for raw in ws:
            m = json.loads(raw)
            ev = m.get("event")
            if ev == "clear":
                cleared = True
                break
            if ev == "mark":
                name = m["mark"]["name"]
                await ws.send(json.dumps({"event": "mark", "mark": {"name": name}}))
                if name.startswith("c") and not barged:  # agent is mid-reply → interrupt now
                    barged = True
                    await send_audio(ws, interruption)
                elif name == "turn_done" and not barged:
                    print("agent finished before we could interrupt (reply too short); rerun")

    async with httpx.AsyncClient(base_url=BASE) as http:
        turns = (await http.get(f"/api/candidates/{cid}/transcript")).json()["turns"]
    interrupted_during_reply = any("[interrupted]" in t["text"] for t in turns if t["role"] == "agent")

    print(f"\nTwilio `clear` sent by server: {cleared}")
    print(f"Agent turn marked [interrupted]:  {interrupted_during_reply}")
    print("\n--- transcript ---")
    for t in turns:
        print(f"[{t['role']}] {t['text']}")
    assert cleared, "server did not send a clear on barge-in"
    assert interrupted_during_reply, "agent turn was not truncated to what the caller heard"
    print("\nBARGE-IN OK")


if __name__ == "__main__":
    asyncio.run(main())
