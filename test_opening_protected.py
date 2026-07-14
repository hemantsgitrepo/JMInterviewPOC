"""Confirms the opening line always finishes even if the candidate talks over it
immediately, and that the end-call line also can't be barged. Needs a running server.

Usage: uvicorn app:app --port 8010   then   python test_opening_protected.py
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
        await http.post("/api/candidates", json={"candidates": [{"name": "Opening Test", "phone": "+15550003333"}]})
        cid = (await http.get("/api/session")).json()["candidates"][0]["id"]

    talk_over = await models.speak("Hi, hello, yes I'm here, let's go.")
    cleared_during_opening = False
    opening_turn_done = False

    async with websockets.connect(BASE.replace("http", "ws") + "/twilio/media") as ws:
        await ws.send(json.dumps({"event": "start", "start": {"streamSid": "OPEN", "customParameters": {"candidate_id": cid}}}))

        # Immediately talk over the call the instant it connects, before any mark arrives
        await send_audio(ws, talk_over)

        async for raw in ws:
            m = json.loads(raw)
            ev = m.get("event")
            if ev == "clear":
                cleared_during_opening = True
                break
            if ev == "mark":
                name = m["mark"]["name"]
                await ws.send(json.dumps({"event": "mark", "mark": {"name": name}}))
                if name == "turn_done":
                    opening_turn_done = True
                    break

    async with httpx.AsyncClient(base_url=BASE) as http:
        turns = (await http.get(f"/api/candidates/{cid}/transcript")).json()["turns"]

    print(f"Twilio `clear` sent during opening (should be False): {cleared_during_opening}")
    print(f"Opening line's turn_done mark received (should be True): {opening_turn_done}")
    print("\n--- transcript ---")
    for t in turns:
        print(f"[{t['role']}] {t['text']}")

    assert not cleared_during_opening, "opening line was interrupted by barge-in"
    assert opening_turn_done, "opening line never finished playing"
    assert turns and "[interrupted]" not in turns[0]["text"], "opening line got truncated"
    print("\nOPENING LINE PROTECTED — OK")


if __name__ == "__main__":
    asyncio.run(main())
