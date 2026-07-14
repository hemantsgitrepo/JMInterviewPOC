"""Loopback harness: plays the candidate against a running server — full STT->LLM->TTS
loop over the real /twilio/media WebSocket, no Twilio and no phone call.
Needs OPENROUTER_API_KEY (it uses the TTS endpoint to fake the candidate's voice).

Usage: uvicorn app:app --port 8000   then   python loopback.py
"""

import asyncio
import base64
import json

import httpx
import websockets

import models

BASE = "http://localhost:8010"
ANSWERS = [
    "Yes, I'm ready to begin.",
    "I was a backend engineer at my last company, building payment APIs for three years.",
    "Mostly Python with FastAPI, and some TypeScript on the frontend.",
    "We had a race condition in our billing worker; I tracked it down with distributed tracing and fixed it with idempotency keys.",
    "No, that covers it. Thank you.",
]
SILENT_FRAME = base64.b64encode(b"\xff" * 160).decode()  # mu-law silence


async def main():
    async with httpx.AsyncClient(base_url=BASE) as http:
        r = await http.post("/api/candidates", json={"candidates": [{"name": "Loopback Tester", "phone": "+15550000001"}]})
        r.raise_for_status()
        cid = (await http.get("/api/session")).json()["candidates"][0]["id"]

    async with websockets.connect(BASE.replace("http", "ws") + "/twilio/media") as ws:
        await ws.send(json.dumps({"event": "start", "start": {"streamSid": "LOOP", "customParameters": {"candidate_id": cid}}}))
        answer = iter(ANSWERS)
        while True:
            msg = json.loads(await ws.recv())
            if msg["event"] != "mark":
                continue  # drain agent media frames
            name = msg["mark"]["name"]
            # echo every mark back the way Twilio does after playback (drives clauses_played)
            await ws.send(json.dumps({"event": "mark", "mark": {"name": name}}))
            if name == "end":
                print("agent ended the call cleanly")
                break
            if name != "turn_done":
                continue  # clause ("c0"...) or "filler" mark — not a turn boundary
            text = next(answer, "I have nothing further to add.")
            print(f"candidate says: {text}")
            ulaw = await models.speak(text)
            for i in range(0, len(ulaw), 160):
                await ws.send(json.dumps({"event": "media", "media": {"payload": base64.b64encode(ulaw[i:i+160]).decode()}}))
            for _ in range(45):  # ~900ms trailing silence to trip the endpointer
                await ws.send(json.dumps({"event": "media", "media": {"payload": SILENT_FRAME}}))

    async with httpx.AsyncClient(base_url=BASE) as http:
        t = (await http.get(f"/api/candidates/{cid}/transcript")).json()
        print("\n--- transcript ---")
        for turn in t["turns"]:
            print(f"[{turn['role']}] {turn['text']}")


if __name__ == "__main__":
    asyncio.run(main())
