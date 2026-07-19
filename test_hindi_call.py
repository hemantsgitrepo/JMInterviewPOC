"""Loopback E2E for a Hindi interview (needs the dev server on :8010 and real
STT/LLM/TTS keys; ~60-90s). Configures a 2-question Hindi interview over the DEFAULT
provider pair (Whisper + Cartesia) with language=hi, drives a full call over the
media websocket with Hindi spoken answers, then restores the previous server config
and settings exactly.

Asserts: the agent conducts the interview in Devanagari, every question gets a
candidate answer turn (transcribed as Devanagari), and the call ends with the
configured Hindi closing line.

Usage: uvicorn app:app --port 8010, then python test_hindi_call.py
"""

import asyncio
import base64
import json
import os
import re

import httpx
import websockets
from dotenv import load_dotenv

load_dotenv()

import models
import settings as settings_mod

BASE = "http://localhost:8010"
ADMIN = {"X-Admin-Pass": os.environ.get("ADMIN_PASS", "")}
DEVANAGARI = re.compile(r"[ऀ-ॿ]")

CONFIG = {
    "opening_line": "नमस्ते, यह जॉबमंच एआई की ओर से एक इंटरव्यू कॉल है। यह कॉल रिकॉर्ड की जा सकती है। क्या आप शुरू करने के लिए तैयार हैं?",
    "company_name": "Acme Corp",
    "role_name": "Software Engineer",
    "questions": [
        "अपने पिछले रोल और ज़िम्मेदारियों के बारे में बताइए।",
        "आप कौन सी प्रोग्रामिंग languages में सबसे comfortable हैं?",
    ],
    "end_call_line": "आपके समय के लिए धन्यवाद। हमारी टीम जल्द ही आपसे संपर्क करेगी। अलविदा!",
}
ANSWERS = [
    "जी हाँ, मैं तैयार हूँ।",
    "मैं तीन साल से एक fintech कंपनी में backend engineer हूँ और billing सिस्टम की ज़िम्मेदारी मेरी थी।",
    "मैं Python में सबसे comfortable हूँ और पिछले दो साल से Go भी इस्तेमाल कर रहा हूँ।",
]
SILENT = base64.b64encode(b"\xff" * 160).decode()
SETTINGS_KEYS = ("llm_model", "stt_provider", "tts_provider", "tts_voice_gender", "language", "extra_instructions")


async def main():
    async with httpx.AsyncClient(base_url=BASE) as http:
        saved_cfg = (await http.get("/api/config")).json()
        saved_settings = (await http.get("/api/settings")).json()["settings"]
        try:
            r = await http.post("/api/settings", json={
                "stt_provider": "openrouter-whisper", "tts_provider": "cartesia-sonic",
                "tts_voice_gender": "default", "language": "hi", "extra_instructions": "",
            }, headers=ADMIN)
            assert r.status_code == 200, f"settings switch failed: {r.text}"
            r = await http.post("/api/config", json=CONFIG)
            assert r.status_code == 200, f"config save failed: {r.text}"
            await http.post("/api/candidates", json={"candidates": [{"name": "Hindi Tester", "phone": "+15550005555"}]})
            cid = (await http.get("/api/session")).json()["candidates"][-1]["id"]

            async with websockets.connect(BASE.replace("http", "ws") + "/twilio/media") as ws:
                await ws.send(json.dumps({"event": "start", "start": {"streamSid": "HINDI", "customParameters": {"candidate_id": cid}}}))
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

            turns = (await http.get(f"/api/candidates/{cid}/transcript")).json()["turns"]
        finally:
            # restore the server exactly as found (settings first: config POST is open,
            # settings POST needs the admin header)
            await http.post("/api/settings", json={k: saved_settings[k] for k in SETTINGS_KEYS}, headers=ADMIN)
            await http.post("/api/config", json={k: saved_cfg[k] for k in
                            ("opening_line", "company_name", "role_name", "questions", "end_call_line")})

    print("--- transcript ---")
    for t in turns:
        print(f"[{t['role']}] {t['text']}")

    agent_turns = [t for t in turns if t["role"] == "agent"]
    cand_turns = [t for t in turns if t["role"] == "candidate"]
    assert len(cand_turns) >= len(ANSWERS), f"expected >= {len(ANSWERS)} candidate turns, got {len(cand_turns)}"
    for t in agent_turns:
        assert DEVANAGARI.search(t["text"]), f"agent turn not in Hindi: {t['text']!r}"
    hindi_answers = [t for t in cand_turns if DEVANAGARI.search(t["text"])]
    assert hindi_answers, "no candidate turn transcribed as Devanagari — hi STT not in effect"
    assert turns[-1]["role"] == "agent" and "अलविदा" in turns[-1]["text"], "call must end with the configured Hindi closing line"
    print(f"\nHindi interview completed: {len(agent_turns)} agent turns all Devanagari, "
          f"{len(cand_turns)} candidate turns, closed with the Hindi end line — OK")


if __name__ == "__main__":
    asyncio.run(main())
