"""Smallest checks that fail if the audio path breaks: codec round-trip, resample, endpointing."""

import math
import struct

import audio


def tone_pcm(ms: int, rate: int = 8000, amp: int = 8000) -> bytes:
    n = rate * ms // 1000
    return b"".join(struct.pack("<h", int(amp * math.sin(2 * math.pi * 440 * i / rate))) for i in range(n))


SILENT_FRAME = audio.pcm_to_ulaw(b"\x00\x00" * 160)  # 20ms of silence
LOUD_FRAME = audio.pcm_to_ulaw(tone_pcm(20))


def test_ulaw_roundtrip():
    pcm = tone_pcm(100)
    back = audio.ulaw_to_pcm(audio.pcm_to_ulaw(pcm))
    assert len(back) == len(pcm)
    # mu-law is lossy; energy should survive within ~10%
    import audioop
    assert abs(audioop.rms(back, 2) - audioop.rms(pcm, 2)) / audioop.rms(pcm, 2) < 0.1


def test_resample_24k_to_8k():
    pcm = tone_pcm(100, rate=24000)
    out = audio.resample(pcm, 24000, 8000)
    assert abs(len(out) - len(pcm) // 3) <= 4


def test_wav_header():
    wav = audio.pcm_to_wav(tone_pcm(40), 8000)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"


def test_endpointer_detects_utterance():
    ep = audio.Endpointer()
    got = None
    for _ in range(50):  # 1s of speech
        assert ep.feed(LOUD_FRAME) is None
    for _ in range(40):  # 800ms silence > 700ms threshold
        got = ep.feed(SILENT_FRAME)
        if got:
            break
    assert got, "utterance should complete after trailing silence"
    assert len(got) >= 50 * 160  # contains the speech


def test_endpointer_ignores_pure_silence():
    ep = audio.Endpointer()
    assert all(ep.feed(SILENT_FRAME) is None for _ in range(200))


def test_bargein_fires_on_sustained_speech():
    bd = audio.BargeInDetector()
    # a couple loud frames then silence must NOT fire (blip rejection)
    assert bd.feed(LOUD_FRAME, audio.SPEECH_RMS) is None
    assert bd.feed(LOUD_FRAME, audio.SPEECH_RMS) is None
    assert bd.feed(SILENT_FRAME, audio.SPEECH_RMS) is None
    # sustained speech fires and returns the onset frames
    got = None
    for _ in range(audio.BARGE_ONSET_FRAMES):
        got = bd.feed(LOUD_FRAME, audio.SPEECH_RMS)
    assert got is not None and len(got) == audio.BARGE_ONSET_FRAMES


def test_split_clauses():
    from call import split_clauses
    assert split_clauses("Got it. || What languages?") == ["Got it.", "What languages?"]
    assert split_clauses("One sentence here. Another one.") == ["One sentence here.", "Another one."]
    assert split_clauses("   ") == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
