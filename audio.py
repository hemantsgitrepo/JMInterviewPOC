"""Audio helpers: mu-law <-> PCM16, resampling, WAV wrap, energy-based endpointing.
Everything rides on stdlib audioop (audioop-lts on Python 3.13+)."""

import audioop
import io
import wave
from collections import deque

FRAME_MS = 20  # Twilio media frames are 20ms of 8kHz mu-law (160 bytes)

# ponytail: SPEECH_RMS is the floor; the live threshold adapts up from the measured
# noise floor (see CallSession._update_noise). Tune SPEECH_RMS if a quiet line clips.
SPEECH_RMS = 400
# ponytail: 700ms. Raising this to 900 was tried and reverted — it made turn-taking feel
# sluggish (every turn waits this long before the agent reacts). Mid-sentence pauses are
# handled instead by the mid-thought/stitch path in call.process_utterance, which stays
# silent rather than answering over the caller.
SILENCE_END_MS = 700
MAX_UTTERANCE_MS = 60_000
PREROLL_FRAMES = 5  # 100ms kept before detected onset so first phoneme isn't clipped
BARGE_ONSET_FRAMES = 6  # 120ms of sustained speech to confirm a barge-in (rejects blips)


def rms(ulaw_frame: bytes) -> int:
    return audioop.rms(audioop.ulaw2lin(ulaw_frame, 2), 2)


def ulaw_to_pcm(b: bytes) -> bytes:
    return audioop.ulaw2lin(b, 2)


def pcm_to_ulaw(b: bytes) -> bytes:
    return audioop.lin2ulaw(b, 2)


def resample(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    out, _ = audioop.ratecv(pcm, 2, 1, src_rate, dst_rate, None)
    return out


def pcm_to_wav(pcm: bytes, rate: int = 8000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


class Endpointer:
    """Feed 20ms mu-law frames; returns the full mu-law utterance once the
    speaker has been silent for SILENCE_END_MS after speaking."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.buf = bytearray()
        self.speaking = False
        self.silence_ms = 0
        self.preroll = deque(maxlen=PREROLL_FRAMES)

    def feed(self, frame: bytes, threshold: int = SPEECH_RMS) -> bytes | None:
        loud = rms(frame) > threshold
        if not self.speaking:
            if not loud:
                self.preroll.append(frame)
                return None
            self.speaking = True
            for f in self.preroll:
                self.buf += f
        self.buf += frame
        self.silence_ms = 0 if loud else self.silence_ms + FRAME_MS
        duration_ms = len(self.buf) // 160 * FRAME_MS
        if self.silence_ms >= SILENCE_END_MS or duration_ms >= MAX_UTTERANCE_MS:
            utt = bytes(self.buf)
            self.reset()
            return utt
        return None


class BargeInDetector:
    """Fires when the caller speaks over the agent: BARGE_ONSET_FRAMES of
    sustained loud audio. Returns those onset frames so the endpointer can be
    seeded with them (the interruption's start isn't lost). Any quiet frame
    resets the run, so a single click or breath won't trigger it."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.onset = []

    def feed(self, frame: bytes, threshold: int) -> list[bytes] | None:
        if rms(frame) > threshold:
            self.onset.append(frame)
            if len(self.onset) >= BARGE_ONSET_FRAMES:
                frames = self.onset
                self.reset()
                return frames
        else:
            self.onset = []
        return None
