"""make_call.py - build a FICTIONAL two-speaker sales call with espeak-ng.
Writes call.wav (clean), call_noisy.wav (added noise) and turns.json
(ground-truth speaker turns, used as a stand-in for diarization)."""
import json, subprocess, wave
import numpy as np

TURNS = [
 ("SARAH", "en-us+f3", "Hello Markus, this is Sarah from Alpenline. How can I help?"),
 ("MARKUS", "en-gb+m3", "Hi Sarah. We need twenty five hundred STM32 controllers for the new board."),
 ("MARKUS", "en-gb+m3", "The production run starts in three weeks, so delivery is critical."),
 ("SARAH", "en-us+f3", "Understood. I will check the supplier and send you a date by Thursday."),
 ("MARKUS", "en-gb+m3", "Good. Price is less important this time, but please do not be late again."),
 ("SARAH", "en-us+f3", "I will also look for an alternate part in case stock is short."),
 ("MARKUS", "en-gb+m3", "Perfect. You can reach me on my mobile, zero six six zero one two three four."),
]
SR = 16000
def tts(voice, text, path):
    subprocess.run(["espeak-ng", "-v", voice, "-s", "155", "-w", path, text], check=True)
    with wave.open(path) as w:
        sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    # resample to 16 kHz by linear interpolation (espeak writes 22.05 kHz)
    t = np.linspace(0, len(x) / sr, int(len(x) * SR / sr), endpoint=False)
    return np.interp(t, np.arange(len(x)) / sr, x).astype(np.float32) / 32768

audio, turns, t0 = [], [], 0.0
gap = np.zeros(int(0.4 * SR), np.float32)
for spk, voice, text in TURNS:
    y = tts(voice, text, "/tmp/_turn.wav")
    turns.append({"speaker": spk, "start": round(t0, 2),
                  "end": round(t0 + len(y) / SR, 2), "text": text})
    audio += [y, gap]; t0 += (len(y) + len(gap)) / SR
x = np.concatenate(audio)

def write(path, sig):
    s = np.clip(sig, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((s * 32767).astype(np.int16).tobytes())
write("call.wav", x)
rng = np.random.default_rng(31)
p_sig = np.mean(x[np.abs(x) > 0.01] ** 2)
for snr in (10, 0):
    noise = rng.normal(0, np.sqrt(p_sig / 10 ** (snr / 10)), len(x)).astype(np.float32)
    write(f"call_snr{snr}.wav", x + noise)
json.dump(turns, open("turns.json", "w"), indent=1)
print(f"duration {len(x)/SR:.1f} s, {len(turns)} turns")
