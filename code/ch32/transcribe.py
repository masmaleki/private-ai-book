"""transcribe.py - local transcription with faster-whisper (CPU, int8).
Usage: python transcribe.py [model_size] [wav]
Audio never leaves the machine; the model weights are downloaded once
(in an air-gapped site, copy them in and point to a local directory)."""
import json, sys, time, wave
import numpy as np
from faster_whisper import WhisperModel

size = sys.argv[1] if len(sys.argv) > 1 else "base"
wav = sys.argv[2] if len(sys.argv) > 2 else "call.wav"
model = WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=4)
t = time.time()
with wave.open(wav) as w:      # 16 kHz mono PCM -> float32 array
    audio = np.frombuffer(w.readframes(w.getnframes()), np.int16) / 32768.0
segments, info = model.transcribe(audio.astype(np.float32), language="en", beam_size=5,
                                  vad_filter=True, word_timestamps=False)
segs = [{"start": round(s.start, 2), "end": round(s.end, 2),
         "text": s.text.strip()} for s in segments]   # generator -> list
dt = time.time() - t
print(f"{size} on {wav}: {dt:.1f} s for {info.duration:.1f} s audio "
      f"(real-time factor {dt/info.duration:.2f})")
for s in segs:
    print(f"[{s['start']:5.1f}-{s['end']:5.1f}] {s['text']}")
json.dump({"model": size, "wav": wav, "seconds": round(dt, 1),
           "duration": round(info.duration, 1), "segments": segs},
          open(f"out_{size}_{wav.replace('.wav','')}.json", "w"), indent=1)
