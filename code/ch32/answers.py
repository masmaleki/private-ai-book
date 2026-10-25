"""answers.py - runnable answers for chapter 31 (fictional data)."""
import json
from wer import wer
from call_to_crm import assign_speakers

# 31-2: WER by hand, checked by code
ref = "please send the new quote for five thousand pieces before friday noon"
hyp = "please send a new quote for five thousand piece before friday"
w, S, D, I, N, ops = wer(ref, hyp, numbers=False)
print(f"31-2 N={N} S={S} D={D} I={I} WER={w:.3f}",
      [o for o in ops if o[0] != "="])

# 31-4: compute and storage estimate for 40 calls/day, 6 min each
calls, minutes, rtf = 40, 6, 0.07          # rtf of 'small' measured in this chapter
audio_min = calls * minutes
print(f"31-4 audio/day={audio_min} min, CPU time/day={audio_min*rtf:.1f} min, "
      f"WAV 16kHz/16bit/mono per call={16000*2*60*minutes/1e6:.1f} MB, "
      f"per day={16000*2*60*audio_min/1e6:.0f} MB")

# 31-7: WER per speaker on the noisy (10 dB) small transcript
turns = json.load(open("turns.json"))
for f in ["out_small_call.json", "out_small_call_snr10.json"]:
    segs = assign_speakers(json.load(open(f))["segments"], turns)
    res = {}
    for spk in ("SARAH", "MARKUS"):
        r = " ".join(t["text"] for t in turns if t["speaker"] == spk)
        h = " ".join(s["text"] for s in segs if s["speaker"] == spk)
        res[spk] = round(wer(r, h)[0], 3)
    print("31-7", f, res)

# 31-5: entity-level checks on top of validate()
from call_to_crm import validate, mock_llm_note, scrub
def validate_v2(note, lines, min_overlap=0.6):
    transcript = " ".join(l["text"] for l in lines)
    errs = validate(note, transcript)
    low = transcript.lower()
    uncertain = " ".join(l["text"].lower() for l in lines if l["overlap"] < min_overlap)
    for c in note["commitments"]:
        if c["due"] and c["due"].lower() not in low:
            errs.append(f"commitment due {c['due']!r} not in transcript")
        if c["quote"].lower() in uncertain:
            errs.append(f"commitment from UNCERTAIN segment: {c['quote']!r}")
    return errs
for f in ["out_small_call.json", "out_small_call_snr10.json"]:
    lines = assign_speakers(json.load(open(f))["segments"], turns)
    for l in lines: l["text"] = scrub(l["text"])
    e = validate_v2(mock_llm_note(lines), lines)
    print("31-5", f, len(e), [x for x in e if "due" in x or "UNCERTAIN" in x])
