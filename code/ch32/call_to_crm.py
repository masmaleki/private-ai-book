"""call_to_crm.py - transcript -> speaker turns -> PII scrub -> CRM note.
Speaker turns come from turns.json (ground truth) as a stand-in for a
diarization model. The 'LLM' step is a mock that returns the JSON a
private LLM would; validation is real. All data FICTIONAL."""
import json, re, sys
sys.path.insert(0, "../ch30")
from extract import scrub as scrub_email

def assign_speakers(segs, turns):
    out = []
    for s in segs:
        best, ov = "UNKNOWN", 0.0
        for t in turns:
            o = min(s["end"], t["end"]) - max(s["start"], t["start"])
            if o > ov:
                best, ov = t["speaker"], o
        out.append({**s, "speaker": best, "overlap": round(ov / (s["end"] - s["start"]), 2)})
    return out

SPOKEN_DIGITS = r"\b(?:zero|nero|one|two|three|four|five|six|seven|eight|nine|\d)(?:[\s-]*(?:zero|nero|one|two|three|four|five|six|seven|eight|nine|\d+)){5,}\b"
def scrub(text):
    text = re.sub(r"\b\d{7,}\b", "<PHONE>", text)
    text = re.sub(SPOKEN_DIGITS, "<PHONE>", text, flags=re.I)
    text = re.sub(r"\bNero\w*\s+Nero\w*", "<PHONE>", text)   # ASR-mangled digits
    return scrub_email(text)

def mock_llm_note(lines):
    # what the private LLM is asked to return (see prompt in chapter 31)
    return {
      "summary": "Customer needs 2,500 STM32 controllers; run starts in 3 weeks.",
      "needs": [{"mpn_hint": "STM32", "qty": 2500, "quote": "We need 2,500 STM32 controllers"}],
      "sensitivity": "lead_time",
      "commitments": [
        {"owner": "SARAH", "action": "send delivery date", "due": "Thursday",
         "quote": "send you a date by Thursday"},
        {"owner": "SARAH", "action": "look for alternate part", "due": None,
         "quote": "look for an alternate part"}],
      "risks": [{"text": "previous late delivery", "quote": "do not be late again"}],
    }

def validate(note, transcript):
    low = transcript.lower()
    errs = []
    for k in ("needs", "commitments", "risks"):
        for item in note[k]:
            if item["quote"].lower() not in low:
                errs.append(f"{k}: quote not in transcript: {item['quote']!r}")
    for n in note["needs"]:
        if str(n["qty"]) not in re.sub(r"[,.](?=\d{3})", "", transcript):
            errs.append(f"needs: qty {n['qty']} not in transcript")
    return errs

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "out_small_call.json"
    segs = json.load(open(name))["segments"]
    turns = json.load(open("turns.json"))
    lines = assign_speakers(segs, turns)
    for l in lines:
        l["text"] = scrub(l["text"])
        print(f"{l['speaker']:7s} ({l['overlap']:.2f}) {l['text']}")
    transcript = " ".join(l["text"] for l in lines)
    note = mock_llm_note(lines)
    errs = validate(note, transcript)
    print("validation:", errs or "OK")
    json.dump({"source": name, "note": note, "errors": errs},
              open("crm_note.json", "w"), indent=1)
