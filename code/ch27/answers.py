"""answers.py - runnable answers for exercises 26-4 and 26-5."""
import json, re
from alternates import CANDIDATES, ORIGINAL, REQ, hard_filter, soft_score

def margin_warnings(p, req, factor=1.10):
    w = []
    if p.iout_max < factor * req["iout_needed"]:
        w.append(f"low current margin: {p.iout_max}A < "
                 f"{factor:.0%} of {req['iout_needed']}A")
    return w

for c in CANDIDATES:
    if hard_filter(c, REQ):
        continue
    s = soft_score(c, ORIGINAL)
    print(c.mpn, s["total"], margin_warnings(c, REQ))

def numbers_in(s: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", s))

def unsupported_numbers(text: str, evidence_json: str) -> set[str]:
    ev = numbers_in(evidence_json)
    return {n for n in numbers_in(text) if n not in ev}

evidence = json.dumps({"fsw_khz": 570, "iout_max": 2.5})
print(unsupported_numbers("fsw is 570 kHz, Iq is 40 uA", evidence))
