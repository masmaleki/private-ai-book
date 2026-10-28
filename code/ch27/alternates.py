"""alternates.py - parametric alternate finder for buck regulators (teaching).
All parts below are FICTIONAL ("Example Semiconductors" etc.). example data.
Steps: 1) hard filters (must-pass)  2) weighted soft score  3) evidence table
       4) build a grounded prompt for the private LLM (no LLM call here).
"""
from dataclasses import dataclass, asdict
import json

@dataclass
class Part:
    mpn: str
    vin_max: float        # V, max input voltage
    iout_max: float       # A, max output current
    fsw_khz: int          # switching frequency
    package: str
    pinout: str           # pinout family id from datasheet
    temp_min: int         # deg C
    temp_max: int
    lifecycle: str        # ACTIVE / NRND / EOL
    datasheet: str        # doc id in our index (example)

# Original part requested by the customer (fictional)
ORIGINAL = Part("EXS-BUCK36-SO8", 36, 3.0, 500, "SOIC-8-EP", "PX8-A",
                -40, 125, "NRND", "DOC-EXS-001")

# Candidate pool (fictional)
CANDIDATES = [
    Part("EXS-BUCK40-SO8", 40, 3.0, 500, "SOIC-8-EP", "PX8-A", -40, 125,
         "ACTIVE", "DOC-EXS-002"),
    Part("NVL-B3628", 36, 2.5, 570, "SOIC-8-EP", "PX8-A", -40, 125,
         "ACTIVE", "DOC-NVL-014"),
    Part("ORB-SW42A", 42, 3.5, 400, "SOIC-8-EP", "PX8-B", -40, 105,
         "ACTIVE", "DOC-ORB-007"),
    Part("KST-DC3010", 30, 3.0, 1000, "QFN-10", "QX10", -40, 85,
         "ACTIVE", "DOC-KST-003"),
]

# Hard requirements derived from the customer's design (example)
REQ = {"vin_needed": 32.0, "iout_needed": 2.5, "temp_min": -40, "temp_max": 105}

def hard_filter(p: Part, req: dict) -> list[str]:
    """Return list of violations; empty list means the part passes."""
    v = []
    if p.vin_max < req["vin_needed"]:
        v.append(f"vin_max {p.vin_max}V < needed {req['vin_needed']}V")
    if p.iout_max < req["iout_needed"]:
        v.append(f"iout_max {p.iout_max}A < needed {req['iout_needed']}A")
    if p.temp_min > req["temp_min"] or p.temp_max < req["temp_max"]:
        v.append(f"temp {p.temp_min}..{p.temp_max} not covering "
                 f"{req['temp_min']}..{req['temp_max']}")
    if p.lifecycle == "EOL":
        v.append("lifecycle EOL")
    return v

WEIGHTS = {"package": 0.30, "pinout": 0.30, "vin": 0.10,
           "iout": 0.10, "fsw": 0.15, "lifecycle": 0.05}

def soft_score(p: Part, o: Part) -> dict:
    """Similarity per attribute in [0, 1]; then weighted sum."""
    s = {
        "package": 1.0 if p.package == o.package else 0.0,
        "pinout": 1.0 if p.pinout == o.pinout else 0.0,
        # headroom is good, but closer to original is "more drop-in"
        "vin": max(0.0, 1 - abs(p.vin_max - o.vin_max) / o.vin_max),
        "iout": max(0.0, 1 - abs(p.iout_max - o.iout_max) / o.iout_max),
        "fsw": max(0.0, 1 - abs(p.fsw_khz - o.fsw_khz) / o.fsw_khz),
        "lifecycle": 1.0 if p.lifecycle == "ACTIVE" else 0.5,
    }
    s["total"] = round(sum(WEIGHTS[k] * s[k] for k in WEIGHTS), 3)
    return s

def classify(p: Part, o: Part, s: dict) -> str:
    """Alternate level. 'drop-in' still needs engineering approval!"""
    if s["package"] == 1 and s["pinout"] == 1 and p.fsw_khz == o.fsw_khz:
        return "candidate drop-in"
    if s["package"] == 1 and s["pinout"] == 1:
        return "pin-compatible (check external parts)"
    return "functional alternate (PCB change)"

def build_prompt(o: Part, rows: list[dict]) -> str:
    """Grounded prompt: the LLM may ONLY use the evidence JSON."""
    evidence = json.dumps({"original": asdict(o), "candidates": rows},
                          indent=1)
    return (
        "You are a component engineer assistant. Use ONLY the EVIDENCE.\n"
        "Never say 'compatible' unless pinout AND package are equal in the\n"
        "evidence. For every claim cite the datasheet id. List risks.\n"
        "Answer in Persian, max 120 words per candidate.\n\n"
        f"EVIDENCE:\n{evidence}\n")

if __name__ == "__main__":
    rows = []
    for c in CANDIDATES:
        viol = hard_filter(c, REQ)
        if viol:
            print(f"REJECT {c.mpn:15s} -> {'; '.join(viol)}")
            continue
        s = soft_score(c, ORIGINAL)
        rows.append({"mpn": c.mpn, "score": s["total"],
                     "level": classify(c, ORIGINAL, s),
                     "detail": {k: round(v, 2) for k, v in s.items()
                                if k != "total"},
                     "datasheet": c.datasheet})
    rows.sort(key=lambda r: r["score"], reverse=True)
    print()
    print(f"{'mpn':15s} {'score':>6s}  level")
    for r in rows:
        print(f"{r['mpn']:15s} {r['score']:6.3f}  {r['level']}")
    print()
    print(json.dumps(rows[0]["detail"]))
    prompt = build_prompt(ORIGINAL, rows)
    print(f"\nprompt length: {len(prompt)} chars")
