"""extract.py - email intelligence pipeline (teaching).
1) PII scrub  2) LLM extraction (mocked by a lexicon baseline so it runs
offline)  3) schema validation  4) evaluation vs human labels.
In production step 2 calls the private LLM with the prompt below.
"""
import re
from collections import Counter
from typing import Literal
from pydantic import BaseModel, Field
from emails import EMAILS

class EmailSignals(BaseModel):
    urgency: int = Field(ge=0, le=3)
    sensitivity: Literal["price", "lead_time", "both", "none"]
    tone: Literal["pos", "neu", "neg"]
    competitor: bool
    evidence: list[str] = Field(max_length=5)   # quoted spans from the email

PROMPT = """Extract sales signals from the customer email (EN or DE).
Return JSON matching the schema. Use the rubric:
urgency 0=no time pressure, 1=normal request, 2=explicit deadline,
3=production at risk / "urgent" / line stop.
sensitivity: what the customer stresses: price, lead_time, both, none.
tone: pos / neu / neg toward us. competitor: another supplier mentioned.
evidence: up to 5 SHORT verbatim quotes that justify the labels.
Do not guess: if not stated, choose the lower urgency and "none".
EMAIL:
{email}"""

# ---------- 1) PII scrubbing before anything is stored or logged ----------
PII = [(r"[\w.+-]+@[\w-]+\.[\w.]+", "<EMAIL>"),
       (r"\+?\d[\d\s/-]{7,}\d", "<PHONE>")]
def scrub(text: str) -> str:
    for pat, rep in PII:
        text = re.sub(pat, rep, text)
    return text

# ---------- 2) offline baseline that mimics the LLM's JSON output ----------
LEX = {
 "urg3": [r"\burgent\b", r"\bdringend\b", r"line (stop|steht)|linie steht",
          r"\basap\b", r"production .* risk", r"delivery is critical"],
 "urg2": [r"\bby (friday|monday|end of month|tomorrow)\b", r"bis morgen",
          r"decide by", r"\bdeadline\b", r"need delivery by"],
 "urg0": [r"no rush", r"planning for", r"prototype"],
 "price": [r"zu teuer", r"too expensive", r"cheaper", r"günstiger",
           r"zielpreis|target price", r"best price", r"match\?", r"preis war höher"],
 "lead": [r"delivery", r"lieferung", r"lead time", r"\blate\b", r"zu spät",
          r"brauchen die ware", r"confirm a date"],
 "neg": [r"you were late", r"enttäuscht", r"zu spät", r"leider", r"steht"],
 "pos": [r"great job", r"thanks for the quick", r"on time"],
 "comp": [r"competitor", r"anderer lieferant", r"other supplier"],
 "price_low": [r"preis ist zweitrangig", r"price is ok"],
}
def hits(key, t):
    return [m.group(0) for p in LEX[key] for m in [re.search(p, t)] if m]

def baseline_extract(text: str) -> dict:
    t = text.lower()
    ev = []
    if hits("urg3", t): urgency = 3; ev += hits("urg3", t)
    elif hits("urg2", t): urgency = 2; ev += hits("urg2", t)
    elif hits("urg0", t): urgency = 0; ev += hits("urg0", t)
    else: urgency = 1
    p = bool(hits("price", t)) and not hits("price_low", t)
    if hits("price", t) and hits("price_low", t):
        p = "best price" in t            # "price is OK, but best price..."
    l = bool(hits("lead", t)) and not ("same lead time" in t)
    sens = {(True, True): "both", (True, False): "price",
            (False, True): "lead_time", (False, False): "none"}[(p, l)]
    if p: ev += hits("price", t)
    if l: ev += hits("lead", t)
    tone = "neg" if hits("neg", t) else "pos" if hits("pos", t) else "neu"
    return {"urgency": urgency, "sensitivity": sens, "tone": tone,
            "competitor": bool(hits("comp", t)), "evidence": ev[:5]}

# ---------- 3+4) validate and evaluate ----------
if __name__ == "__main__":
    print(scrub(EMAILS[0]["text"]))
    print()
    res = Counter()
    conf = Counter()
    for e in EMAILS:
        out = EmailSignals(**baseline_extract(scrub(e["text"])))  # validates
        lab = e["label"]
        for k in ("urgency", "sensitivity", "tone", "competitor"):
            res[k] += int(getattr(out, k) == lab[k])
        conf[(lab["urgency"], out.urgency)] += 1
        flag = "" if out.sensitivity == lab["sensitivity"] else "  <-- sens"
        print(f"{e['id']} urg={out.urgency} (gold {lab['urgency']}) "
              f"sens={out.sensitivity:9s} tone={out.tone} "
              f"comp={out.competitor}{flag}")
    n = len(EMAILS)
    print("\naccuracy vs human labels:",
          {k: f"{v}/{n}" for k, v in res.items()})
    print("urgency confusion (gold, predicted):", dict(sorted(conf.items())))
    print("\nexample JSON:", EmailSignals(**baseline_extract(
        scrub(EMAILS[1]["text"]))).model_dump_json())

def evidence_ok(sig: EmailSignals, email: str) -> bool:
    # every evidence span must literally occur in the (scrubbed) email
    low = email.lower()
    return all(e.lower() in low for e in sig.evidence)
