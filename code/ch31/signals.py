"""signals.py - combine text signals with purchase behaviour, then draft
a reply (mock LLM) and check it. All data FICTIONAL."""
import math, re

# Prior from behaviour (ch29): share of past quotes where this customer
# chose the cheaper of two offers. Text evidence updates it in log-odds.
LR = {"price": 2.0, "both": 1.3, "lead_time": 0.5, "none": 1.0}  # likelihood ratios

def update(prior: float, sens_seq: list[str]) -> list[float]:
    lo = math.log(prior / (1 - prior))
    out = [prior]
    for s in sens_seq:
        lo += math.log(LR[s])
        lo = max(min(lo, 3.0), -3.0)          # cap: no single source dominates
        out.append(1 / (1 + math.exp(-lo)))
    return out

def mock_llm_reply(signals: dict, facts: dict) -> str:
    # stands in for the private LLM; it may only use numbers from `facts`
    lines = [f"Dear {facts['contact']},", "",
             f"thank you for your message about {facts['mpn']}."]
    if signals["sensitivity"] in ("lead_time", "both"):
        lines.append(f"Our current estimate is {facts['lt_p50']}-{facts['lt_p90']} "
                     "days; we will confirm a firm date after checking with the supplier.")
    if signals["sensitivity"] in ("price", "both"):
        lines.append("We are reviewing the price and will send an updated offer.")
    lines += ["", "Best regards,", "Sarah"]
    return "\n".join(lines)

BANNED = [r"\bguarantee", r"\bpromise", r"\bgarantier", r"\bversprech"]
def check_reply(text: str, facts: dict) -> list[str]:
    bad = [b for b in BANNED if re.search(b, text.lower())]
    allowed = {str(v) for v in facts.values()}
    nums = re.findall(r"\d+(?:[.,]\d+)?", text)
    bad += [n for n in nums if n not in allowed and not any(n in a for a in allowed)]
    return bad

if __name__ == "__main__":
    seq = ["price", "price", "both", "price"]
    print("NordSensor P(price-focused):", [round(p, 3) for p in update(0.55, seq)])
    seq2 = ["lead_time", "lead_time", "none"]
    print("AlpenBoard P(price-focused):", [round(p, 3) for p in update(0.30, seq2)])
    facts = {"contact": "Markus", "mpn": "STM32F407VGT6", "lt_p50": 31, "lt_p90": 38}
    sig = {"sensitivity": "lead_time", "urgency": 3}
    r = mock_llm_reply(sig, facts)
    print(r)
    print("checks:", check_reply(r, facts) or "OK")
    bad = r.replace("Our current estimate is 31-38 days",
                    "We guarantee delivery in 21 days")
    print("bad reply checks:", check_reply(bad, facts))
