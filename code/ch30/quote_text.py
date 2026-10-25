"""quote_text.py - turn a quote draft JSON into an email draft.
The LLM call is mocked (offline); the checks are real. example data.
"""
import json, re

draft = json.load(open("quote_draft.json"))

PROMPT = """You write quote emails for a component distributor.
Use ONLY values from QUOTE_JSON. Tone: concise, professional, English.
The customer cares most about lead time: mention it first.
Never mention internal fields (win_prob, cost, floor, versions).
QUOTE_JSON:
{q}"""

def mock_llm(prompt: str) -> str:
    """Offline stand-in for the private LLM (see ch21 client)."""
    l = draft["lines"][0]
    return (f"Dear Markus,\n\nthank you for your RFQ. For {l['qty']} pcs "
            f"{l['mpn']} we can deliver within {l['lead_time_days_p90']} days "
            f"at {l['unit_price']:.2f} {l['currency']} per unit. "
            f"This offer is valid for {l['valid_days']} days.\n\n"
            f"Best regards,\nSarah")

INTERNAL = ["win_prob", "cost_p50", "cost_p90", "price_floor", "winprob",
            "probability", "margin"]

def check(text: str, d: dict) -> list[str]:
    probs = [f"leaks internal term: {w}" for w in INTERNAL if w in text.lower()]
    l = d["lines"][0]
    allowed = {str(l["qty"]), f"{l['unit_price']:.2f}",
               str(l["lead_time_days_p90"]), str(l["valid_days"])}
    nums = set(re.findall(r"\d+(?:\.\d+)?", text.replace(l["mpn"], "")))
    probs += [f"number not in draft: {n}" for n in nums - allowed]
    if d["status"] != "DRAFT_NEEDS_APPROVAL":
        probs.append("draft is not in approval state")
    return probs

if __name__ == "__main__":
    q = {"customer": draft["customer"], "lines": draft["lines"]}  # no internals
    text = mock_llm(PROMPT.format(q=json.dumps(q)))
    print(text)
    print("\nchecks:", check(text, draft) or "OK")
    bad = text.replace("within 38 days", "within 21 days (win_prob 0.87)")
    print("bad checks:", check(bad, draft))
