"""toy_orchestrator.py - Chapter 4: route intents to stub tools (no network).
Teaching toy: rule-based intent detection stands in for the LLM.
All data is fictional example data.
"""
import textwrap
from dataclasses import dataclass, field

# ---- stub tools (block 3 and block 4); real ones would call services ----
def get_part_offers(tenant, mpn):
    data = {"STM32F407VGT6": [("Supplier A", 12000, 7.95, 21),
                              ("Supplier B", 3000, 7.40, 98)]}
    return data.get(mpn, [])

def predict_price(tenant, mpn, qty):
    # pretend model output: P10 / P50 / P90 in EUR (example numbers)
    return {"p10": 7.30, "p50": 7.85, "p90": 8.60, "model": "price-v3"}

def clean_bom_line(tenant, line):
    return {"mpn": line.strip().upper().replace(" ", ""), "confidence": 0.93}

TOOLS = {"get_part_offers": get_part_offers, "predict_price": predict_price,
         "clean_bom_line": clean_bom_line}

# ---- policy (block 4): which tenant may use which tool ----
POLICY = {"alpenboard": {"get_part_offers", "predict_price", "clean_bom_line"},
          "tenant-d": {"clean_bom_line"}}   # air-gapped: no market tools

@dataclass
class Request:
    tenant: str            # comes from the auth token, never from the text
    user: str
    text: str
    audit: list = field(default_factory=list)

def detect_intent(text):
    """Stand-in for the LLM: classify the request into an intent."""
    t = text.lower()
    if "explain" in t and "price" in t:
        return "explain_price"
    if "bom" in t:
        return "clean_bom"
    return "unknown"

def call_tool(req, name, *args):
    allowed = name in POLICY.get(req.tenant, set())
    req.audit.append((req.user, req.tenant, name, "ALLOW" if allowed else "DENY"))
    if not allowed:
        raise PermissionError(f"{name} not allowed for {req.tenant}")
    return TOOLS[name](req.tenant, *args)

def handle(req):
    intent = detect_intent(req.text)
    try:
        if intent == "explain_price":
            offers = call_tool(req, "get_part_offers", "STM32F407VGT6")
            pred = call_tool(req, "predict_price", "STM32F407VGT6", 2500)
            best = min(offers, key=lambda o: o[3])   # fastest lead time
            return (f"[draft for {req.user}] Suggested range "
                    f"{pred['p10']:.2f}-{pred['p90']:.2f} EUR (median "
                    f"{pred['p50']:.2f}, {pred['model']}). Fastest source: "
                    f"{best[0]}, {best[3]} days.")
        if intent == "clean_bom":
            res = call_tool(req, "clean_bom_line", " stm32f407vgt6 ")
            return f"[draft] cleaned MPN {res['mpn']} (conf {res['confidence']})"
        return "Sorry, I can't help with that yet."
    except PermissionError as e:
        return f"[blocked] {e}"

if __name__ == "__main__":
    for r in [Request("alpenboard", "sara", "Explain this price for STM32"),
              Request("alpenboard", "reza", "Clean this BOM line"),
              Request("tenant-d", "eng7", "Explain this price for STM32"),
              Request("tenant-d", "eng7", "Clean this BOM line")]:
        reply = handle(r)
        print(f"{r.tenant} | {r.text}")
        print(textwrap.fill(reply, 72, initial_indent="  -> ",
                            subsequent_indent="     "))
        for entry in r.audit:
            print("     audit:", entry)
