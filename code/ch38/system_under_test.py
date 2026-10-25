"""system_under_test.py - a MOCK assistant so the harness runs offline.

In a real project this module calls your local LLM server + RAG + tools.
The mock is deterministic and has two deliberate bugs, so the harness
has something to catch (see chapter text).
"""
import re


def run(case: dict) -> dict:
    """Return {"text": str, "json": dict|None, "tool": dict|None,
    "citations": list[str], "refused": bool}."""
    kind, text = case["kind"], case["input"]
    out = {"text": "", "json": None, "tool": None,
           "citations": [], "refused": False}

    if kind == "extraction":
        mpn = re.search(r"[A-Z0-9][A-Z0-9\-,]{5,}", text).group(0)
        m = re.search(r"(\d[\d,]*)\s*(k)?\b", text)
        qty = int(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1)
        low = text.lower()
        prio = ("lead_time" if "week" in low or "fast" in low else
                "price" if "price" in low else "unknown")
        out["json"] = {"mpn": mpn, "qty": qty, "priority": prio}

    elif kind in ("rag", "security", "refusal"):
        ctx = case.get("context", [])
        if not ctx:
            out["text"] = "I don't have data to answer that."
            out["refused"] = True
        else:
            doc = ctx[0]
            # BUG 1 (deliberate): blindly copies the whole context,
            # including injected instructions
            out["text"] = f"{doc['text']} [{doc['doc']}]"
            out["citations"] = [doc["doc"]]

    elif kind == "tool":
        mpn = re.search(r"[A-Z0-9][A-Z0-9\-]{5,}", text).group(0)
        if "offer" in text.lower():
            out["tool"] = {"name": "get_part_offers", "args": {"mpn": mpn}}
        else:
            # BUG 2 (deliberate): forgets the quantity argument
            out["tool"] = {"name": "predict_price", "args": {"mpn": mpn}}
    return out
