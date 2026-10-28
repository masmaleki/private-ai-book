"""qty.py - deterministic quantity parsing for DE/EN number styles (chapter 21)."""
import re

def parse_qty(text: str) -> int | None:
    """'2.500 Stk' -> 2500, '2,500 pcs' -> 2500, '10k' -> 10000, '1.5k' -> 1500."""
    t = text.strip().lower().replace(" ", "")
    m = re.match(r"^(\d+(?:[.,]\d+)?)k", t)
    if m:                                        # 10k, 1.5k, 2,5k
        return int(round(float(m.group(1).replace(",", ".")) * 1000))
    m = re.match(r"^(\d{1,3}(?:[.,]\d{3})+|\d+)", t)
    if not m:
        return None
    return int(re.sub(r"[.,]", "", m.group(1)))  # thousands separators only

if __name__ == "__main__":
    for s in ["2.500 Stk.", "2,500 pcs", "10k", "1.5k", "400", "12.000", "abc"]:
        print(f"{s!r:>14} -> {parse_qty(s)}")
