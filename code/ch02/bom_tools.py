"""bom_tools.py - Chapter 2: load a messy BOM, normalise it, price it.
All data is fictional example data (synthetic).
"""
import re
import pandas as pd

# 1) Load the messy BOM exactly as a customer might send it
bom = pd.read_csv("messy_bom.csv", dtype=str, keep_default_na=False)

# 2) Manufacturer aliases -> canonical name (example mapping, extend over time)
MANUFACTURER_ALIASES = {
    "st micro": "STMicroelectronics",
    "stmicroelectronics": "STMicroelectronics",
    "texas instruments": "Texas Instruments",
    "texas instr.": "Texas Instruments",
    "ti": "Texas Instruments",
    "murata": "Murata",
    "murata mfg": "Murata",
    "yageo": "Yageo",
    "nexperia": "Nexperia",
    "microchip": "Microchip",
}

def normalise_manufacturer(raw: str) -> str:
    key = raw.strip().lower()
    if not key:
        return "UNKNOWN"
    return MANUFACTURER_ALIASES.get(key, raw.strip())

def normalise_mpn(raw: str) -> str:
    # upper-case, drop whitespace; keep '-', ',', '/' and '.' (they can matter)
    return re.sub(r"\s+", "", raw).upper()

def count_designators(raw: str) -> int:
    # "C1-C10" -> 10, "U2, U3" -> 2, "R2 R3" -> 2
    total = 0
    for token in re.split(r"[,\s]+", raw.strip()):
        if not token:
            continue
        m = re.fullmatch(r"([A-Za-z]+)(\d+)-([A-Za-z]+)?(\d+)", token)
        total += int(m.group(4)) - int(m.group(2)) + 1 if m else 1
    return total

bom["manufacturer_norm"] = bom["Manufacturer"].map(normalise_manufacturer)
bom["mpn_norm"] = bom["MPN"].map(normalise_mpn)
bom["qty"] = bom["Qty"].astype(int)
bom["designator_count"] = bom["Designator"].map(count_designators)
bom["dnp"] = bom["Description"].str.contains(r"\bDNP\b", case=False) | (bom["qty"] == 0)

# 3) Simple quality flags (rules, not guesses)
flags = []
for _, r in bom.iterrows():
    f = []
    if r["mpn_norm"] == "":
        f.append("missing_mpn")
    if not r["dnp"] and r["qty"] != r["designator_count"]:
        f.append("qty_vs_designator")
    if r["manufacturer_norm"] == "UNKNOWN":
        f.append("missing_manufacturer")
    flags.append(";".join(f))
bom["flags"] = flags

# 4) Merge duplicate lines (same manufacturer + MPN), skip DNP
active = bom[~bom["dnp"] & (bom["mpn_norm"] != "")]
merged = (active.groupby(["manufacturer_norm", "mpn_norm"], as_index=False)
                .agg(qty=("qty", "sum"), lines=("Line", ",".join)))

pd.set_option("display.width", 100)
print("== normalised BOM ==")
print(bom[["Line", "manufacturer_norm", "mpn_norm", "qty",
           "designator_count", "dnp", "flags"]].to_string(index=False))
print("\n== merged buy list (per board) ==")
print(merged.to_string(index=False))

# 5) Price breaks -> unit price at a quantity (example supplier offer)
PRICE_BREAKS = {  # qty threshold -> unit price EUR (fictional)
    "TPS5430DDAR": [(1, 2.10), (100, 1.62), (1000, 1.21), (2500, 1.08)],
}

def unit_price_at(breaks, qty):
    price = None
    for threshold, p in sorted(breaks):
        if qty >= threshold:
            price = p
    return price

def cheapest_buy(breaks, need, moq=1, spq=1):
    """Compare buying exactly what we need vs. jumping to the next break."""
    options = []
    for threshold, _ in sorted(breaks):
        q = max(need, threshold, moq)
        q = -(-q // spq) * spq  # round up to a full SPQ multiple
        p = unit_price_at(breaks, q)
        options.append((q, p, round(q * p, 2)))
    return sorted(set(options), key=lambda o: o[2])

boards = 400
need = int(merged.loc[merged.mpn_norm == "TPS5430DDAR", "qty"].iloc[0]) * boards
print(f"\nneed {need} pcs TPS5430DDAR for {boards} boards")
for q, p, total in cheapest_buy(PRICE_BREAKS["TPS5430DDAR"], need, spq=2500):
    print(f"  buy {q:>5} pcs @ {p:.2f} EUR -> total {total:>8.2f} EUR")
for q, p, total in cheapest_buy(PRICE_BREAKS["TPS5430DDAR"], need, spq=1):
    print(f"  (cut tape) buy {q:>5} pcs @ {p:.2f} EUR -> total {total:>8.2f} EUR")
